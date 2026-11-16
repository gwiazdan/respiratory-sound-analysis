from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self, cast

import librosa
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed

from src.dataset import Dataset


class DataFrameTransformer(Protocol):
    """Transform one dataframe into another dataframe."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the transformed dataframe."""
        ...


SchemaTransformer = DataFrameTransformer | Callable[[pd.DataFrame], pd.DataFrame]
Record = int | np.integer | pd.Series | dict[str, object]


@dataclass(frozen=True)
class _DatasetSource:
    dataset: Dataset
    transformer: SchemaTransformer


class MultiDatasetPipelineBuilder:
    """Build train/test dataframes from multiple dataset sources."""

    FEATURE_SAMPLE_RATE = 4000
    _MANIFEST_NAME = "manifest.json"
    _ALL_DATASET_NAME = "all.parquet"
    _FEATURE_DATASET_NAME = "all_features.parquet"
    _TRAIN_FEATURES_NAME = "train_features.parquet"
    _TEST_FEATURES_NAME = "test_features.parquet"
    _DROP_COLUMNS = (
        "patient_id",
        "recording_id",
        "patient_age",
        "patient_gender",
        "truncation",
        "diagnosis",
        "date",
        "split",
        "acquisition_mode",
    )

    def __init__(self, cache_path: str | Path | None = None) -> None:
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self._sources: list[_DatasetSource] = []
        self._transformers: list[DataFrameTransformer] = []
        self._train_df: pd.DataFrame | None = None
        self._test_df: pd.DataFrame | None = None

    def add_source(
        self,
        dataset: Dataset,
        schema_transformer: SchemaTransformer,
    ) -> Self:
        """Register a dataset and its source-specific schema transformer."""
        self._sources.append(_DatasetSource(dataset, schema_transformer))
        return self

    def add_transformer(self, transformer: DataFrameTransformer) -> Self:
        """Append a global dataframe transformation step."""
        self._transformers.append(transformer)
        return self

    def build_train_test(
        self,
        split: float = 0.7,
        random_state: int = 42,
        force_recompute: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build or load the cached train/test dataframes."""
        cache_key = self._cache_key(split, random_state)
        if not force_recompute and self._cache_exists(cache_key):
            return self._load_cached_train_test()

        if not self._sources:
            raise ValueError("At least one dataset source must be registered.")

        train_frames: list[pd.DataFrame] = []
        test_frames: list[pd.DataFrame] = []

        for source in self._sources:
            train_df, test_df = source.dataset.get_train_test(
                split=split,
                random_state=random_state,
            )
            train_frames.append(self._apply_schema_transformer(source.transformer, train_df))
            test_frames.append(self._apply_schema_transformer(source.transformer, test_df))

        train_df = self._merge_frames(train_frames)
        test_df = self._merge_frames(test_frames)

        for transformer in self._transformers:
            train_df = self._apply_transformer(transformer, train_df)
            test_df = self._apply_transformer(transformer, test_df)

        self._train_df = train_df
        self._test_df = test_df
        self._save_cached_train_test(train_df, test_df, cache_key)
        return train_df, test_df

    def build_dataset(
        self,
        split: float = 0.7,
        random_state: int = 42,
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """Build the complete dataset by combining train and test rows."""
        if self.cache_path is not None and not force_recompute:
            cache_key = self._cache_key(split, random_state)
            if self._cached_artifact_matches(self._ALL_DATASET_NAME, cache_key):
                return pq.read_table(self.cache_path / self._ALL_DATASET_NAME).to_pandas()

        train_df, test_df = self.build_train_test(
            split=split,
            random_state=random_state,
            force_recompute=force_recompute,
        )
        dataframe = pd.concat([train_df, test_df], ignore_index=True)
        self._save_cached_dataframe(
            dataframe, self._ALL_DATASET_NAME, self._cache_key(split, random_state)
        )
        return dataframe

    def build_all(
        self,
        split: float = 0.7,
        random_state: int = 42,
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """Alias for :meth:`build_dataset`."""
        return self.build_dataset(split, random_state, force_recompute)

    def build_dataset_with_features(
        self,
        split: float = 0.7,
        random_state: int = 42,
        force_recompute: bool = False,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """Build the complete dataset and append cached audio features."""
        cache_key = self._feature_cache_key(split, random_state, n_jobs)
        if (
            self.cache_path is not None
            and not force_recompute
            and self._cached_artifact_matches(self._FEATURE_DATASET_NAME, cache_key)
        ):
            return pq.read_table(self.cache_path / self._FEATURE_DATASET_NAME).to_pandas()

        train_df, test_df = self.build_train_test_with_features(
            split=split,
            random_state=random_state,
            force_recompute=force_recompute,
            n_jobs=n_jobs,
        )
        result = pd.concat([train_df, test_df], ignore_index=True)
        self._save_cached_dataframe(result, self._FEATURE_DATASET_NAME, cache_key)
        return result

    def build_all_with_features(
        self,
        split: float = 0.7,
        random_state: int = 42,
        force_recompute: bool = False,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """Alias for :meth:`build_dataset_with_features`."""
        return self.build_dataset_with_features(split, random_state, force_recompute, n_jobs)

    def build_train_test_with_features(
        self,
        split: float = 0.7,
        random_state: int = 42,
        force_recompute: bool = False,
        n_jobs: int = -1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build train/test dataframes and extract features without data leakage."""
        feature_cache_key = self._feature_cache_key(split, random_state, n_jobs)
        if (
            not force_recompute
            and self._cached_artifact_matches(self._TRAIN_FEATURES_NAME, feature_cache_key)
            and self._cached_artifact_matches(self._TEST_FEATURES_NAME, feature_cache_key)
        ):
            return self._load_cached_train_test_features()

        train_df, test_df = self.build_train_test(
            split=split,
            random_state=random_state,
            force_recompute=force_recompute,
        )
        train_features = self._extract_features(train_df, n_jobs=n_jobs)
        test_features = self._extract_features(test_df, n_jobs=n_jobs)

        self._train_df = train_features
        self._test_df = test_features
        self._save_cached_dataframe(
            train_features,
            self._TRAIN_FEATURES_NAME,
            feature_cache_key,
        )
        self._save_cached_dataframe(
            test_features,
            self._TEST_FEATURES_NAME,
            feature_cache_key,
        )
        return train_features, test_features

    def resolve(
        self,
        record: Record,
        dataframe: pd.DataFrame | None = None,
    ) -> pd.Series | dict[str, object]:
        """Resolve a row index to a record from a built dataframe."""
        if isinstance(record, (int, np.integer)):
            source_df = dataframe if dataframe is not None else self._combined_dataframe()
            record = source_df.iloc[int(record)]

        if isinstance(record, (pd.Series, dict)):
            return record
        raise TypeError("record must be a row index, pandas Series, or dictionary.")

    def resolve_record(
        self,
        record: Record,
        dataframe: pd.DataFrame | None = None,
    ) -> pd.Series | dict[str, object]:
        """Resolve a row index to a record from a built dataframe."""
        return self.resolve(record, dataframe)

    def load_audio(
        self,
        record: Record,
        sr: int | None = None,
        dataframe: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, int]:
        """Load a complete recording identified by a row or row index."""
        resolved_record = self.resolve(record, dataframe)
        wav_path = cast(str, resolved_record["wav_path"])
        audio, sample_rate = librosa.load(wav_path, sr=sr)
        return audio, int(sample_rate)

    def load_audio_fragment(
        self,
        record: Record,
        start: float | None = None,
        end: float | None = None,
        sr: int | None = None,
        dataframe: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, int]:
        """Load a recording fragment identified by a row and time bounds."""
        resolved_record = self.resolve(record, dataframe)
        wav_path = cast(str, resolved_record["wav_path"])

        raw_start = start if start is not None else resolved_record.get("start", 0.0)
        raw_start = cast(float | int, raw_start)
        resolved_start = float(raw_start) if raw_start is not None else 0.0
        raw_end = end if end is not None else resolved_record.get("end")
        raw_end = cast(float | int | None, raw_end)
        resolved_end = float(raw_end) if raw_end is not None else None
        duration = resolved_end - resolved_start if resolved_end is not None else None

        audio, sample_rate = librosa.load(
            wav_path,
            sr=sr,
            offset=resolved_start,
            duration=duration,
        )
        return audio, int(sample_rate)

    def _cache_exists(self, cache_key: str) -> bool:
        if self.cache_path is None:
            return False
        return self._cached_artifact_matches(
            "train.parquet", cache_key
        ) and self._cached_artifact_matches("test.parquet", cache_key)

    def _load_cached_train_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.cache_path is None:
            raise RuntimeError("Cache path is not configured.")

        train_df = pq.read_table(self.cache_path / "train.parquet").to_pandas()
        test_df = pq.read_table(self.cache_path / "test.parquet").to_pandas()
        self._train_df = train_df
        self._test_df = test_df
        return train_df, test_df

    def _load_cached_train_test_features(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.cache_path is None:
            raise RuntimeError("Cache path is not configured.")

        train_df = pq.read_table(self.cache_path / self._TRAIN_FEATURES_NAME).to_pandas()
        test_df = pq.read_table(self.cache_path / self._TEST_FEATURES_NAME).to_pandas()
        self._train_df = train_df
        self._test_df = test_df
        return train_df, test_df

    def _save_cached_train_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cache_key: str,
    ) -> None:
        if self.cache_path is None:
            return

        self.cache_path.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(train_df, preserve_index=False), self.cache_path / "train.parquet"
        )
        pq.write_table(
            pa.Table.from_pandas(test_df, preserve_index=False), self.cache_path / "test.parquet"
        )
        self._write_manifest("train.parquet", cache_key)
        self._write_manifest("test.parquet", cache_key)

    def _combined_dataframe(self) -> pd.DataFrame:
        if self._train_df is None or self._test_df is None:
            raise RuntimeError("Build the train/test pipeline before resolving row indices.")
        return pd.concat([self._train_df, self._test_df], ignore_index=True)

    def _extract_features(self, dataframe: pd.DataFrame, n_jobs: int) -> pd.DataFrame:
        grouped_cycles: list[tuple[str, list[tuple[Hashable, float, float]]]] = []
        for wav_path, group in dataframe.groupby("wav_path", sort=False):
            cycles = [
                (
                    index,
                    cast(float, row["start"]),
                    cast(float, row["end"]),
                )
                for index, row in group.iterrows()
            ]
            grouped_cycles.append((cast(str, wav_path), cycles))

        grouped_features = cast(
            list[list[dict[str, object]]],
            Parallel(n_jobs=n_jobs)(
                delayed(self._extract_file_features)(wav_path, cycles)
                for wav_path, cycles in grouped_cycles
            ),
        )
        feature_rows = [row for rows in grouped_features for row in rows]
        if not feature_rows:
            return dataframe.copy()

        feature_dataframe = pd.DataFrame(feature_rows).set_index("index")
        result = dataframe.copy()
        feature_columns = [column for column in feature_dataframe if column in result]
        if feature_columns:
            result = result.drop(columns=feature_columns)
        return result.join(feature_dataframe, how="left")

    def _extract_file_features(
        self,
        wav_path: str,
        cycles: list[tuple[object, float, float]],
    ) -> list[dict[str, object]]:
        full_audio, sample_rate = librosa.load(
            wav_path,
            sr=self.FEATURE_SAMPLE_RATE,
            mono=True,
        )
        sample_rate = int(sample_rate)
        features: list[dict[str, object]] = []
        for index, start, end in cycles:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            audio = full_audio[start_sample:end_sample]
            if len(audio) < 128:
                features.append({"index": index})
                continue
            features.append(self._extract_cycle_features(audio, sample_rate, index))
        return features

    @staticmethod
    def _extract_cycle_features(
        audio: np.ndarray,
        sample_rate: int,
        index: object,
    ) -> dict[str, object]:
        n_fft = max(64, min(2048, 2 ** int(np.floor(np.log2(len(audio))))))
        hop_length = max(1, n_fft // 4)

        def stats(values: np.ndarray, prefix: str) -> dict[str, float]:
            return {
                f"{prefix}_mean": float(np.mean(values)),
                f"{prefix}_std": float(np.std(values)),
            }

        rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=n_fft,
            hop_length=hop_length,
        )
        rms_mean = float(np.mean(rms))
        features: dict[str, object] = {
            "index": index,
            "crest_factor": float(np.max(np.abs(audio)) / (rms_mean + 1e-8)),
        }
        features.update(stats(rms, "audio_rms"))
        features.update(stats(zcr, "audio_zcr"))

        stft_mag = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        spectral_features = {
            "spectral_centroid": librosa.feature.spectral_centroid(
                S=stft_mag, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
            ),
            "spectral_bandwidth": librosa.feature.spectral_bandwidth(
                S=stft_mag, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
            ),
            "spectral_rolloff": librosa.feature.spectral_rolloff(
                S=stft_mag,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                roll_percent=0.85,
            ),
            "spectral_flatness": librosa.feature.spectral_flatness(
                S=stft_mag, n_fft=n_fft, hop_length=hop_length
            ),
        }
        for name, values in spectral_features.items():
            features.update(stats(values, name))

        mel_spec = librosa.feature.melspectrogram(
            S=stft_mag**2,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        mfcc = librosa.feature.mfcc(
            S=librosa.power_to_db(mel_spec),
            sr=sample_rate,
            n_mfcc=13,
        )
        for coefficient, values in enumerate(mfcc):
            features[f"mfcc_{coefficient}_mean"] = float(np.mean(values))
            features[f"mfcc_{coefficient}_std"] = float(np.std(values))
        return features

    def _cache_key(self, split: float, random_state: int) -> str:
        return self._hash_state(
            {
                "builder": self._builder_state(),
                "split": split,
                "random_state": random_state,
            }
        )

    def _feature_cache_key(self, split: float, random_state: int, n_jobs: int) -> str:
        return self._hash_state(
            {
                "base": self._cache_key(split, random_state),
                "sample_rate": self.FEATURE_SAMPLE_RATE,
                "n_jobs": n_jobs,
            }
        )

    def _builder_state(self) -> dict[str, object]:
        return {
            "sources": [
                {
                    "dataset": self._object_state(source.dataset),
                    "transformer": self._object_state(source.transformer),
                }
                for source in self._sources
            ],
            "transformers": [self._object_state(transformer) for transformer in self._transformers],
        }

    @classmethod
    def _object_state(cls, value: object) -> object:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [cls._object_state(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): cls._object_state(item)
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            }
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            state = {
                key: item
                for key, item in attributes.items()
                if key not in {"df", "_train_df", "_test_df"}
            }
        else:
            state = None
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "state": cls._object_state(state) if state is not None else None,
        }

    @staticmethod
    def _hash_state(state: object) -> str:
        serialized = json.dumps(state, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def _manifest_path(self) -> Path | None:
        return self.cache_path / self._MANIFEST_NAME if self.cache_path is not None else None

    def _cached_artifact_matches(self, artifact: str, cache_key: str) -> bool:
        manifest_path = self._manifest_path()
        if manifest_path is None or not manifest_path.is_file():
            return False
        cache_path = self.cache_path
        if cache_path is None:
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get(artifact) == cache_key and (cache_path / artifact).is_file()

    def _write_manifest(self, artifact: str, cache_key: str) -> None:
        manifest_path = self._manifest_path()
        if manifest_path is None:
            return
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, str] = {}
        if manifest_path.is_file():
            manifest = cast(dict[str, str], json.loads(manifest_path.read_text(encoding="utf-8")))
        manifest[artifact] = cache_key
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    def _save_cached_dataframe(
        self, dataframe: pd.DataFrame, artifact: str, cache_key: str
    ) -> None:
        if self.cache_path is None:
            return
        self.cache_path.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(dataframe, preserve_index=False),
            self.cache_path / artifact,
        )
        self._write_manifest(artifact, cache_key)

    @staticmethod
    def _apply_schema_transformer(
        transformer: SchemaTransformer,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        if callable(transformer):
            transformed = transformer(dataframe)
        else:
            transformed = transformer.transform(dataframe)
        if not isinstance(transformed, pd.DataFrame):
            raise TypeError("A schema transformer must return a pandas DataFrame.")
        return transformed

    @staticmethod
    def _apply_transformer(
        transformer: DataFrameTransformer,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        transformed = transformer.transform(dataframe)
        if not isinstance(transformed, pd.DataFrame):
            raise TypeError("A dataframe transformer must return a pandas DataFrame.")
        return transformed

    @classmethod
    def _merge_frames(cls, frames: list[pd.DataFrame]) -> pd.DataFrame:
        merged = pd.concat(frames, ignore_index=True)
        if "rhonchi" in merged.columns:
            merged["rhonchi"] = merged["rhonchi"].fillna(False).astype(int)
        columns_to_drop = [column for column in cls._DROP_COLUMNS if column in merged.columns]
        return merged.drop(columns=columns_to_drop)


def transform_spr(df: pd.DataFrame) -> pd.DataFrame:
    """Add the common source marker to an SPRSound dataframe."""
    transformed = df.copy()
    transformed["dataset_source"] = "SPRSound"
    base_columns = [
        "start",
        "end",
        "filename",
        "wav_path",
        "chest_location",
        "wheezes",
        "crackles",
        "dataset_source",
    ]
    remaining_columns = [column for column in transformed if column not in base_columns]
    return cast(pd.DataFrame, transformed[base_columns + remaining_columns])


def transform_icbhi(df: pd.DataFrame) -> pd.DataFrame:
    """Add the common source marker to an ICBHI dataframe."""
    transformed = df.copy()
    transformed["dataset_source"] = "ICBHI"
    if "chest_location" not in transformed:
        transformed["chest_location"] = None
    return transformed


def transform_hf(df: pd.DataFrame) -> pd.DataFrame:
    """Add the common source marker to an HFLung dataframe."""
    transformed = df.copy()
    transformed["dataset_source"] = "HFLung"
    base_columns = [
        "start",
        "end",
        "filename",
        "wav_path",
        "chest_location",
        "wheezes",
        "crackles",
        "dataset_source",
    ]
    remaining_columns = [column for column in transformed if column not in base_columns]
    return cast(pd.DataFrame, transformed[base_columns + remaining_columns])
