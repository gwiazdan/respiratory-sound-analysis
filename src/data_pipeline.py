from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
import time
import types
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol, Self, cast

import librosa
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from joblib import Parallel, delayed
from scipy.signal import resample_poly

from src.dataset import Dataset


class DataFrameTransformer(Protocol):
    """Transform one dataframe into another dataframe."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the transformed dataframe."""
        ...


SchemaTransformer = DataFrameTransformer | Callable[[pd.DataFrame], pd.DataFrame]
AudioProcessor = Callable[[np.ndarray], np.ndarray]
type Record = int | np.integer | pd.Series | dict[str, object]


@lru_cache(maxsize=16)
def _mel_filterbank(sample_rate: int, n_fft: int) -> np.ndarray:
    return librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=128,
        fmin=0.0,
        fmax=None,
        htk=False,
        norm="slaney",
        dtype=np.float32,
    )


@lru_cache(maxsize=16)
def _dct_basis(n_mfcc: int, n_mels: int) -> np.ndarray:
    mel_indices = np.arange(n_mels, dtype=np.float32)
    coefficient_indices = np.arange(n_mfcc, dtype=np.float32)[:, None]
    basis = np.cos(np.pi / n_mels * (mel_indices + 0.5) * coefficient_indices)
    basis[0] *= np.sqrt(1 / n_mels)
    basis[1:] *= np.sqrt(2 / n_mels)
    return basis.astype(np.float32, copy=False)


@lru_cache(maxsize=16)
def _fft_frequencies(sample_rate: int, n_fft: int) -> np.ndarray:
    return np.fft.rfftfreq(n_fft, 1 / sample_rate).astype(np.float32)


def _load_feature_audio(
    wav_path: str,
    target_sample_rate: int,
    audio_preprocessor: AudioProcessor | None = None,
) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != target_sample_rate:
        audio = resample_poly(audio, target_sample_rate, sample_rate)

    audio = np.asarray(audio, dtype=np.float32)
    return _apply_audio_preprocessor(audio, audio_preprocessor), target_sample_rate


def _apply_audio_preprocessor(
    audio: np.ndarray,
    audio_preprocessor: AudioProcessor | None,
) -> np.ndarray:
    if audio_preprocessor is None:
        return audio
    processed = np.asarray(audio_preprocessor(audio))
    if processed.ndim != 1:
        raise ValueError("An audio preprocessor must return a one-dimensional signal.")
    return processed


def _extract_recording_features(
    wav_path: str,
    index: object,
    feature_sample_rate: int,
    audio_preprocessor: AudioProcessor | None,
) -> dict[str, object]:
    full_audio, sample_rate = _load_feature_audio(
        wav_path,
        feature_sample_rate,
        audio_preprocessor,
    )
    if len(full_audio) < 128:
        return {"index": index}
    return _extract_audio_features(full_audio, sample_rate, index)


def _extract_audio_features(
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
            f"{prefix}_min": float(np.min(values)),
            f"{prefix}_max": float(np.max(values)),
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

    stft_mag = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)).astype(
        np.float32,
        copy=False,
    )
    frequencies = _fft_frequencies(sample_rate, n_fft)[:, None]
    magnitude_sum = np.sum(stft_mag, axis=0)
    safe_magnitude_sum = np.maximum(magnitude_sum, np.finfo(np.float32).tiny)
    centroid = np.sum(frequencies * stft_mag, axis=0) / safe_magnitude_sum
    bandwidth = np.sqrt(
        np.sum(((frequencies - centroid) ** 2) * stft_mag, axis=0) / safe_magnitude_sum
    )
    rolloff_threshold = 0.85 * magnitude_sum
    cumulative_magnitude = np.cumsum(stft_mag, axis=0)
    rolloff_bins = np.argmax(cumulative_magnitude >= rolloff_threshold, axis=0)
    rolloff = _fft_frequencies(sample_rate, n_fft)[rolloff_bins]
    safe_power = np.maximum(stft_mag**2, 1e-10)
    flatness = np.exp(np.mean(np.log(safe_power), axis=0)) / np.mean(
        safe_power,
        axis=0,
    )
    spectral_features = {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "spectral_flatness": flatness,
    }
    for name, values in spectral_features.items():
        features.update(stats(values, name))

    mel_spec = _mel_filterbank(sample_rate, n_fft) @ (stft_mag**2)
    mfcc = _dct_basis(13, mel_spec.shape[0]) @ librosa.power_to_db(mel_spec)
    for coefficient, values in enumerate(mfcc):
        features[f"mfcc_{coefficient}_mean"] = float(np.mean(values))
        features[f"mfcc_{coefficient}_std"] = float(np.std(values))
    return features


@dataclass(frozen=True)
class _DatasetSource:
    dataset: Dataset
    transformer: SchemaTransformer


class MultiDatasetPipelineBuilder:
    """Build train/test dataframes from multiple dataset sources."""

    FEATURE_SAMPLE_RATE = 4000
    _FEATURE_EXTRACTION_VERSION = 3
    _MANIFEST_NAME = "manifest.json"
    _ALL_DATASET_NAME = "all.parquet"
    _FEATURE_DATASET_NAME = "all_features.parquet"
    _CACHE_LOCK_NAME = ".cache.lock"
    _CACHE_LOCK_TIMEOUT = 30.0
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

    def __init__(
        self,
        cache_path: str | Path | None = None,
        cache_suffix: str | None = None,
    ) -> None:
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.cache_suffix = cache_suffix or ""
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
        """Build train/test dataframes in memory."""
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
                dataframe = pq.read_table(self._artifact_path(self._ALL_DATASET_NAME)).to_pandas()
                self._restore_cached_split(
                    dataframe,
                    self._ALL_DATASET_NAME,
                    split=split,
                    random_state=random_state,
                )
                return dataframe

        train_df, test_df = self.build_train_test(
            split=split,
            random_state=random_state,
            force_recompute=force_recompute,
        )
        dataframe = pd.concat([train_df, test_df], ignore_index=True)
        self._save_cached_dataframe(
            dataframe,
            self._ALL_DATASET_NAME,
            self._cache_key(split, random_state),
            metadata={"train_rows": str(len(train_df))},
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
        audio_preprocessor: AudioProcessor | None = None,
    ) -> pd.DataFrame:
        """Build the complete dataset and append cached audio features."""
        cache_key = self._feature_cache_key(split, random_state, audio_preprocessor)
        if (
            self.cache_path is not None
            and not force_recompute
            and self._cached_artifact_matches(self._FEATURE_DATASET_NAME, cache_key)
        ):
            result = pq.read_table(self._artifact_path(self._FEATURE_DATASET_NAME)).to_pandas()
            self._restore_cached_split(
                result,
                self._FEATURE_DATASET_NAME,
                split=split,
                random_state=random_state,
            )
            return result

        train_df, test_df = self.build_train_test_with_features(
            split=split,
            random_state=random_state,
            force_recompute=force_recompute,
            n_jobs=n_jobs,
            audio_preprocessor=audio_preprocessor,
        )
        result = pd.concat([train_df, test_df], ignore_index=True)
        self._save_cached_dataframe(
            result,
            self._FEATURE_DATASET_NAME,
            cache_key,
            metadata={"train_rows": str(len(train_df))},
        )
        return result

    def build_all_with_features(
        self,
        split: float = 0.7,
        random_state: int = 42,
        force_recompute: bool = False,
        n_jobs: int = -1,
        audio_preprocessor: AudioProcessor | None = None,
    ) -> pd.DataFrame:
        """Alias for :meth:`build_dataset_with_features`."""
        return self.build_dataset_with_features(
            split,
            random_state,
            force_recompute,
            n_jobs,
            audio_preprocessor,
        )

    def build_train_test_with_features(
        self,
        split: float = 0.7,
        random_state: int = 42,
        force_recompute: bool = False,
        n_jobs: int = -1,
        audio_preprocessor: AudioProcessor | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build train/test dataframes and extract features without data leakage."""
        train_df, test_df = self.build_train_test(
            split=split,
            random_state=random_state,
            force_recompute=force_recompute,
        )
        train_length = len(train_df)
        combined_features = self._extract_features(
            pd.concat([train_df, test_df], ignore_index=True),
            n_jobs=n_jobs,
            audio_preprocessor=audio_preprocessor,
        )
        train_features = combined_features.iloc[:train_length].reset_index(drop=True)
        test_features = combined_features.iloc[train_length:].reset_index(drop=True)

        self._train_df = train_features
        self._test_df = test_features
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
        audio_preprocessor: AudioProcessor | None = None,
    ) -> tuple[np.ndarray, int]:
        """Load and optionally preprocess a complete recording."""
        resolved_record = self.resolve(record, dataframe)
        wav_path = cast(str, resolved_record["wav_path"])
        target_sample_rate = sr
        if target_sample_rate is None and audio_preprocessor is not None:
            target_sample_rate = self._preprocessor_sample_rate(audio_preprocessor)
        audio, sample_rate = librosa.load(wav_path, sr=target_sample_rate)
        return _apply_audio_preprocessor(audio, audio_preprocessor), int(sample_rate)

    def load_audio_fragment(
        self,
        record: Record,
        start: float | None = None,
        end: float | None = None,
        sr: int | None = None,
        dataframe: pd.DataFrame | None = None,
        audio_preprocessor: AudioProcessor | None = None,
    ) -> tuple[np.ndarray, int]:
        """Load a recording fragment, preprocessing the full recording first when configured."""
        resolved_record = self.resolve(record, dataframe)
        wav_path = cast(str, resolved_record["wav_path"])

        raw_start = start if start is not None else resolved_record.get("start", 0.0)
        raw_start = cast(float | int, raw_start)
        resolved_start = float(raw_start) if raw_start is not None else 0.0
        raw_end = end if end is not None else resolved_record.get("end")
        raw_end = cast(float | int | None, raw_end)
        resolved_end = float(raw_end) if raw_end is not None else None
        duration = resolved_end - resolved_start if resolved_end is not None else None

        if audio_preprocessor is not None:
            processed_audio, processed_sample_rate = self.load_audio(
                resolved_record,
                sr=sr,
                audio_preprocessor=audio_preprocessor,
            )
            start_sample = int(resolved_start * processed_sample_rate)
            end_sample = (
                int(resolved_end * processed_sample_rate)
                if resolved_end is not None
                else len(processed_audio)
            )
            return processed_audio[start_sample:end_sample], processed_sample_rate

        audio, sample_rate = librosa.load(
            wav_path,
            sr=sr,
            offset=resolved_start,
            duration=duration,
        )
        return audio, int(sample_rate)

    @staticmethod
    def _preprocessor_sample_rate(audio_preprocessor: AudioProcessor) -> int | None:
        sample_rate = getattr(audio_preprocessor, "sr", None)
        return int(sample_rate) if sample_rate is not None else None

    def _combined_dataframe(self) -> pd.DataFrame:
        if self._train_df is None or self._test_df is None:
            raise RuntimeError("Build the train/test pipeline before resolving row indices.")
        return pd.concat([self._train_df, self._test_df], ignore_index=True)

    @staticmethod
    def _group_recordings(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return one data row per recording, preserving recording metadata."""
        if dataframe.empty or "wav_path" not in dataframe.columns:
            return dataframe.copy()

        anomaly_columns = [
            column
            for column in ("crackles", "wheezes", "rhonchi", "stridor")
            if column in dataframe.columns
        ]
        constant_columns = [
            column
            for column in dataframe.columns
            if column not in {"wav_path", "start", "end", *anomaly_columns}
        ]
        aggregation = {
            **dict.fromkeys(constant_columns, "first"),
            **dict.fromkeys(anomaly_columns, "max"),
        }
        recordings = dataframe.groupby("wav_path", sort=False, as_index=False).agg(aggregation)
        if "start" in dataframe.columns:
            starts = dataframe.groupby("wav_path", sort=False)["start"].first()
            recordings["start"] = recordings["wav_path"].map(starts)
        if "end" in dataframe.columns:
            ends = dataframe.groupby("wav_path", sort=False)["end"].max()
            recordings["end"] = recordings["wav_path"].map(ends)
        for column in anomaly_columns:
            recordings[column] = recordings[column].fillna(False).astype(bool)
        return cast(pd.DataFrame, recordings.reset_index(drop=True))

    def _extract_features(
        self,
        dataframe: pd.DataFrame,
        n_jobs: int,
        audio_preprocessor: AudioProcessor | None,
    ) -> pd.DataFrame:
        recordings = self._group_recordings(dataframe)
        grouped_recordings = [
            (cast(str, wav_path), int(index))
            for (wav_path, group) in recordings.groupby("wav_path", sort=False)
            for index in [group.index[0]]
        ]

        grouped_features = cast(
            list[dict[str, object]],
            Parallel(n_jobs=n_jobs)(
                delayed(_extract_recording_features)(
                    wav_path,
                    index,
                    self.FEATURE_SAMPLE_RATE,
                    audio_preprocessor,
                )
                for wav_path, index in grouped_recordings
            ),
        )
        if not grouped_features:
            return dataframe.copy()

        feature_dataframe = pd.DataFrame(grouped_features).set_index("index")
        result = recordings.copy()
        feature_columns = [column for column in feature_dataframe if column in result]
        if feature_columns:
            result = result.drop(columns=feature_columns)
        return result.join(feature_dataframe, how="left")

    def _cache_key(self, split: float, random_state: int) -> str:
        return self._hash_state(
            {
                "builder": self._builder_state(),
                "split": split,
                "random_state": random_state,
            }
        )

    def _feature_cache_key(
        self,
        split: float,
        random_state: int,
        audio_preprocessor: AudioProcessor | None,
    ) -> str:
        return self._hash_state(
            {
                "base": self._cache_key(split, random_state),
                "sample_rate": self.FEATURE_SAMPLE_RATE,
                "audio_preprocessor": self._object_state(audio_preprocessor),
                "implementation": self._FEATURE_EXTRACTION_VERSION,
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
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            contiguous = np.ascontiguousarray(value)
            return {
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
                "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
            }
        if isinstance(value, Path):
            return cls._path_state(value)
        if isinstance(value, (list, tuple)):
            return [cls._object_state(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): cls._object_state(item)
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (types.FunctionType, types.BuiltinFunctionType)):
            try:
                source = inspect.getsource(value)
            except (OSError, TypeError):
                source = None
            closure = getattr(value, "__closure__", None)
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "module": getattr(value, "__module__", None),
                "qualname": getattr(value, "__qualname__", None),
                "source_sha256": (
                    hashlib.sha256(source.encode("utf-8")).hexdigest()
                    if source is not None
                    else None
                ),
                "closure": (
                    [cls._object_state(cell.cell_contents) for cell in closure]
                    if closure is not None
                    else None
                ),
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

    @classmethod
    def _path_state(cls, path: Path) -> dict[str, object]:
        state: dict[str, object] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            return state

        if path.is_file():
            stat = path.stat()
            state["file"] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            return state

        entries = []
        for entry in sorted(path.rglob("*")):
            if not entry.is_file():
                continue
            stat = entry.stat()
            entries.append(
                {
                    "path": str(entry.relative_to(path)),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        state["files"] = entries
        return state

    @staticmethod
    def _hash_state(state: object) -> str:
        serialized = json.dumps(state, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def _manifest_path(self) -> Path | None:
        return self._artifact_path(self._MANIFEST_NAME) if self.cache_path is not None else None

    def _lock_path(self) -> Path | None:
        if self.cache_path is None:
            return None
        return self.cache_path / f"{self._CACHE_LOCK_NAME}{self.cache_suffix}"

    def _artifact_path(self, artifact: str) -> Path:
        if self.cache_path is None:
            raise RuntimeError("Cache path is not configured.")
        artifact_path = Path(artifact)
        filename = f"{artifact_path.stem}{self.cache_suffix}{artifact_path.suffix}"
        return self.cache_path / filename

    def _cached_artifact_matches(self, artifact: str, cache_key: str) -> bool:
        manifest = self._read_manifest()
        if manifest is None:
            return False
        return manifest.get(artifact) == cache_key and self._artifact_path(artifact).is_file()

    def _read_manifest(self) -> dict[str, object] | None:
        manifest_path = self._manifest_path()
        if manifest_path is None or not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError) as error:
            warnings.warn(
                f"Ignoring invalid cache manifest {manifest_path}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        if not isinstance(manifest, dict):
            warnings.warn(
                f"Ignoring invalid cache manifest {manifest_path}: expected an object",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        return manifest

    def _restore_cached_split(
        self,
        dataframe: pd.DataFrame,
        artifact: str,
        *,
        split: float,
        random_state: int,
    ) -> None:
        manifest = self._read_manifest()
        train_rows = manifest.get(f"{artifact}:train_rows") if manifest is not None else None
        if isinstance(train_rows, (str, int)):
            try:
                train_length = int(train_rows)
            except ValueError:
                train_length = None
        else:
            train_length = None

        if train_length is None or not 0 <= train_length <= len(dataframe):
            train_df, test_df = self.build_train_test(
                split=split,
                random_state=random_state,
            )
            train_length = len(train_df)
            if train_length + len(test_df) != len(dataframe):
                raise ValueError(f"Cached artifact {artifact} has an unexpected number of rows.")

        self._train_df = dataframe.iloc[:train_length].reset_index(drop=True)
        self._test_df = dataframe.iloc[train_length:].reset_index(drop=True)

    def _write_manifest(self, artifact: str, cache_key: str) -> None:
        manifest_path = self._manifest_path()
        if manifest_path is None:
            return
        with self._cache_lock():
            self._write_manifest_unlocked(artifact, cache_key)

    def _write_manifest_unlocked(
        self,
        artifact: str,
        cache_key: str,
        metadata: dict[str, str] | None = None,
    ) -> None:
        manifest_path = self._manifest_path()
        if manifest_path is None:
            return
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, str] = {}
        if manifest_path.is_file():
            try:
                loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                loaded_manifest = {}
            if isinstance(loaded_manifest, dict):
                manifest = {
                    str(key): str(value)
                    for key, value in loaded_manifest.items()
                    if isinstance(value, str)
                }
        manifest[artifact] = cache_key
        if metadata is not None:
            manifest.update({f"{artifact}:{key}": value for key, value in metadata.items()})
        self._atomic_write_text(
            manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True),
        )

    def _save_cached_dataframe(
        self,
        dataframe: pd.DataFrame,
        artifact: str,
        cache_key: str,
        metadata: dict[str, str] | None = None,
    ) -> None:
        if self.cache_path is None:
            return
        self.cache_path.mkdir(parents=True, exist_ok=True)
        artifact_path = self._artifact_path(artifact)
        with self._cache_lock():
            temporary_path = self._temporary_path(artifact_path)
            try:
                pq.write_table(
                    pa.Table.from_pandas(dataframe, preserve_index=False),
                    temporary_path,
                )
                os.replace(temporary_path, artifact_path)
                self._write_manifest_unlocked(artifact, cache_key, metadata)
            finally:
                temporary_path.unlink(missing_ok=True)

    @contextmanager
    def _cache_lock(self) -> Iterator[None]:
        lock_path = self._lock_path()
        if lock_path is None:
            yield
            return

        if self.cache_path is None:
            raise RuntimeError("Cache path is not configured.")
        self.cache_path.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self._CACHE_LOCK_TIMEOUT
        while True:
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(descriptor)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for cache lock {lock_path}.") from None
                time.sleep(0.05)

        try:
            yield
        finally:
            lock_path.unlink(missing_ok=True)

    @staticmethod
    def _temporary_path(target_path: Path) -> Path:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        return Path(temporary_name)

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        temporary_path = MultiDatasetPipelineBuilder._temporary_path(path)
        try:
            temporary_path.write_text(content, encoding="utf-8")
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

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
        merged = merged.drop(columns=columns_to_drop)
        return cls._group_recordings(merged)


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
