"""Noisy/clean training pairs for denoisers, built from the classifier's train split only.

The classifier pipeline splits every dataset into train/test (``split=0.7, random_state=42``).
A denoiser that saw test recordings during training would leak information into the
classifier's evaluation, so :func:`collect_train_recordings` reproduces the pipeline split
and returns only train recordings. A validation subset for model selection is then held out
from them by patient (:func:`split_by_group`).

Pairs are built exactly the way the denoiser is used at inference time: recordings are
loaded and resampled to the pipeline rate (4 kHz) with the pipeline's own loader, noise is
mixed at 4 kHz, and only then both signals are resampled to the model sample rate.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.signal import resample_poly
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from src.data_pipeline import MultiDatasetPipelineBuilder, _load_feature_audio
from src.dataset import Dataset, HFLungSoundDataset, ICBHIDataset, SPRSoundDataset
from src.denoising.noise import NoiseSource, mix_at_snr

log = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = MultiDatasetPipelineBuilder.FEATURE_SAMPLE_RATE
DATASET_CHOICES = ("spr", "icbhi", "hf")
MIN_RMS = 1e-4  # near-silent segments make SNR meaningless


@dataclass(frozen=True)
class Recording:
    path: Path
    group: str  # patient id where known, otherwise the recording itself
    source: str
    # recording-level annotations, used to check that clean-recording selection does not
    # remove pathological recordings disproportionately
    crackles: bool = False
    wheezes: bool = False


def _datasets(data_dir: Path, names: Sequence[str]) -> dict[str, Dataset]:
    factories = {
        "spr": lambda: SPRSoundDataset(data_dir / "SPRSound"),
        "icbhi": lambda: ICBHIDataset(
            data_dir / "ICBHI_final_database",
            data_dir / "ICBHI_final_database" / "ICBHI_Challenge_diagnosis.txt",
        ),
        "hf": lambda: HFLungSoundDataset(data_dir),
    }
    unknown = set(names) - set(factories)
    if unknown:
        raise ValueError(f"Unknown datasets {sorted(unknown)}; choose from {DATASET_CHOICES}.")
    return {name: factories[name]() for name in names}


def collect_train_recordings(
    data_dir: Path,
    datasets: Sequence[str] = DATASET_CHOICES,
    split: float = 0.7,
    random_state: int = 42,
) -> list[Recording]:
    """Recordings from the classifier *train* split (same split as the pipeline)."""
    recordings: list[Recording] = []
    for name, dataset in _datasets(data_dir, datasets).items():
        train_df, _ = dataset.get_train_test(split=split, random_state=random_state)
        anomalies = {
            column: train_df.groupby("wav_path")[column].max().astype(bool)
            for column in ("crackles", "wheezes")
            if column in train_df.columns
        }
        for row in train_df.drop_duplicates("wav_path").itertuples(index=False):
            patient = getattr(row, "patient_id", None)
            group = f"{name}:{patient}" if patient is not None else f"{name}:{row.wav_path}"
            flags = {column: bool(values[row.wav_path]) for column, values in anomalies.items()}
            recordings.append(Recording(Path(str(row.wav_path)), group, name, **flags))
        log.info("%s: %d train recordings", name, train_df["wav_path"].nunique())
    return recordings


def split_by_group(
    recordings: Sequence[Recording], val_fraction: float = 0.1, seed: int = 0
) -> tuple[list[Recording], list[Recording]]:
    """Hold out ``val_fraction`` of the groups (patients) for validation."""
    groups = sorted({rec.group for rec in recordings})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    val_groups = set(groups[: max(1, round(len(groups) * val_fraction))])
    train = [rec for rec in recordings if rec.group not in val_groups]
    val = [rec for rec in recordings if rec.group in val_groups]
    return train, val


def _cache_file(recording: Recording, cache_dir: Path) -> Path:
    digest = hashlib.sha1(str(recording.path).encode("utf-8"), usedforsecurity=False)
    return cache_dir / f"{recording.source}_{recording.path.stem}_{digest.hexdigest()[:10]}.npy"


def _cache_recording(recording: Recording, cache_dir: Path) -> None:
    target = _cache_file(recording, cache_dir)
    if target.exists():
        return
    audio, _ = _load_feature_audio(str(recording.path), PIPELINE_SAMPLE_RATE)
    temporary = target.with_suffix(".tmp.npy")
    np.save(temporary, audio.astype(np.float32))
    temporary.replace(target)


def build_cache(recordings: Sequence[Recording], cache_dir: Path, n_jobs: int = -1) -> None:
    """Resample every recording to the pipeline rate once and store it as ``.npy``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [rec for rec in recordings if not _cache_file(rec, cache_dir).exists()]
    if not missing:
        return
    Parallel(n_jobs=n_jobs)(
        delayed(_cache_recording)(rec, cache_dir)
        for rec in tqdm(missing, desc=f"Caching recordings at {PIPELINE_SAMPLE_RATE} Hz")
    )


def resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    divisor = gcd(source_sr, target_sr)
    return np.asarray(
        resample_poly(audio, target_sr // divisor, source_sr // divisor), dtype=np.float32
    )


class NoisyCleanDataset(TorchDataset[tuple[torch.Tensor, torch.Tensor]]):
    """``(clean, noisy)`` pairs of ``segment_seconds`` at ``model_sample_rate``.

    Training mode draws a fresh time shift, noise sample and SNR on every access; eval mode
    fixes them with ``seed`` so validation scores are comparable across epochs.
    """

    def __init__(
        self,
        recordings: Sequence[Recording],
        cache_dir: Path,
        noise: NoiseSource,
        snrs_db: Sequence[float],
        segment_seconds: float,
        model_sample_rate: int,
        train: bool = True,
        seed: int = 0,
        max_segments: int | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.noise = noise
        self.snrs = tuple(snrs_db)
        self.segment_len = int(segment_seconds * PIPELINE_SAMPLE_RATE)
        self.model_sample_rate = model_sample_rate
        self.train = train
        self.seed = seed
        self.files = [_cache_file(rec, cache_dir) for rec in recordings]
        self.index = self._segment_index()
        if max_segments is not None and len(self.index) > max_segments:
            picks = np.random.default_rng(seed).choice(len(self.index), max_segments, replace=False)
            self.index = [self.index[i] for i in sorted(picks)]
        self._audio: dict[int, np.ndarray] = {}

    def _segment_index(self) -> list[tuple[int, int]]:
        index: list[tuple[int, int]] = []
        for file_id, path in enumerate(self.files):
            audio = np.load(path, mmap_mode="r")
            for k in range(max(1, audio.size // self.segment_len)):
                segment = audio[k * self.segment_len : (k + 1) * self.segment_len]
                if segment.size and np.sqrt(np.mean(np.square(segment))) >= MIN_RMS:
                    index.append((file_id, k * self.segment_len))
        return index

    def __len__(self) -> int:
        return len(self.index)

    def _get_audio(self, file_id: int) -> np.ndarray:
        if file_id not in self._audio:
            self._audio[file_id] = np.load(self.files[file_id], mmap_mode="r")
        return self._audio[file_id]

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_id, start = self.index[item]
        audio = self._get_audio(file_id)
        if self.train:
            rng = np.random.default_rng()
            half = self.segment_len // 2
            start = int(
                np.clip(start + rng.integers(-half, half), 0, max(audio.size - self.segment_len, 0))
            )
        else:
            rng = np.random.default_rng(self.seed * 1_000_003 + item)

        if audio.size < self.segment_len:  # short recordings are repeat-padded
            audio = np.tile(audio, int(np.ceil(self.segment_len / max(audio.size, 1))))
            start = 0
        clean = np.array(audio[start : start + self.segment_len], dtype=np.float32)
        snr = float(self.snrs[int(rng.integers(0, len(self.snrs)))])
        noisy = mix_at_snr(clean, self.noise.sample(self.segment_len, rng), snr)

        clean = resample(clean, PIPELINE_SAMPLE_RATE, self.model_sample_rate)
        noisy = resample(noisy, PIPELINE_SAMPLE_RATE, self.model_sample_rate)
        return torch.from_numpy(clean), torch.from_numpy(noisy)


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB per row of ``(batch, samples)`` tensors."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    alpha = (estimate * reference).sum(-1, keepdim=True) / (
        reference.pow(2).sum(-1, keepdim=True) + eps
    )
    target = alpha * reference
    residual = estimate - target
    return 10 * torch.log10(target.pow(2).sum(-1) / (residual.pow(2).sum(-1) + eps) + eps)
