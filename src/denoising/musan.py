"""Real-world interference from MUSAN (Snyder, Chen & Povey, 2015, arXiv:1510.08484).

MUSAN contains speech (Librivox, US government recordings), music and technical/ambient noise
under Creative Commons or public-domain licences (see ``LICENSE`` in each subdirectory). Here it
complements the synthetic clinical noise with real speech, music and noises - the interference
that the CNN14 selection removes from the recordings themselves.

- :func:`list_musan` indexes the corpus (category, source, duration),
- :func:`split_musan` splits files into train/validation (disjoint files) and caps the number
  of hours per category,
- :func:`build_musan_cache` resamples the chosen files once to the pipeline rate (4 kHz) and
  stores them as float16 ``.npy`` files that DataLoader workers memory-map,
- :class:`MusanNoise` is a :class:`~src.denoising.noise.NoiseSource` drawing random crops.

**Stethoscope coupling.** Airborne sound reaches the chest piece strongly attenuated and
muffled, so speech or music recorded with a microphone is much brighter than when heard
through a stethoscope. Each crop is therefore low-pass filtered with a random cut-off
(``coupling_cutoff_hz``, default 500-1500 Hz). This is an approximation, not a measured
transfer function.

Librivox annotations do not identify readers, so the split is by file; a reader with several
files could appear in both parts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from scipy import signal
from tqdm.auto import tqdm

from src.data_pipeline import _load_feature_audio
from src.denoising.noise import MIN_NOISE_FREQ_HZ

MUSAN_CATEGORIES = ("speech", "music", "noise")
PIPELINE_SAMPLE_RATE = 4000
DEFAULT_MAX_HOURS: dict[str, float | None] = {"speech": 10.0, "music": 5.0, "noise": None}
MIN_DURATION_S = 0.5


def list_musan(root: str | Path) -> pd.DataFrame:
    """One row per MUSAN file: ``path``, ``category``, ``source``, ``duration`` (seconds)."""
    root = Path(root)
    rows = []
    for category in MUSAN_CATEGORIES:
        for path in sorted((root / category).rglob("*.wav")):
            info = sf.info(str(path))
            rows.append(
                {
                    "path": str(path),
                    "category": category,
                    "source": path.parent.name,
                    "duration": info.frames / info.samplerate,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No MUSAN recordings found in {root}.")
    frame = pd.DataFrame(rows)
    return frame[frame["duration"] >= MIN_DURATION_S].reset_index(drop=True)


def split_musan(
    files: pd.DataFrame,
    val_fraction: float = 0.2,
    max_hours: Mapping[str, float | None] = DEFAULT_MAX_HOURS,
    exclude: Sequence[str] = (),
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Disjoint train/validation files per category, capped at ``max_hours`` of training audio.

    Validation gets ``val_fraction`` of the files of every category and at most
    ``val_fraction / (1 - val_fraction)`` times the training cap. ``exclude`` lists paths to
    leave out (e.g. noise files tagged as breathing).
    """
    rng = np.random.default_rng(seed)
    excluded = set(map(str, exclude))
    train_parts, val_parts = [], []
    for category in MUSAN_CATEGORIES:
        part = files[(files["category"] == category) & ~files["path"].isin(excluded)]
        if part.empty:
            continue
        part = part.iloc[rng.permutation(len(part))]
        n_val = max(1, round(len(part) * val_fraction))
        val, train = part.iloc[:n_val], part.iloc[n_val:]
        cap = max_hours.get(category)
        if cap is not None:
            train = train[train["duration"].cumsum() <= cap * 3600]
            val_cap = cap * 3600 * val_fraction / (1 - val_fraction)
            val = val[val["duration"].cumsum() <= max(val_cap, val["duration"].iloc[0])]
        train_parts.append(train)
        val_parts.append(val)
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
    )


def _cache_file(path: str, cache_dir: Path) -> Path:
    digest = hashlib.sha1(str(Path(path).resolve()).encode("utf-8"), usedforsecurity=False)
    return cache_dir / f"{Path(path).stem}_{digest.hexdigest()[:10]}.npy"


def _cache_one(path: str, cache_dir: Path) -> None:
    target = _cache_file(path, cache_dir)
    if target.exists():
        return
    audio, _ = _load_feature_audio(path, PIPELINE_SAMPLE_RATE)
    temporary = target.with_suffix(".tmp.npy")
    np.save(temporary, audio.astype(np.float16))
    temporary.replace(target)


def build_musan_cache(files: pd.DataFrame, cache_dir: str | Path, n_jobs: int = -1) -> list[Path]:
    """Resample ``files`` to 4 kHz once (float16 ``.npy``); returns the cache paths."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = files["path"].tolist()
    missing = [path for path in paths if not _cache_file(path, cache_dir).exists()]
    if missing:
        Parallel(n_jobs=n_jobs)(
            delayed(_cache_one)(path, cache_dir)
            for path in tqdm(missing, desc=f"Caching MUSAN at {PIPELINE_SAMPLE_RATE} Hz")
        )
    return [_cache_file(path, cache_dir) for path in paths]


class MusanNoise:
    """Random crops of MUSAN recordings, one category per sample.

    ``category_weights`` sets how often each category is drawn; within a category files are
    drawn proportionally to their duration, so every second of audio is equally likely. Crops
    much quieter than their file (pauses in speech, silence) are redrawn, because normalising
    them to unit variance would turn background hiss into loud noise.
    """

    def __init__(
        self,
        files: pd.DataFrame,
        cache_dir: str | Path,
        category_weights: Mapping[str, float],
        coupling_cutoff_hz: tuple[float, float] | None = (500.0, 1500.0),
        sample_rate: int = PIPELINE_SAMPLE_RATE,
        min_relative_rms: float = 0.3,
    ) -> None:
        self.sr = sample_rate
        self.coupling_cutoff_hz = coupling_cutoff_hz
        self.min_relative_rms = min_relative_rms
        cache_dir = Path(cache_dir)
        self.categories: list[str] = []
        self.category_probabilities: list[float] = []
        self.files: dict[str, list[Path]] = {}
        self.file_probabilities: dict[str, np.ndarray] = {}
        for category, weight in category_weights.items():
            part = files[files["category"] == category]
            if weight <= 0 or part.empty:
                continue
            self.categories.append(category)
            self.category_probabilities.append(weight)
            self.files[category] = [_cache_file(path, cache_dir) for path in part["path"]]
            durations = part["duration"].to_numpy(dtype=float)
            self.file_probabilities[category] = durations / durations.sum()
        if not self.categories:
            raise ValueError("MusanNoise needs at least one category with files and weight > 0.")
        total = sum(self.category_probabilities)
        self.category_probabilities = [weight / total for weight in self.category_probabilities]
        self._audio: dict[Path, np.ndarray] = {}
        self._file_rms: dict[Path, float] = {}

    def _load(self, path: Path) -> np.ndarray:
        if path not in self._audio:
            audio = np.load(path, mmap_mode="r")
            self._audio[path] = audio
            step = max(1, audio.size // 200_000)  # RMS estimate from a subsample
            sub = np.asarray(audio[::step], dtype=np.float32)
            self._file_rms[path] = float(np.sqrt(np.mean(sub**2))) + 1e-8
        return self._audio[path]

    def _crop(self, path: Path, length: int, rng: np.random.Generator) -> np.ndarray:
        audio = self._load(path)
        if audio.size < length:
            return np.tile(np.asarray(audio, dtype=np.float32), length // audio.size + 1)[:length]
        crop = np.empty(0, dtype=np.float32)
        for _ in range(5):
            start = int(rng.integers(0, audio.size - length + 1))
            crop = np.asarray(audio[start : start + length], dtype=np.float32)
            if np.sqrt(np.mean(crop**2)) >= self.min_relative_rms * self._file_rms[path]:
                break
        return crop

    def sample(self, length: int, rng: np.random.Generator) -> np.ndarray:
        category = self.categories[
            int(rng.choice(len(self.categories), p=self.category_probabilities))
        ]
        files = self.files[category]
        path = files[int(rng.choice(len(files), p=self.file_probabilities[category]))]
        noise = self._crop(path, length, rng)
        # no infrasound, as in the synthetic noise; optional stethoscope coupling low-pass
        if self.coupling_cutoff_hz is not None:
            cutoff = rng.uniform(*self.coupling_cutoff_hz)
            sos = signal.butter(
                2, [MIN_NOISE_FREQ_HZ, cutoff], btype="bandpass", fs=self.sr, output="sos"
            )
        else:
            sos = signal.butter(2, MIN_NOISE_FREQ_HZ, btype="highpass", fs=self.sr, output="sos")
        noise = signal.sosfilt(sos, noise)
        return np.asarray(noise / (np.std(noise) + 1e-12), dtype=np.float32)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_audio"] = {}  # memory maps are reopened in each DataLoader worker
        return state
