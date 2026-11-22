"""Denoising whole recordings and saving processed copies of a dataset.

:func:`enhance_signal` runs a model on a recording of any length: the signal is resampled to
the model rate, cut into overlapping segments (the length the model was trained on),
enhanced in batches and joined with linear cross-fades.

:func:`save_processed_audio` stores a processed copy of every recording once, e.g.::

    model = load_model("models/wave_u_net/best.pt", device="cuda")
    save_processed_audio(wav_paths, Path("data/denoised/wave_u_net"),
                         lambda audio: enhance_signal(model, audio))

The pipeline reads such a folder with ``build_dataset_with_features(audio_dir=...)``.
Existing files are reused, so after retraining a model use a new folder (or delete it).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from tqdm.auto import tqdm

from src.data_pipeline import MultiDatasetPipelineBuilder, _load_feature_audio, processed_audio_path
from src.denoising.base import DenoisingModel

PIPELINE_SAMPLE_RATE = MultiDatasetPipelineBuilder.FEATURE_SAMPLE_RATE


def _resample(audio: np.ndarray, source: int, target: int) -> np.ndarray:
    if source == target:
        return audio
    divisor = gcd(source, target)
    return np.asarray(resample_poly(audio, target // divisor, source // divisor), dtype=np.float32)


def _fit(audio: np.ndarray, length: int) -> np.ndarray:
    return audio[:length] if len(audio) >= length else np.pad(audio, (0, length - len(audio)))


@torch.inference_mode()
def _run(model: DenoisingModel, chunks: list[np.ndarray]) -> list[np.ndarray]:
    device = next(model.parameters()).device
    batch = torch.from_numpy(np.stack(chunks)).to(device)
    enhanced = model.enhance(batch).float().cpu().numpy().reshape(len(chunks), -1)
    return [_fit(row, len(chunk)) for row, chunk in zip(enhanced, chunks, strict=True)]


def enhance_signal(
    model: DenoisingModel,
    audio: np.ndarray,
    sr: int = PIPELINE_SAMPLE_RATE,
    segment_seconds: float | None = 4.0,
    overlap_seconds: float = 0.5,
    batch_size: int = 16,
) -> np.ndarray:
    """Enhanced copy of ``audio`` (1-D, at ``sr``), with the same length."""
    model.eval()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    signal = _resample(audio, sr, model.sample_rate)
    segment = len(signal) if segment_seconds is None else int(segment_seconds * model.sample_rate)

    if len(signal) <= segment:
        output = _run(model, [signal])[0]
    else:
        overlap = int(overlap_seconds * model.sample_rate)
        hop = segment - overlap
        starts = []
        for start in range(0, len(signal), hop):
            starts.append(start)
            if start + segment >= len(signal):
                break
        chunks = [signal[start : start + segment] for start in starts]
        # full-length segments in batches; a shorter last segment on its own
        enhanced: list[np.ndarray] = []
        full = [chunk for chunk in chunks if len(chunk) == segment]
        for first in range(0, len(full), batch_size):
            enhanced += _run(model, full[first : first + batch_size])
        if len(full) < len(chunks):
            enhanced += _run(model, chunks[len(full) :])

        # strictly positive cross-fade ramps keep the normalising weight > 0 everywhere
        window = np.ones(segment, dtype=np.float32)
        if overlap > 0:
            ramp = np.linspace(0.0, 1.0, overlap + 2, dtype=np.float32)[1:-1]
            window[:overlap] = ramp
            window[-overlap:] = ramp[::-1]
        output = np.zeros(len(signal), dtype=np.float32)
        weight = np.zeros(len(signal), dtype=np.float32)
        for start, piece in zip(starts, enhanced, strict=True):
            output[start : start + len(piece)] += piece * window[: len(piece)]
            weight[start : start + len(piece)] += window[: len(piece)]
        output /= np.maximum(weight, 1e-8)

    return _fit(_resample(output, model.sample_rate, sr), len(audio)).astype(np.float32)


def save_processed_audio(
    wav_paths: Sequence[str],
    out_dir: Path,
    process: Callable[[np.ndarray], np.ndarray] | None = None,
    workers: int = 4,
) -> list[Path]:
    """Save every recording at 4 kHz (float16 ``.npy``), optionally processed; returns the
    file paths in the order of ``wav_paths``. Files that already exist are skipped.

    Loading runs in ``workers`` threads; ``process`` (e.g. a model on the GPU) runs in the
    calling thread.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [processed_audio_path(path, out_dir) for path in wav_paths]
    missing = [
        (path, target)
        for path, target in zip(wav_paths, targets, strict=True)
        if not target.exists()
    ]

    def load(item: tuple[str, Path]) -> tuple[Path, np.ndarray]:
        return item[1], _load_feature_audio(item[0], PIPELINE_SAMPLE_RATE)[0]

    with ThreadPoolExecutor(max(1, workers)) as pool:
        loaded: Iterator[tuple[Path, np.ndarray]] = pool.map(load, missing)
        for target, audio in tqdm(loaded, total=len(missing), desc=out_dir.name):
            result = audio if process is None else process(audio)
            temporary = target.with_suffix(".tmp.npy")
            np.save(temporary, np.asarray(result, dtype=np.float16))
            temporary.replace(target)
    return targets
