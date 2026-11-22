"""Selection of clean recordings with an AudioSet tagger, as in Tzeng et al. (JMIR AI, 2025).

Tzeng et al. kept only "clean" ICBHI recordings as references for noisy/clean pairs: an
audio-tagging filter with the pretrained CNN14 (PANNs; Kong et al., 2020; AudioSet, 527 classes)
excluded recordings tagged with "music", "speech", "fire", "animal", "cat" or "domestic animals,
pets" (labels among the tagger's most common predictions). 889 of 920 ICBHI recordings were kept;
a manual check gave 92.5% tagging precision. The paper does not state a probability threshold,
so it is a parameter here.

This module contains a dependency-free PyTorch port of CNN14 (log-mel front end computed with
``torch.stft``) that loads the official checkpoint ``Cnn14_mAP=0.431.pth``, a cached tagging of
recordings (``tag_recordings``) and the selection rule (``select_clean``).

Recordings are tagged from the original files resampled to 32 kHz (the tagger's rate), not from
the 4 kHz pipeline audio, so the tagger sees all the bandwidth the recording has. Recordings
stored at 4 kHz (HF Lung, part of ICBHI) or 8 kHz (SPRSound) still contain only up to 2/4 kHz,
which the tagger was not trained on; check its decisions by listening.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from pathlib import Path
from typing import cast

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F  # noqa: N812
from scipy.signal import resample_poly
from torch import nn
from tqdm.auto import tqdm

from src.denoising.base import load_checkpoint

TAGGER_SAMPLE_RATE = 32000
#: Labels excluded by Tzeng et al. (AudioSet ontology names).
TZENG_EXCLUDED_LABELS: tuple[str, ...] = (
    "Music",
    "Speech",
    "Fire",
    "Animal",
    "Cat",
    "Domestic animals, pets",
)
#: AudioSet classes (as predicted by CNN14) that belong to the signal, not to interference:
#: breathing sounds and heart sounds, which are present in almost every "clean" auscultation
#: recording. Noise recordings tagged with them would put the signal into the "noise" of a
#: training pair. Cough, sneeze and throat clearing are kept - they are interference.
#: ("Respiratory sounds" is only a parent node of the ontology, not a CNN14 output class.)
RESPIRATORY_LABELS: tuple[str, ...] = (
    "Breathing",
    "Wheeze",
    "Snoring",
    "Gasp",
    "Pant",
    "Snort",
    "Sigh",
    "Heart sounds, heartbeat",
    "Heart murmur",
)
DEFAULT_WEIGHTS = Path("models/panns/Cnn14_mAP=0.431.pth")
DEFAULT_LABELS = Path("models/panns/class_labels_indices.csv")
DOWNLOAD_HINT = (
    "Download the PANNs CNN14 checkpoint 'Cnn14_mAP=0.431.pth' "
    "(https://zenodo.org/record/3987831) and the AudioSet label list "
    "'class_labels_indices.csv' "
    "(http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv) "
    "into models/panns/."
)


# --------------------------------------------------------------------------- CNN14
class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, pool_size: tuple[int, int]) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class _LogMel(nn.Module):
    """Power spectrogram -> log-mel (dB), as torchlibrosa's LogmelFilterBank (ref 1, amin 1e-10)."""

    def __init__(self, sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float):
        super().__init__()
        mel = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        # same name and layout as torchlibrosa, so the checkpoint's filter bank is loaded
        self.register_buffer("melW", torch.from_numpy(mel.T.astype(np.float32)))

    def forward(self, power: torch.Tensor) -> torch.Tensor:
        mel = torch.matmul(power, cast(torch.Tensor, self.melW))
        return 10.0 * torch.log10(torch.clamp(mel, min=1e-10))


class Cnn14(nn.Module):
    """CNN14 from PANNs (Kong et al., 2020): waveform at 32 kHz -> 527 AudioSet probabilities."""

    def __init__(
        self,
        sample_rate: int = TAGGER_SAMPLE_RATE,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: float = 50.0,
        fmax: float = 14000.0,
        classes: int = 527,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.register_buffer("window", torch.hann_window(window_size), persistent=False)
        self.logmel_extractor = _LogMel(sample_rate, window_size, mel_bins, fmin, fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins)
        channels = (1, 64, 128, 256, 512, 1024, 2048)
        for index in range(6):
            setattr(
                self, f"conv_block{index + 1}", _ConvBlock(channels[index], channels[index + 1])
            )
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes, bias=True)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """``(batch, samples)`` -> ``(batch, classes)`` clip-wise probabilities."""
        spectrum = torch.stft(
            waveform,
            self.window_size,
            hop_length=self.hop_size,
            window=cast(torch.Tensor, self.window),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spectrum.abs().pow(2).transpose(1, 2).unsqueeze(1)  # (B, 1, frames, freq)
        x = self.logmel_extractor(power)  # (B, 1, frames, mel)

        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for index in range(6):
            block: _ConvBlock = getattr(self, f"conv_block{index + 1}")
            x = block(x, pool_size=(2, 2) if index < 5 else (1, 1))
        x = x.mean(dim=3)  # over mel
        x = x.max(dim=2).values + x.mean(dim=2)  # over time
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc_audioset(x))


# --------------------------------------------------------------------------- tagger
def load_audioset_labels(path: str | Path = DEFAULT_LABELS) -> list[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. {DOWNLOAD_HINT}")
    with path.open(encoding="utf-8") as file:
        rows = sorted(csv.DictReader(file), key=lambda row: int(row["index"]))
    return [row["display_name"] for row in rows]


class AudioTagger:
    """Pretrained CNN14 with AudioSet label names."""

    #: torchlibrosa's STFT kernels; this port computes the STFT with ``torch.stft`` instead.
    _IGNORED_KEYS = (
        "spectrogram_extractor.stft.conv_real.weight",
        "spectrogram_extractor.stft.conv_imag.weight",
    )

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        labels: str | Path = DEFAULT_LABELS,
        device: str = "auto",
    ) -> None:
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"{weights} not found. {DOWNLOAD_HINT}")
        self.labels = load_audioset_labels(labels)
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        state = load_checkpoint(weights)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if not isinstance(state, dict):
            raise ValueError(f"Unexpected checkpoint format in {weights}.")
        state = {key: value for key, value in state.items() if key not in self._IGNORED_KEYS}
        self.model = Cnn14(classes=len(self.labels))
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, waveform: np.ndarray) -> np.ndarray:
        """Clip-wise probabilities for one waveform at 32 kHz."""
        tensor = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
        probabilities = self.model(tensor[None].to(self.device))[0]
        return np.asarray(probabilities.float().cpu().numpy())


def load_tagger_audio(path: str | Path) -> np.ndarray:
    """Original recording, mono, resampled to 32 kHz with the same polyphase filter as the pipeline."""
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != TAGGER_SAMPLE_RATE:
        divisor = gcd(int(sample_rate), TAGGER_SAMPLE_RATE)
        audio = resample_poly(audio, TAGGER_SAMPLE_RATE // divisor, int(sample_rate) // divisor)
    return np.asarray(audio, dtype=np.float32)


def _save_tags(tags: pd.DataFrame, cache_file: Path) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_suffix(".tmp.parquet")
    tags.to_parquet(temporary)
    temporary.replace(cache_file)


def tag_recordings(
    paths: Sequence[str | Path],
    cache_file: str | Path,
    tagger: AudioTagger | None = None,
    chunk_size: int = 256,
    loader_threads: int = 4,
    **tagger_options: object,
) -> pd.DataFrame:
    """AudioSet probabilities per recording (index ``wav_path``, one column per label).

    Results are cached in ``cache_file`` (Parquet) and only missing recordings are tagged, so
    the selection rule can be changed without running the network again; the cache is saved
    after every ``chunk_size`` recordings, so an interrupted run resumes where it stopped.
    Audio is loaded and resampled in ``loader_threads`` threads while the GPU tags. The tagger
    is created only when something has to be tagged.
    """
    cache_file = Path(cache_file)
    requested = list(dict.fromkeys(str(path) for path in paths))
    # the cache is keyed by absolute paths, so it is shared between working directories
    keys = [str(Path(path).resolve()) for path in requested]
    cached = pd.read_parquet(cache_file) if cache_file.is_file() else pd.DataFrame()
    missing = list(dict.fromkeys(key for key in keys if key not in cached.index))
    if missing:
        tagger = tagger or AudioTagger(**tagger_options)  # type: ignore[arg-type]
        progress = tqdm(total=len(missing), desc="Tagging recordings with CNN14")
        with ThreadPoolExecutor(max_workers=loader_threads) as pool:
            for start in range(0, len(missing), chunk_size):
                chunk = missing[start : start + chunk_size]
                probabilities = []
                for audio in pool.map(load_tagger_audio, chunk):
                    probabilities.append(tagger.predict(audio))
                    progress.update()
                new = pd.DataFrame(
                    np.stack(probabilities),
                    index=pd.Index(chunk, name="wav_path"),
                    columns=tagger.labels,
                )
                cached = pd.concat([cached, new]) if not cached.empty else new
                _save_tags(cached, cache_file)
        progress.close()
    result = cached.loc[keys]
    result.index = pd.Index(requested, name="wav_path")  # paths as the caller gave them
    return result


def select_clean(
    tags: pd.DataFrame,
    excluded_labels: Sequence[str] = TZENG_EXCLUDED_LABELS,
    threshold: float = 0.2,
) -> pd.DataFrame:
    """Decide per recording whether it is clean.

    A recording is excluded when any of ``excluded_labels`` has probability >= ``threshold``.
    Returns a frame indexed like ``tags`` with ``clean`` (bool), ``reason`` (excluded labels,
    strongest first) and ``max_excluded_probability``.
    """
    unknown = sorted(set(excluded_labels) - set(tags.columns))
    if unknown:
        raise KeyError(f"Labels not in the AudioSet ontology: {unknown}")
    excluded = tags[list(excluded_labels)]
    hits = excluded >= threshold

    def reason(row: pd.Series) -> str:
        names = row[hits.loc[row.name]].sort_values(ascending=False)
        return ", ".join(f"{name} ({value:.2f})" for name, value in names.items())

    return pd.DataFrame(
        {
            "clean": ~hits.any(axis=1),
            "reason": excluded.apply(reason, axis=1),
            "max_excluded_probability": excluded.max(axis=1),
        },
        index=tags.index,
    )
