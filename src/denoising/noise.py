"""Noise sources and mixing for building noisy/clean pairs.

Tzeng et al. mixed ICBHI recordings with the NTUH clinical noise dataset
(stethoscope friction, hospital environment and patient noises), which is not
public. This module offers two interchangeable sources:

* :class:`FileNoiseBank` - your own noise recordings (``--noise-dir``), split
  into disjoint train/test files like in the paper, and
* :class:`SyntheticClinicalNoise` - procedurally generated noise that imitates
  the same three categories (friction/rubbing, environment, patient sounds),
  used when no noise recordings are given. Train and test use different random
  streams and different parameter ranges, so test noise is unseen.

Noise is generated at the pipeline rate (4 kHz by default); band edges above Nyquist are
clamped by ``_bandpass``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from scipy import signal

TRAIN_SNRS_DB: tuple[float, ...] = (15.0, 10.0, 5.0, 0.0)
TEST_SNRS_DB: tuple[float, ...] = (17.5, 12.5, 7.5, 2.5)
#: Coloured noise has no content below this frequency (DC and infrasound are inaudible).
MIN_NOISE_FREQ_HZ = 20.0
#: Bumped whenever the generated noise changes; results with different versions are not
#: comparable (v1: infrasonic coloured noise and unnormalised friction bursts).
NOISE_VERSION = 2


class NoiseSource(Protocol):
    def sample(self, length: int, rng: np.random.Generator) -> np.ndarray: ...


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale ``noise`` so that clean/noise power ratio equals ``snr_db`` and add it."""
    p_clean = float(np.mean(clean**2)) + 1e-12
    p_noise = float(np.mean(noise**2)) + 1e-12
    scale = np.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10)))
    return np.asarray(clean + scale * noise, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Synthetic clinical-like noise
# --------------------------------------------------------------------------- #
def _colored(
    length: int,
    exponent: float,
    rng: np.random.Generator,
    sr: int,
    min_freq: float = MIN_NOISE_FREQ_HZ,
) -> np.ndarray:
    """1/f^exponent noise (0 white, 1 pink, 2 brown) without content below ``min_freq``.

    Without a lower limit a 1/f^2 spectrum puts almost all of its power into DC and
    sub-audible drift (99% below 20 Hz for brown noise), which ``mix_at_snr`` would count as
    noise although it is inaudible and removed by the pipeline's high-pass filter anyway.
    """
    spec = np.fft.rfft(rng.standard_normal(length))
    freqs = np.fft.rfftfreq(length, d=1 / sr)
    audible = freqs >= min_freq
    spec[audible] /= freqs[audible] ** (exponent / 2)
    spec[~audible] = 0.0  # DC and infrasound
    out = np.fft.irfft(spec, n=length)
    return out / (np.std(out) + 1e-12)


def _bandpass(x: np.ndarray, lo: float, hi: float, sr: int, order: int = 4) -> np.ndarray:
    hi = min(hi, sr / 2 * 0.95)
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return np.asarray(signal.sosfilt(sos, x))


def _smooth_envelope(length: int, rate_hz: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    n_ctrl = max(int(length / sr * rate_hz) + 2, 2)
    ctrl = rng.random(n_ctrl)
    env = np.interp(np.linspace(0, n_ctrl - 1, length), np.arange(n_ctrl), ctrl)
    return env


class SyntheticClinicalNoise:
    """Procedural noise imitating stethoscope friction, ward ambience and patient sounds."""

    CATEGORIES = ("friction", "environment", "patient")

    def __init__(self, sample_rate: int = 4000, split: str = "train") -> None:
        self.sr = sample_rate
        self.split = split
        # Test noise uses shifted parameter ranges so it is genuinely unseen.
        self.shift = 0.0 if split == "train" else 0.15

    # -- friction: stethoscope head rubbing on skin / clothes --------------- #
    def _friction(self, n: int, rng: np.random.Generator) -> np.ndarray:
        lo = rng.uniform(80, 300) * (1 + self.shift)
        hi = rng.uniform(1200, 4000) * (1 - self.shift / 2)
        base = _bandpass(rng.standard_normal(n), lo, hi, self.sr)
        env = _smooth_envelope(n, rng.uniform(1.5, 6.0), self.sr, rng) ** rng.uniform(1.5, 4.0)
        # occasional scratchy bursts
        bursts = np.zeros(n)
        for _ in range(rng.integers(1, 6)):
            start = rng.integers(0, n)
            dur = int(rng.uniform(0.05, 0.6) * self.sr)
            bursts[start : start + dur] = rng.uniform(0.5, 2.0)
        # unit-sum window: smoothing keeps the burst level (an unnormalised Hann window of
        # 30 ms would scale it by ~half the window length, drowning the rubbing envelope)
        window = np.hanning(int(0.03 * self.sr))
        bursts = signal.convolve(bursts, window / window.sum(), mode="same")
        return np.asarray(base * (env + bursts))

    # -- environment: ward ambience, monitors, mains hum, distant speech ---- #
    def _environment(self, n: int, rng: np.random.Generator) -> np.ndarray:
        t = np.arange(n) / self.sr
        out = _colored(n, rng.uniform(0.5, 2.0), rng, self.sr) * rng.uniform(0.3, 1.0)
        if rng.random() < 0.6:  # mains hum with harmonics
            f0 = rng.choice([50.0, 60.0])
            hum = sum(
                rng.uniform(0.1, 1.0) / k * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28))
                for k in range(1, 6)
            )
            out += hum * rng.uniform(0.2, 1.0)
        if rng.random() < 0.5:  # monitor beeps
            f_beep = rng.uniform(700, 1500) * (1 + self.shift)
            period = rng.uniform(0.6, 1.5)
            gate = ((t % period) < rng.uniform(0.05, 0.15)).astype(float)
            out += gate * np.sin(2 * np.pi * f_beep * t) * rng.uniform(0.3, 1.5)
        if rng.random() < 0.6:  # babble-like distant speech
            babble = np.zeros(n)
            for _ in range(rng.integers(2, 6)):
                syll = _smooth_envelope(n, rng.uniform(3.0, 6.0), self.sr, rng) ** 2
                babble += _bandpass(rng.standard_normal(n), 250, 3500, self.sr) * syll
            out += babble * rng.uniform(0.3, 1.2)
        return out

    # -- patient: coughs, speech/groans, movement thumps -------------------- #
    def _patient(self, n: int, rng: np.random.Generator) -> np.ndarray:
        out = _colored(n, 1.0, rng, self.sr) * 0.05
        for _ in range(rng.integers(1, 5)):
            kind = rng.integers(0, 3)
            start = int(rng.integers(0, n))
            if kind == 0:  # cough: broadband burst with fast decay
                dur = int(rng.uniform(0.15, 0.4) * self.sr)
                burst = rng.standard_normal(dur) * np.exp(-np.linspace(0, rng.uniform(4, 9), dur))
                burst = _bandpass(burst, 150, 5000, self.sr)
            elif kind == 1:  # voiced sound (groan / talking): harmonic tone with vibrato
                dur = int(rng.uniform(0.3, 1.2) * self.sr)
                tt = np.arange(dur) / self.sr
                f0 = rng.uniform(90, 250) * (1 + self.shift)
                phase = (
                    2 * np.pi * np.cumsum(f0 * (1 + 0.03 * np.sin(2 * np.pi * 5 * tt))) / self.sr
                )
                burst = sum(np.sin(k * phase) / k for k in range(1, 8)) * np.hanning(dur)
            else:  # thump from movement / bumping the chest piece
                dur = int(rng.uniform(0.05, 0.2) * self.sr)
                tt = np.arange(dur) / self.sr
                burst = np.sin(2 * np.pi * rng.uniform(30, 120) * tt) * np.exp(
                    -tt * rng.uniform(20, 60)
                )
            seg = burst[: max(n - start, 0)]
            out[start : start + seg.size] += seg * rng.uniform(1.0, 4.0)
        return out

    def sample(self, length: int, rng: np.random.Generator) -> np.ndarray:
        kind = self.CATEGORIES[int(rng.integers(0, len(self.CATEGORIES)))]
        noise = getattr(self, f"_{kind}")(length, rng)
        if rng.random() < 0.3:  # sometimes combine two categories
            other = self.CATEGORIES[int(rng.integers(0, len(self.CATEGORIES)))]
            extra = getattr(self, f"_{other}")(length, rng)
            noise = noise / (np.std(noise) + 1e-12) + 0.5 * extra / (np.std(extra) + 1e-12)
        return np.asarray(noise / (np.std(noise) + 1e-12), dtype=np.float32)


class MixedNoise:
    """Draws each sample from one of several noise sources (e.g. synthetic + MUSAN).

    ``sources`` maps a name to ``(source, weight)``. With probability ``combine_probability`` a
    second source (drawn the same way, possibly the same one) is added at half the amplitude,
    e.g. stethoscope friction with a conversation in the background.
    """

    def __init__(
        self,
        sources: dict[str, tuple[NoiseSource, float]],
        combine_probability: float = 0.3,
    ) -> None:
        active = {name: entry for name, entry in sources.items() if entry[1] > 0}
        if not active:
            raise ValueError("MixedNoise needs at least one source with weight > 0.")
        self.names = list(active)
        self.sources = [source for source, _ in active.values()]
        weights = np.array([weight for _, weight in active.values()], dtype=float)
        self.probabilities = weights / weights.sum()
        self.combine_probability = combine_probability

    def _draw(self, length: int, rng: np.random.Generator) -> np.ndarray:
        source = self.sources[int(rng.choice(len(self.sources), p=self.probabilities))]
        noise = source.sample(length, rng)
        return np.asarray(noise / (np.std(noise) + 1e-12), dtype=np.float32)

    def sample(self, length: int, rng: np.random.Generator) -> np.ndarray:
        noise = self._draw(length, rng)
        if rng.random() < self.combine_probability:
            noise = noise + 0.5 * self._draw(length, rng)
        return np.asarray(noise / (np.std(noise) + 1e-12), dtype=np.float32)


# --------------------------------------------------------------------------- #
# Noise recordings from disk
# --------------------------------------------------------------------------- #
class FileNoiseBank:
    """Random crops from noise recordings (any format librosa can read)."""

    def __init__(self, files: list[Path], sample_rate: int = 4000) -> None:
        import librosa  # local import keeps module import light

        if not files:
            raise ValueError("FileNoiseBank needs at least one noise file.")
        self.clips = [
            librosa.load(str(f), sr=sample_rate, mono=True)[0].astype(np.float32) for f in files
        ]

    @staticmethod
    def split_files(
        noise_dir: Path, test_fraction: float = 0.25, seed: int = 0
    ) -> tuple[list[Path], list[Path]]:
        exts = {".wav", ".flac", ".mp3", ".ogg"}
        files = sorted(p for p in noise_dir.rglob("*") if p.suffix.lower() in exts)
        if len(files) < 2:
            raise ValueError(f"Need at least 2 noise files in {noise_dir}, found {len(files)}.")
        rng = np.random.default_rng(seed)
        rng.shuffle(files)
        n_test = max(1, round(len(files) * test_fraction))
        return files[n_test:], files[:n_test]

    def sample(self, length: int, rng: np.random.Generator) -> np.ndarray:
        clip = self.clips[int(rng.integers(0, len(self.clips)))]
        if clip.size < length:
            clip = np.tile(clip, int(np.ceil(length / clip.size)) + 1)
        start = int(rng.integers(0, clip.size - length + 1))  # random time shift
        out = clip[start : start + length]
        return (out / (np.std(out) + 1e-12)).astype(np.float32)
