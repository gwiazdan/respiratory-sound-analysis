import numpy as np


class AudioPreprocessor:
    def __init__(
        self,
        sr: int = 4000,
        f_low: float = 100.0,
        f_high: float = 400.0,
        target_rms: float = 0.1,
    ) -> None:
        self.sr = sr
        self.f_low = f_low
        self.f_high = f_high
        self.target_rms = target_rms

    def __call__(self, y: np.ndarray) -> np.ndarray:
        if y.ndim != 1:
            raise ValueError("Sygnał audio musi być tablicą jednowymiarową.")

        window = np.hanning(len(y))
        spectrum = np.fft.rfft(y * window)
        freqs = np.fft.rfftfreq(len(y), d=1.0 / self.sr)

        mask = (freqs >= self.f_low) & (freqs <= self.f_high)
        power = 2.0 * np.sum(np.abs(spectrum[mask]) ** 2) / (len(y) * np.sum(window**2))
        band_rms = np.sqrt(power)

        gain = self.target_rms / (band_rms + 1e-12)
        return y * gain
