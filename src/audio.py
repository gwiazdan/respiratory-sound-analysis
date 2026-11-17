from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


class AudioPreprocessor:
    def __init__(
        self,
        sr: int = 4000,
        f_hp: float = 80.0,
        f_low: float = 100.0,
        f_high: float = 400.0,
        target_rms: float = 0.1,
        frame_ms: float = 50.0,
    ) -> None:
        self.sr = sr
        self.target_rms = target_rms
        self.frame_len = int(sr * (frame_ms / 1000.0))
        self.sos_hp = butter(4, f_hp, btype="highpass", fs=sr, output="sos")
        self.sos_band = butter(4, [f_low, f_high], btype="bandpass", fs=sr, output="sos")

    def __call__(self, y: np.ndarray) -> np.ndarray:
        if y.ndim != 1:
            raise ValueError("Sygnał audio musi być tablicą jednowymiarową.")

        y_clean = sosfiltfilt(self.sos_hp, y)
        y_band = sosfiltfilt(self.sos_band, y_clean)

        num_frames = len(y_band) // self.frame_len
        if num_frames == 0:
            band_rms = np.sqrt(np.mean(y_band**2))
        else:
            frames = y_band[: num_frames * self.frame_len].reshape(num_frames, self.frame_len)
            frame_rms = np.sqrt(np.mean(frames**2, axis=1))
            p30, p85 = np.percentile(frame_rms, 30), np.percentile(frame_rms, 85)
            valid_frames = frame_rms[(frame_rms >= p30) & (frame_rms <= p85)]
            band_rms = np.mean(valid_frames) if len(valid_frames) > 0 else np.mean(frame_rms)

        gain = self.target_rms / (band_rms + 1e-12)
        y_norm = y_clean * gain
        y_norm = np.clip(y_norm, -0.99, 0.99)
        return y_norm
