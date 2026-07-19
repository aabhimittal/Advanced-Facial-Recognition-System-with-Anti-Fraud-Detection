"""Remote photoplethysmography (rPPG) pulse detector.

Why it works
------------
Every heartbeat pushes blood through the capillaries just under facial skin,
which very slightly changes how much light the skin absorbs. A camera sees this
as a tiny, *periodic* colour fluctuation (strongest in green) — invisible to the
eye but recoverable from a short clip. A genuine, live face therefore contains a
clean periodic signal inside the human heart-rate band (~42-240 bpm). A printed
photo has no pulse at all; a replayed video usually has a smeared or absent one.

We recover the pulse with the **POS** algorithm (Plane-Orthogonal-to-Skin,
Wang et al. 2017) — a projection that is robust to illumination changes — then
measure the signal-to-noise ratio of the strongest in-band frequency. High
in-band SNR ⇒ a real beating heart ⇒ live.

Reliability is low when the clip is too short for a trustworthy frequency
estimate, so single frames simply abstain and let the other detectors decide.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from ..types import DetectorResult
from .base import as_clip

_HR_LOW_HZ = 0.7   # 42 bpm
_HR_HIGH_HZ = 4.0  # 240 bpm


class RPPGDetector:
    name = "rppg"

    def __init__(self, fps: float = 30.0, min_frames: int = 30):
        self.fps = float(fps)
        self.min_frames = int(min_frames)

    def __call__(self, frames: np.ndarray) -> DetectorResult:
        clip = as_clip(frames)  # (T, H, W, C)
        t = clip.shape[0]
        if clip.shape[-1] < 3 or t < self.min_frames:
            # No colour or too short -> abstain (reliability 0).
            return DetectorResult(self.name, 0.5, 0.0, {"frames": float(t)})

        # Spatial-mean RGB trace per frame -> (3, T).
        rgb = clip[..., :3].reshape(t, -1, 3).mean(axis=1).T
        pulse = _pos(rgb)
        pulse = _bandpass(pulse, self.fps)

        snr, hr_hz = _band_snr(pulse, self.fps)
        # Map SNR (dB-like ratio) to a liveness score with a soft threshold at ~3.
        score = float(1.0 / (1.0 + np.exp(-(snr - 3.0))))

        # Reliability scales with clip length: a 1-2s clip is marginal, 5s+ is solid.
        reliability = float(np.clip((t - self.min_frames) / (5 * self.fps), 0.05, 1.0))
        return DetectorResult(
            self.name, score, reliability,
            {"snr": float(snr), "hr_bpm": float(hr_hz * 60.0), "frames": float(t)},
        )


def _pos(rgb: np.ndarray) -> np.ndarray:
    """POS pulse extraction from a (3, T) RGB temporal trace."""
    eps = 1e-9
    mean = rgb.mean(axis=1, keepdims=True) + eps
    cn = rgb / mean  # temporal normalisation
    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    s = proj @ cn  # (2, T)
    alpha = (s[0].std() + eps) / (s[1].std() + eps)
    h = s[0] + alpha * s[1]
    return h - h.mean()


def _bandpass(x: np.ndarray, fps: float) -> np.ndarray:
    nyq = fps / 2.0
    low = max(_HR_LOW_HZ / nyq, 1e-3)
    high = min(_HR_HIGH_HZ / nyq, 0.99)
    if low >= high:
        return x
    b, a = sp_signal.butter(3, [low, high], btype="band")
    return sp_signal.filtfilt(b, a, x, method="gust")


def _band_snr(x: np.ndarray, fps: float):
    """SNR of the dominant in-band frequency vs the rest of the HR band."""
    x = x - x.mean()
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    ps = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    band = (freqs >= _HR_LOW_HZ) & (freqs <= _HR_HIGH_HZ)
    if not band.any() or ps[band].sum() <= 0:
        return 0.0, 0.0
    band_ps = ps[band]
    band_freqs = freqs[band]
    peak = int(np.argmax(band_ps))
    signal_power = band_ps[peak]
    noise = np.delete(band_ps, peak).mean() + 1e-12
    snr = float(signal_power / noise)
    return snr, float(band_freqs[peak])
