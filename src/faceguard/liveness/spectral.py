"""Spectral fingerprint detector.

Why it works
------------
A face captured *directly* by a camera has an approximately natural image
spectrum: energy falls off smoothly with spatial frequency (roughly a ``1/f``
power law). Common spoofs break this in characteristic, hard-to-hide ways:

* **Replayed / printed attacks** (phone screen, printout) add periodic
  structure — the pixel grid, moiré, or halftone dots — which injects *excess
  high-frequency energy* and *sharp periodic peaks* into the spectrum.
* **GAN / deepfake** faces carry upsampling "checkerboard" fingerprints that
  show up as regular peaks in the high-frequency corners of the 2-D spectrum.

So we score a crop by how *natural* its radial frequency profile looks. This
detector needs no training and no temporal information — it works on a single
image, which makes it a robust, always-available leg of the fusion.
"""

from __future__ import annotations

import numpy as np

from ..types import DetectorResult
from .base import to_grayscale

# Baselines calibrated against natural face crops. Spoofs push metrics above these.
_NATURAL_HF_RATIO = 0.18   # typical fraction of energy beyond 0.5*Nyquist
_NATURAL_PEAKINESS = 3.0   # typical excess-kurtosis of the HF band


class SpectralDetector:
    name = "spectral"

    def __init__(self, min_side: int = 48):
        # Below this crop size a frequency estimate is unreliable.
        self.min_side = min_side

    def __call__(self, image: np.ndarray) -> DetectorResult:
        gray = to_grayscale(image)
        h, w = gray.shape
        side = min(h, w)

        # Reliability grows with resolution and saturates once we have enough
        # pixels to estimate a spectrum (~128px). Tiny crops are untrustworthy.
        reliability = float(np.clip((side - self.min_side) / (128 - self.min_side), 0.0, 1.0))
        if side < self.min_side:
            return DetectorResult(self.name, 0.5, 0.0, {"reason_too_small": float(side)})

        # Window to suppress edge-wrap leakage, then 2-D FFT magnitude spectrum.
        win = np.outer(np.hanning(h), np.hanning(w))
        spec = np.abs(np.fft.fftshift(np.fft.fft2(gray * win))) + 1e-8
        power = spec**2

        radial = _radial_profile(power)
        n = len(radial)
        # Energy fraction in the outer (high-frequency) half of the spectrum.
        hf_ratio = float(power_beyond(radial, 0.5))
        # Peakiness of the high-frequency band captures periodic moiré/GAN spikes.
        hf_band = radial[n // 2:]
        peakiness = float(_excess_kurtosis(hf_band))

        # Each cue is 0 when natural, growing positive as it looks spoofed.
        hf_excess = max(0.0, (hf_ratio - _NATURAL_HF_RATIO) / _NATURAL_HF_RATIO)
        peak_excess = max(0.0, (peakiness - _NATURAL_PEAKINESS) / (_NATURAL_PEAKINESS + 1e-6))
        spoof_evidence = 0.6 * hf_excess + 0.4 * peak_excess
        score = float(np.exp(-spoof_evidence))  # 1 -> natural, ->0 as spoof cues rise

        return DetectorResult(
            self.name,
            score,
            reliability,
            {"hf_ratio": hf_ratio, "peakiness": peakiness, "spoof_evidence": spoof_evidence},
        )


def _radial_profile(power: np.ndarray) -> np.ndarray:
    """Azimuthally averaged power as a function of radius from the DC centre."""
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    profile = tbin / np.maximum(nr, 1)
    return profile[: min(cy, cx)]  # drop corners beyond the inscribed circle


def power_beyond(radial: np.ndarray, frac: float) -> float:
    """Fraction of total radial power beyond ``frac`` of the max radius."""
    total = radial.sum() + 1e-12
    cut = int(len(radial) * frac)
    return radial[cut:].sum() / total


def _excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = x.mean()
    s = x.std() + 1e-12
    return float(np.mean(((x - m) / s) ** 4) - 3.0)
