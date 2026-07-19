"""DCT deepfake-fingerprint detector.

Why it works
------------
GAN / diffusion face generators build their output with repeated *upsampling*
(transposed convolutions, pixel-shuffle, interpolate-then-conv). That leaves a
periodic "checkerboard" fingerprint at high spatial frequency which is nearly
invisible in the pixel domain but stands out in the **block-DCT** domain — the
same 8×8 basis JPEG uses. Averaged over many blocks, a real photograph's DCT
coefficients decay smoothly from the DC term toward high frequency; a synthetic
face shows anomalous energy in the highest-frequency coefficients and a
grid-like peak at the Nyquist corner.

This is the frequency-analysis approach of Frank et al. (2020), "Leveraging
Frequency Analysis for Deep Fake Recognition". It is deliberately *distinct* from
:class:`SpectralDetector`: that cue uses a single global FFT and targets the
coarse periodicity of screens/prints, whereas this cue uses local block-DCT
statistics and targets the fine upsampling fingerprint of generated imagery.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dctn

from ..types import DetectorResult
from .base import to_grayscale

# Natural photos concentrate almost all AC energy in low frequencies.
_NATURAL_HF_RATIO = 0.06     # AC energy in the high-frequency corner ring
_NATURAL_CORNER = 1.2        # Nyquist-corner coefficient vs mid-band


class DCTDeepfakeDetector:
    name = "dct"

    def __init__(self, block: int = 8, min_side: int = 64):
        self.block = int(block)
        self.min_side = int(min_side)

    def __call__(self, image: np.ndarray) -> DetectorResult:
        gray = to_grayscale(image)
        h, w = gray.shape
        side = min(h, w)
        if side < self.min_side:
            return DetectorResult(self.name, 0.5, 0.0, {"reason_too_small": float(side)})

        mag = _mean_block_dct(gray, self.block)  # (block, block) mean |DCT|
        ac = mag.copy()
        ac[0, 0] = 0.0  # drop the DC term
        total = ac.sum() + 1e-12

        i, j = np.indices(mag.shape)
        freq = i + j  # 0..2*(block-1); higher = finer detail
        hi = freq >= (2 * self.block - 4)          # outer high-frequency ring
        mid = (freq >= self.block // 2) & (freq < 2 * self.block - 4)

        hf_ratio = float(ac[hi].sum() / total)
        # Checkerboard fingerprint: the Nyquist corner coefficient vs the mid band.
        corner = float(mag[-1, -1] / (mag[mid].mean() + 1e-9))

        hf_excess = max(0.0, (hf_ratio - _NATURAL_HF_RATIO) / _NATURAL_HF_RATIO)
        corner_excess = max(0.0, (corner - _NATURAL_CORNER) / (_NATURAL_CORNER + 1e-6))
        fake_evidence = 0.6 * hf_excess + 0.4 * corner_excess
        score = float(np.exp(-fake_evidence))  # 1 -> natural, ->0 as fingerprint grows

        reliability = float(np.clip((side - self.min_side) / (128 - self.min_side), 0.0, 1.0))
        return DetectorResult(
            self.name, score, reliability,
            {"hf_ratio": hf_ratio, "corner": corner, "fake_evidence": fake_evidence},
        )


def _mean_block_dct(gray: np.ndarray, block: int) -> np.ndarray:
    """Mean magnitude of the 2-D DCT-II taken over every ``block``×``block`` tile."""
    h, w = gray.shape
    h2, w2 = h - h % block, w - w % block
    g = gray[:h2, :w2]
    tiles = g.reshape(h2 // block, block, w2 // block, block).transpose(0, 2, 1, 3)
    coeffs = dctn(tiles, axes=(2, 3), norm="ortho")
    return np.abs(coeffs).mean(axis=(0, 1))
