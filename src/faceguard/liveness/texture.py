"""Micro-texture detector via Local Binary Pattern entropy.

Why it works
------------
Genuine skin, seen at sufficient resolution, is covered in rich, *irregular*
micro-texture — pores, fine wrinkles, sensor noise reacting to a 3-D surface.
Spoof surfaces are more uniform: a matte print blurs micro-detail away, and a
display reduces the texture to a *regular* pixel lattice. Local Binary Patterns
(LBP) encode each pixel by the sign pattern of its 8 neighbours; the Shannon
entropy of the LBP histogram is high for irregular natural texture and low for
smooth or periodic spoof surfaces.

Single-image, training-free, and complementary to the spectral cue (which looks
at *global* frequency structure, while this looks at *local* micro-structure).
"""

from __future__ import annotations

import numpy as np

from ..types import DetectorResult
from .base import to_grayscale

_MAX_ENTROPY = 8.0  # bits, an 8-neighbour LBP code
_NATURAL_ENTROPY_RATIO = 0.62  # normalised LBP entropy typical of live skin


class TextureDetector:
    name = "texture"

    def __init__(self, min_side: int = 32):
        self.min_side = min_side

    def __call__(self, image: np.ndarray) -> DetectorResult:
        gray = to_grayscale(image)
        h, w = gray.shape
        side = min(h, w)
        if side < self.min_side:
            return DetectorResult(self.name, 0.5, 0.0, {"reason_too_small": float(side)})

        codes = _lbp(gray)
        hist = np.bincount(codes.ravel(), minlength=256).astype(np.float64)
        p = hist / hist.sum()
        entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        ratio = entropy / _MAX_ENTROPY

        # Genuine skin texture is *stochastic*; a display grid or halftone is
        # *periodic*. Autocorrelation of the high-pass image exposes that: a
        # strong off-centre peak means a repeating pattern -> spoof.
        periodicity = _periodicity(gray)

        # Live evidence needs BOTH rich (high-entropy) AND aperiodic texture.
        richness = float(1.0 / (1.0 + np.exp(-12.0 * (ratio - _NATURAL_ENTROPY_RATIO + 0.06))))
        score = float(richness * (1.0 - 0.9 * periodicity))
        reliability = float(np.clip((side - self.min_side) / (96 - self.min_side), 0.0, 1.0))
        return DetectorResult(
            self.name, score, reliability,
            {"lbp_entropy": entropy, "entropy_ratio": ratio, "periodicity": periodicity},
        )


def _periodicity(gray: np.ndarray) -> float:
    """Strength of the strongest repeating pattern, in [0, 1].

    High-pass the image, then use its autocorrelation (via FFT). Stochastic skin
    decorrelates immediately (a lone central peak); a periodic grid produces
    strong secondary peaks. We report the largest secondary peak, normalised by
    the zero-lag energy.
    """
    from scipy.ndimage import gaussian_filter

    hp = gray - gaussian_filter(gray, 3.0)
    hp = hp - hp.mean()
    f = np.fft.fft2(hp)
    ac = np.fft.ifft2(f * np.conj(f)).real
    ac = np.fft.fftshift(ac)
    center = np.array(ac.shape) // 2
    peak = ac[tuple(center)]
    if peak <= 0:
        return 0.0
    # Exclude a small neighbourhood around zero-lag, then take the max remainder.
    mask = np.ones_like(ac, dtype=bool)
    cy, cx = center
    mask[cy - 2:cy + 3, cx - 2:cx + 3] = False
    secondary = ac[mask].max()
    return float(np.clip(secondary / peak, 0.0, 1.0))


def _lbp(gray: np.ndarray) -> np.ndarray:
    """Vectorised 8-neighbour Local Binary Pattern codes for the interior."""
    g = gray[1:-1, 1:-1]
    # 8 neighbours in a fixed order -> bit weights 1,2,4,...,128.
    neigh = [
        gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:],
        gray[1:-1, 2:], gray[2:, 2:], gray[2:, 1:-1],
        gray[2:, :-2], gray[1:-1, :-2],
    ]
    code = np.zeros_like(g, dtype=np.uint16)
    for i, nb in enumerate(neigh):
        code |= ((nb >= g).astype(np.uint16) << i)
    return code
