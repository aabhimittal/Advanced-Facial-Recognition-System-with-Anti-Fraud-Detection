"""Micro-motion detector.

Why it works
------------
A living face is never perfectly still: involuntary blinks, micro-expressions,
and small head rotations create *non-rigid* motion and depth parallax across a
clip. A printed photo or a phone held up to the camera moves *rigidly* — every
pixel shifts together — so once you cancel the global (rigid) motion, almost no
residual local motion remains. We therefore:

1. estimate and remove the dominant global translation between consecutive
   frames (so an attacker simply waving a photo doesn't fool us), then
2. measure the energy of the *residual* per-region motion.

High non-rigid residual ⇒ a real, deforming 3-D face. This detector needs a
clip; on a single frame it abstains.
"""

from __future__ import annotations

import numpy as np

from ..types import DetectorResult
from .base import as_clip, to_grayscale


class MotionDetector:
    name = "motion"

    def __init__(self, min_frames: int = 3, grid: int = 6):
        self.min_frames = int(min_frames)
        self.grid = int(grid)

    def __call__(self, frames: np.ndarray) -> DetectorResult:
        clip = as_clip(frames)
        t = clip.shape[0]
        if t < self.min_frames:
            return DetectorResult(self.name, 0.5, 0.0, {"frames": float(t)})

        from scipy.ndimage import gaussian_filter

        # Genuine non-rigid motion (blinks, expressions) is LOW-frequency, whereas
        # a display grid is high-frequency. Working on low-pass frames means the
        # grid cannot masquerade as facial deformation via sub-pixel misalignment.
        gray = np.stack([gaussian_filter(to_grayscale(f), 2.0) for f in clip])  # (T, H, W)
        residuals = []
        for i in range(t - 1):
            a, b = gray[i], gray[i + 1]
            shift = _global_shift(a, b)
            b_aligned = _translate(b, -shift[0], -shift[1])
            diff = np.abs(b_aligned - a)
            residuals.append(_block_energy(diff, self.grid))

        res = np.stack(residuals)  # (T-1, grid*grid)
        # Non-rigid signature: variability of residual motion across regions & time.
        nonrigid = float(res.std())
        # Live faces produce a small but non-zero, spatially varied residual.
        score = float(1.0 - np.exp(-nonrigid / 0.005))
        reliability = float(np.clip((t - self.min_frames) / 20.0, 0.1, 1.0))
        return DetectorResult(
            self.name, score, reliability,
            {"nonrigid_residual": nonrigid, "frames": float(t)},
        )


def _global_shift(a: np.ndarray, b: np.ndarray, max_shift: int = 4):
    """Integer-pixel global translation from a to b via phase correlation.

    Callers pass low-pass frames so a periodic display grid (which would
    otherwise create ambiguous correlation peaks) cannot corrupt the rigid-motion
    estimate — that is what lets us cancel a waved photo and expose the lack of
    *non-rigid* motion behind it.
    """
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conj(fb)
    cross /= np.abs(cross) + 1e-9
    corr = np.fft.ifft2(cross).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy = peak[0] if peak[0] <= a.shape[0] // 2 else peak[0] - a.shape[0]
    dx = peak[1] if peak[1] <= a.shape[1] // 2 else peak[1] - a.shape[1]
    return int(np.clip(dy, -max_shift, max_shift)), int(np.clip(dx, -max_shift, max_shift))


def _translate(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


def _block_energy(diff: np.ndarray, grid: int) -> np.ndarray:
    h, w = diff.shape
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    out = []
    for i in range(grid):
        for j in range(grid):
            block = diff[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            out.append(block.mean() if block.size else 0.0)
    return np.asarray(out)
