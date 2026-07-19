"""Common helpers for liveness detectors."""

from __future__ import annotations

import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an (H, W) or (H, W, C) uint8/float image to float64 grayscale in [0,1]."""
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        # Rec. 601 luma; handles RGB (and treats RGBA by ignoring alpha).
        a = a[..., :3] @ np.array([0.299, 0.587, 0.114])
    if a.max() > 1.0:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def as_clip(frames: np.ndarray) -> np.ndarray:
    """Normalise input to a (T, H, W, C) float clip.

    Accepts a single image (H, W[, C]) -> treated as a 1-frame clip.
    """
    a = np.asarray(frames, dtype=np.float64)
    if a.ndim == 2:  # (H, W)
        a = a[None, :, :, None]
    elif a.ndim == 3:
        # Ambiguous: could be a single colour image (H,W,3) or a gray clip (T,H,W).
        if a.shape[-1] in (1, 3, 4):
            a = a[None, ...]  # single colour frame -> (1, H, W, C)
        else:
            a = a[..., None]  # gray clip -> (T, H, W, 1)
    if a.max() > 1.0:
        a = a / 255.0
    return a
