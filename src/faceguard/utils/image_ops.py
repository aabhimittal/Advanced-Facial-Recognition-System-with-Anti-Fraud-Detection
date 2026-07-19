"""Small image helpers used by the synthetic generator and demos."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def pink_field(rng: np.random.Generator, h: int, w: int, beta: float = 1.0) -> np.ndarray:
    """A random field with a natural ``1/f**beta`` amplitude spectrum, in [0,1]."""
    white = rng.standard_normal((h, w))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(fy**2 + fx**2)
    radius[0, 0] = 1.0
    spec = np.fft.fft2(white) / (radius**beta)
    field = np.fft.ifft2(spec).real
    field -= field.min()
    return field / (field.max() + 1e-9)


def face_blob(h: int, w: int) -> np.ndarray:
    """A smooth bright-centre gradient so demo crops look vaguely face-like."""
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    return np.exp(-(x**2 + y**2) * 1.5)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if img.ndim == 3:
        return np.stack([gaussian_filter(img[..., c], sigma) for c in range(img.shape[-1])], axis=-1)
    return gaussian_filter(img, sigma)
