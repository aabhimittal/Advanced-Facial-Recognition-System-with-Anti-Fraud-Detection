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


def warp_planar(img: np.ndarray, angle_deg: float, scale: float, dy: float, dx: float) -> np.ndarray:
    """Rotate/scale/translate an image as a rigid *plane*.

    This is how a held photo or phone actually moves in front of a kiosk: not a
    pure translation, but a full planar warp. It is the attack that a
    translation-only motion check misses and the parallax check catches.
    """
    from scipy.ndimage import map_coordinates

    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    th = np.deg2rad(angle_deg)
    y0, x0 = yy - cy, xx - cx
    src_y = (np.cos(th) * y0 - np.sin(th) * x0) / scale + cy - dy
    src_x = (np.sin(th) * y0 + np.cos(th) * x0) / scale + cx - dx
    coords = np.stack([src_y, src_x])
    if img.ndim == 3:
        return np.stack(
            [map_coordinates(img[..., c], coords, order=1, mode="nearest") for c in range(img.shape[-1])],
            axis=-1,
        )
    return map_coordinates(img, coords, order=1, mode="nearest")


def warp_depth(img: np.ndarray, depth: np.ndarray, amount: float) -> np.ndarray:
    """Displace pixels horizontally in proportion to ``depth``.

    A first-order model of head yaw: nearer points (nose) sweep further across
    the sensor than farther ones (ears). No planar warp can reproduce it, which
    is precisely why it certifies a 3-D subject.
    """
    from scipy.ndimage import map_coordinates

    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    src_x = xx - amount * depth
    coords = np.stack([yy, src_x])
    if img.ndim == 3:
        return np.stack(
            [map_coordinates(img[..., c], coords, order=1, mode="nearest") for c in range(img.shape[-1])],
            axis=-1,
        )
    return map_coordinates(img, coords, order=1, mode="nearest")
