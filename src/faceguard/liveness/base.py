"""Common helpers for liveness detectors.

Everything downstream assumes clips arrive as a well-formed ``(T, H, W, C)``
float array in ``[0, 1]``. Real capture stacks do not cooperate: a machine-vision
camera hands you 12-bit-in-16-bit frames, a driver hands you BGRA, a dropped
buffer hands you NaNs, and a mis-wired ROI hands you a zero-width crop. All of
that normalisation is centralised here so no detector has to re-implement it —
and so a malformed frame produces a clean, explainable failure rather than a
silent misclassification.
"""

from __future__ import annotations

import numpy as np


class InvalidFrameError(ValueError):
    """Raised when input cannot be interpreted as an image or clip at all."""


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an (H, W) or (H, W, C) uint8/float image to float64 grayscale in [0,1]."""
    a = _sanitise(np.asarray(img, dtype=np.float64))
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 3:
        c = a.shape[-1]
        if c == 1:
            a = a[..., 0]
        else:
            # Rec. 601 luma; handles RGB (and treats RGBA by ignoring alpha).
            a = a[..., :3] @ np.array([0.299, 0.587, 0.114])
    if a.ndim != 2:
        raise InvalidFrameError(f"expected an image, got shape {np.shape(img)}")
    a = _to_unit_range(a)
    return np.clip(a, 0.0, 1.0)


def as_clip(frames: np.ndarray) -> np.ndarray:
    """Normalise input to a ``(T, H, W, C)`` float clip in ``[0, 1]``.

    Accepts a single image ``(H, W[, C])`` (treated as a 1-frame clip), a
    grayscale clip ``(T, H, W)``, a list of frames, and any integer or float
    dtype. NaN/Inf are scrubbed, and degenerate shapes raise
    :class:`InvalidFrameError` instead of propagating garbage.
    """
    a = np.asarray(frames)
    if a.dtype == object:
        # e.g. a ragged list of frames from a lossy capture buffer.
        raise InvalidFrameError("frames are ragged or non-numeric")
    a = _sanitise(a.astype(np.float64, copy=False))

    if a.ndim == 2:  # (H, W)
        a = a[None, :, :, None]
    elif a.ndim == 3:
        # Ambiguous: could be a single colour image (H,W,3) or a gray clip (T,H,W).
        if a.shape[-1] in (1, 3, 4):
            a = a[None, ...]  # single colour frame -> (1, H, W, C)
        else:
            a = a[..., None]  # gray clip -> (T, H, W, 1)
    elif a.ndim == 5 and a.shape[0] == 1:
        a = a[0]  # a stray batch dimension from a video decoder
    if a.ndim != 4:
        raise InvalidFrameError(f"cannot interpret shape {np.shape(frames)} as a clip")
    if min(a.shape[:3]) < 1 or a.shape[3] < 1:
        raise InvalidFrameError(f"empty clip with shape {a.shape}")

    a = _to_unit_range(a)
    return np.clip(a, 0.0, 1.0)


def _sanitise(a: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf (dropped frames, divide-by-zero in a driver) with finite values."""
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def _to_unit_range(a: np.ndarray) -> np.ndarray:
    """Scale by the sensor's full range rather than blindly assuming 8-bit.

    A 10/12/16-bit industrial camera would otherwise saturate to all-white after
    a naive ``/255``, which turns a perfectly good frame into a bogus "spoof".
    """
    peak = float(a.max()) if a.size else 0.0
    if peak <= 1.0:
        return a
    for full_scale in (255.0, 1023.0, 4095.0, 65535.0):
        if peak <= full_scale:
            return a / full_scale
    return a / peak


#: The most "live" an artefact detector may ever claim on its own. Absence of a
#: spoofing artefact is *weak* evidence of life (a good attack leaves none),
#: while its presence is *strong* evidence of an attack. Encoding that asymmetry
#: in the score keeps a one-sided cue from voting like a two-sided one.
MAX_ONE_SIDED_LIVE = 0.65


def one_sided_score(evidence: float, scale: float = 1.0, floor: float = 0.02) -> float:
    """Map non-negative *spoof evidence* to a liveness score in ``[floor, 0.65]``.

    Detectors that can only ever observe an artefact (moiré, GAN checkerboard,
    display banding) must not be able to certify liveness — the cleanest possible
    image is exactly what a sufficiently good attack produces. So a zero-evidence
    reading tops out at :data:`MAX_ONE_SIDED_LIVE` rather than at 1.0, and only a
    consensus of independent cues can push the fused posterior to certainty.
    """
    ev = max(0.0, float(evidence)) / max(scale, 1e-9)
    return float(floor + (MAX_ONE_SIDED_LIVE - floor) * np.exp(-ev))
