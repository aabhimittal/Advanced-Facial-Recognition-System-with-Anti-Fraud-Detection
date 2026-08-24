"""Depth-parallax detector — the *planar surface* leg.

The gap it fills
----------------
:class:`~faceguard.liveness.motion.MotionDetector` asks a statistical question:
*is there residual motion once global translation is removed?* That is a
heuristic — its threshold has to be tuned, and any source of residual (noise,
interpolation, a rotating photo) counts in the attacker's favour.

This detector asks a *geometric* one: **does the motion field have a structure
that only depth can produce?** The answer is a theorem, not a threshold — no
rigid plane, under any camera or object motion, can produce it. That converts a
tuned heuristic into positive proof of three-dimensionality, and it is the cue
that scales to the full six-degree-of-freedom motion a real user gives you when
they tilt and advance a phone towards their face.

The geometry
------------
Any rigid *planar* object — a print, a phone screen, a tablet — projects, under
any camera motion, to a displacement field that a single homography explains
exactly. To first order for small motion that is an affine field::

    d(x, y) = A · (x, y)ᵀ + b

A real face is not planar. Its nose, cheeks and ears sit at different depths, so
head yaw produces **parallax**: block displacements that no single affine model
can fit. So we measure block-wise displacement, fit the best planar model, and
score the *residual* — the part of the motion that requires depth to exist.

This turns the attacker's own movement into evidence against them: the more they
move a flat object, the more precisely we can certify it is flat. Correspondingly,
when there is too little motion to measure — or when the motion is so violent
that block matching itself becomes unreliable — the detector lowers its own
reliability instead of guessing, and the fusion down-weights it accordingly.

Scope, honestly stated: this is a test for *planar* attacks. A 3-D mask has real
depth and will pass it. That is by design — the mask case is decided on the
material axis (pulse, texture), and a fusion of orthogonal cues is exactly how a
system stays robust when any single cue is out of its depth.
"""

from __future__ import annotations

import numpy as np

from ..types import DetectorResult
from .base import as_clip, to_grayscale


class ParallaxDetector:
    name = "parallax"

    def __init__(self, min_frames: int = 6, grid: int = 4, min_block: int = 16):
        self.min_frames = int(min_frames)
        self.grid = int(grid)
        self.min_block = int(min_block)

    def __call__(self, frames: np.ndarray) -> DetectorResult:
        clip = as_clip(frames)
        t, h, w = clip.shape[:3]
        if t < self.min_frames:
            return DetectorResult.abstain(self.name, "too_few_frames")
        if min(h, w) < self.grid * self.min_block:
            return DetectorResult.abstain(self.name, "crop_too_small")

        from scipy.ndimage import gaussian_filter

        gray = [gaussian_filter(to_grayscale(f), 1.2) for f in clip]
        centres, blocks = _block_layout(h, w, self.grid)

        # Parallax is only visible across a *baseline* of head motion, and head
        # motion is not monotonic — averaging over the clip would cancel a yaw
        # that swings out and back. So we try several time pairs and keep the
        # one that actually moved the most: the widest baseline the subject
        # happened to give us.
        disp, motion = _widest_baseline(gray, blocks)

        planar_residual, fit_quality = _affine_residual(centres, disp)
        # Normalise by the motion actually observed: an attacker who barely moves
        # yields a small residual for trivial reasons, not because of depth.
        ratio = planar_residual / (motion + 1e-6)

        score = float(np.clip(1.0 - np.exp(-ratio / 0.35), 0.02, 0.98))

        # Confidence is driven by how much motion we had to work with. Below a
        # third of a pixel of mean displacement, nothing meaningful is measurable.
        reliability = float(np.clip((motion - 0.3) / 2.0, 0.0, 1.0)) * float(
            np.clip(fit_quality, 0.0, 1.0)
        )
        return DetectorResult(
            self.name,
            score,
            reliability,
            {
                "mean_block_motion_px": motion,
                "nonplanar_residual_px": planar_residual,
                "residual_ratio": ratio,
                "blocks": float(len(centres)),
            },
        )


def _widest_baseline(gray, blocks, probes: int = 4):
    """Block displacements for the frame pair with the largest observed motion."""
    t = len(gray)
    # Average a few adjacent frames per endpoint: cuts noise without materially
    # changing the pose at either end of the baseline.
    span = max(1, t // 12)
    marks = [int(round(f * (t - span))) for f in np.linspace(0.0, 1.0, probes + 1)]
    ends = [np.mean(gray[m:m + span], axis=0) for m in marks]

    best_disp, best_motion = None, -1.0
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            disp = np.array([_block_shift(ends[i][sl], ends[j][sl]) for sl in blocks])
            motion = float(np.sqrt((disp**2).sum(axis=1)).mean())
            if motion > best_motion:
                best_disp, best_motion = disp, motion
    return best_disp, best_motion


def _block_layout(h: int, w: int, grid: int):
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    centres, blocks = [], []
    for i in range(grid):
        for j in range(grid):
            blocks.append((slice(ys[i], ys[i + 1]), slice(xs[j], xs[j + 1])))
            centres.append(((ys[i] + ys[i + 1]) / 2.0 / h - 0.5, (xs[j] + xs[j + 1]) / 2.0 / w - 0.5))
    return np.asarray(centres), blocks


def _block_shift(a: np.ndarray, b: np.ndarray):
    """Sub-pixel (dy, dx) displacement of ``b`` relative to ``a``.

    Phase correlation with a parabolic peak fit: parallax on a face at kiosk
    distance is a sub-pixel to few-pixel effect, so integer accuracy would
    quantise the signal away entirely.
    """
    if a.size == 0 or min(a.shape) < 4 or a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0, 0.0
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.fft2(a * win)
    fb = np.fft.fft2(b * win)
    cross = fa * np.conj(fb)
    corr = np.fft.ifft2(cross / (np.abs(cross) + 1e-9)).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy = _parabolic(corr, peak, axis=0)
    dx = _parabolic(corr, peak, axis=1)
    # Wrap to signed displacement.
    if dy > a.shape[0] / 2:
        dy -= a.shape[0]
    if dx > a.shape[1] / 2:
        dx -= a.shape[1]
    return float(-dy), float(-dx)


def _parabolic(corr: np.ndarray, peak, axis: int) -> float:
    """Three-point parabolic interpolation around the correlation peak."""
    n = corr.shape[axis]
    p = peak[axis]
    idx = list(peak)

    def at(offset: int) -> float:
        idx[axis] = (p + offset) % n
        return float(corr[tuple(idx)])

    ym1, y0, yp1 = at(-1), at(0), at(1)
    denom = ym1 - 2 * y0 + yp1
    delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (ym1 - yp1) / denom
    return p + float(np.clip(delta, -0.5, 0.5))


def _trimmed_affine_residual(A: np.ndarray, disp: np.ndarray, trim: float = 0.25):
    """Fit the planar model, discard the worst-fitting blocks, refit.

    Block matching occasionally fails outright (a low-texture patch, a specular
    blowout). One such block would otherwise manufacture "depth" out of a
    measurement error — the exact way a naive version of this test would false-
    accept a photo. Trimming makes the evidence come from the *field*, not from
    an outlier.
    """
    def residual_for(keep: np.ndarray) -> np.ndarray:
        out = np.empty_like(disp)
        for c in range(2):
            coef, *_ = np.linalg.lstsq(A[keep], disp[keep, c], rcond=None)
            out[:, c] = disp[:, c] - A @ coef
        return out

    n = len(disp)
    keep = np.ones(n, dtype=bool)
    res = residual_for(keep)
    n_drop = int(n * trim)
    if n_drop and n - n_drop >= 4:
        worst = np.argsort((res**2).sum(axis=1))[-n_drop:]
        keep[worst] = False
        res = residual_for(keep)
        return res[keep]
    return res


def _affine_residual(centres: np.ndarray, disp: np.ndarray):
    """RMS displacement left over after the best-fit planar (affine) motion."""
    n = len(centres)
    A = np.column_stack([centres[:, 0], centres[:, 1], np.ones(n)])
    if n < 4:
        return 0.0, 0.0
    residuals = _trimmed_affine_residual(A, disp)
    rms = float(np.sqrt((residuals**2).sum(axis=1).mean()))
    # Reject block displacements that are pure noise: if the affine model
    # explains nothing *and* the field is inconsistent, we cannot trust either.
    spread = float(np.sqrt((disp**2).sum(axis=1)).std())
    fit_quality = float(np.clip(1.0 - spread / (np.abs(disp).mean() * 4.0 + 1e-6), 0.2, 1.0))
    return rms, fit_quality
