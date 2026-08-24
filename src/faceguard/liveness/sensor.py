"""Photon-transfer sensor-noise detector — the *injection* attack leg.

The attack this closes
----------------------
Every presentation-attack cue in this library (texture, spectral, moiré, pulse)
assumes the fraudster must hold something up to a real camera. The fastest
growing class of biometric fraud does not: **injection attacks** feed a
generated or replayed video straight into the capture pipeline through a virtual
camera driver, a rooted device's HAL, or a tampered SDK. There is no print, no
screen and no moiré, so presentation-attack detectors see a flawless "live"
face.

The physics that gives it away
------------------------------
A real image sensor obeys the photon-transfer relation: photon arrivals are
Poisson, so the *temporal* variance of a pixel grows linearly with its mean
signal::

    var(pixel) = g · mean(pixel) + read_noise²

That slope ``g`` (the conversion gain) is a property of silicon, not of content.
Synthetic frames, game-engine renders, GAN output and heavily denoised or
re-encoded video all break it: they either carry no temporal noise floor at all,
or carry noise that is *independent* of brightness. So we estimate the
photon-transfer curve directly from the clip and score how sensor-like it looks.

This is a genuinely orthogonal axis of evidence: it holds even for a perfect
deepfake, and — unlike every other detector here — it is unaffected by what the
face is doing.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..types import DetectorResult
from .base import as_clip, to_grayscale

#: Conversion gain of a typical 8-bit consumer sensor, in normalised units.
#: Real cameras vary by an order of magnitude, so the score is deliberately a
#: soft function of "is there *any* intensity-dependent temporal noise".
_TYPICAL_GAIN = 2e-6


class SensorNoiseDetector:
    """Scores how consistent a clip's temporal noise is with a physical sensor."""

    name = "sensor"

    def __init__(self, min_frames: int = 8, bins: int = 8):
        self.min_frames = int(min_frames)
        self.bins = int(bins)

    def __call__(self, frames: np.ndarray) -> DetectorResult:
        clip = as_clip(frames)
        t = clip.shape[0]
        if t < self.min_frames:
            # Temporal noise statistics need a real sample of frames.
            return DetectorResult.abstain(self.name, "too_few_frames")

        gray = np.stack([to_grayscale(f) for f in clip])  # (T, H, W)
        # Hand-held jitter shifts the whole frame by a pixel or two between
        # frames; unaligned, that motion swamps a noise floor three orders of
        # magnitude smaller. Registration is what makes the measurement possible.
        gray = _register(gray)

        # Motion, pulse and expression all vary a pixel over time and would be
        # mistaken for noise. A second temporal difference annihilates any locally
        # smooth signal while passing frame-independent noise untouched, so what
        # survives is the noise floor itself.
        resid = _temporal_highpass(gray)
        # Interpolated motion (a head turning, a photo tilting) survives the
        # temporal filter as a *smooth* spatial field. Sensor noise does not:
        # it is high-frequency by definition. Splitting the residual by spatial
        # band is therefore what isolates silicon from movement.
        fine, whiteness = _spatial_split(resid)
        var = _robust_var(fine)                  # (H, W) temporal noise variance
        mean = gray.mean(axis=0)                 # (H, W) signal level

        # Edges leak residual motion into the variance, so weight the fit towards
        # flat regions where the noise floor is actually measurable.
        flat = _flatness_mask(mean)
        if flat.sum() < 64:
            return DetectorResult.abstain(self.name, "no_flat_region")

        slope, floor, r2 = _photon_transfer_fit(mean[flat], var[flat], self.bins)
        noise_floor = float(np.sqrt(max(np.median(var[flat]), 0.0)))

        # Three independent tells, and the attack has to satisfy all of them:
        #  (a) a measurable temporal noise floor — a rendered or frozen stream is
        #      unnaturally clean;
        #  (b) that noise is spatially *white* — leftover motion is spatially
        #      correlated and would otherwise masquerade as noise;
        #  (c) its variance scales with brightness — noise a generator adds is
        #      uniform across the image, silicon's is not.
        has_noise = 1.0 - float(np.exp(-noise_floor / 1.2e-3))
        photon_like = float(np.clip(slope / _TYPICAL_GAIN, 0.0, 1.0)) * float(np.clip(r2, 0.0, 1.0))
        sensor_like = has_noise * whiteness * (0.35 + 0.65 * photon_like)
        score = float(np.clip(0.10 + 0.88 * sensor_like, 0.02, 0.98))

        # More frames -> a far better variance estimate; this is the dominant
        # driver of how much the fusion should trust this leg.
        reliability = float(np.clip((t - self.min_frames) / 24.0, 0.15, 1.0)) * float(
            np.clip(flat.mean() / 0.2, 0.2, 1.0)
        )
        return DetectorResult(
            self.name,
            score,
            reliability,
            {
                "gain_slope": float(slope),
                "read_noise": float(np.sqrt(max(floor, 0.0))),
                "fit_r2": float(r2),
                "noise_floor": noise_floor,
                "whiteness": whiteness,
                "flat_fraction": float(flat.mean()),
            },
        )


def _register(gray: np.ndarray) -> np.ndarray:
    """Cancel whole-frame integer translation relative to the first frame."""
    ref = np.fft.fft2(gray[0])
    out = np.empty_like(gray)
    out[0] = gray[0]
    h, w = gray.shape[1:]
    for t in range(1, gray.shape[0]):
        cur = np.fft.fft2(gray[t])
        cross = ref * np.conj(cur)
        corr = np.fft.ifft2(cross / (np.abs(cross) + 1e-9)).real
        py, px = np.unravel_index(np.argmax(corr), corr.shape)
        dy = py if py <= h // 2 else py - h
        dx = px if px <= w // 2 else px - w
        out[t] = np.roll(np.roll(gray[t], dy, axis=0), dx, axis=1)
    return out


def _temporal_highpass(gray: np.ndarray) -> np.ndarray:
    """Second temporal difference, scaled to preserve white-noise variance.

    ``(x[t] - 2x[t+1] + x[t+2]) / sqrt(6)`` has unit gain on white noise and zero
    gain on anything locally linear — pulse modulation, lighting drift and slow
    head motion all vanish, leaving the sensor's own noise.
    """
    return (gray[:-2] - 2.0 * gray[1:-1] + gray[2:]) / np.sqrt(6.0)


def _robust_var(resid: np.ndarray) -> np.ndarray:
    """Per-pixel variance via the median absolute deviation.

    A blink or a fast head turn spikes a handful of frames; a plain variance
    would inherit those spikes and report a bogus noise floor. The MAD ignores
    them.
    """
    med = np.median(resid, axis=0)
    mad = np.median(np.abs(resid - med), axis=0)
    return (1.4826 * mad) ** 2


def _spatial_split(resid: np.ndarray, sigma: float = 1.2):
    """Split the temporal residual into its fine-grained part, and score whiteness.

    Returns ``(fine, whiteness)`` where ``fine`` is the spatially high-passed
    residual and ``whiteness`` compares the fine band against the *mid* band —
    1 for sensor noise, →0 for smooth motion residue or codec ringing.

    Comparing fine against mid rather than against the total is deliberate: a
    clip may legitimately contain smooth temporal content (display banding,
    mains flicker, a slow shadow) without that saying anything about whether a
    sensor noise floor exists underneath it.
    """
    fine, mid = _bands(resid, sigma)
    ratio = float((fine**2).mean()) / (float((fine**2).mean() + (mid**2).mean()) + 1e-24)
    return fine, float(np.clip(ratio / _white_band_ratio(sigma), 0.0, 1.0))


def _bands(resid: np.ndarray, sigma: float):
    from scipy.ndimage import gaussian_filter

    low = np.stack([gaussian_filter(r, 2.0 * sigma) for r in resid])
    smooth = np.stack([gaussian_filter(r, sigma) for r in resid])
    return resid - smooth, smooth - low


@lru_cache(maxsize=8)
def _white_band_ratio(sigma: float) -> float:
    """The same fine/(fine+mid) ratio measured on ideal white noise — the yardstick.

    Deterministic in ``sigma`` and independent of the sample, so it is computed
    once per filter width and cached.
    """
    w = np.random.default_rng(0).standard_normal((1, 96, 96))
    fine, mid = _bands(w, sigma)
    return float((fine**2).mean() / ((fine**2).mean() + (mid**2).mean()))


def _flatness_mask(mean: np.ndarray) -> np.ndarray:
    """Pixels whose local spatial gradient is in the lowest tertile.

    Steep gradients turn sub-pixel motion into large temporal swings, which is
    the single biggest source of false "sensor noise".
    """
    gy, gx = np.gradient(mean)
    grad = np.hypot(gy, gx)
    thresh = np.percentile(grad, 33.0)
    mask = grad <= thresh
    # Ignore clipped pixels: a saturated well has no shot noise by construction.
    return mask & (mean > 0.02) & (mean < 0.98)


def _photon_transfer_fit(mean: np.ndarray, var: np.ndarray, bins: int):
    """Weighted least-squares fit of ``var = slope * mean + floor``.

    Binning by intensity before fitting makes the estimate robust to the wildly
    unequal number of pixels at each brightness level in a face crop.
    """
    lo, hi = float(mean.min()), float(mean.max())
    if hi - lo < 1e-3:
        return 0.0, float(var.mean()), 0.0
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(mean, edges) - 1, 0, bins - 1)
    xs, ys = [], []
    for b in range(bins):
        sel = idx == b
        if sel.sum() >= 16:
            xs.append(float(mean[sel].mean()))
            ys.append(float(np.median(var[sel])))  # median: robust to edge leakage
    if len(xs) < 3:
        return 0.0, float(np.median(var)), 0.0
    xs_a, ys_a = np.asarray(xs), np.asarray(ys)
    A = np.stack([xs_a, np.ones_like(xs_a)], axis=1)
    (slope, floor), *_ = np.linalg.lstsq(A, ys_a, rcond=None)
    pred = A @ np.array([slope, floor])
    ss_res = float(((ys_a - pred) ** 2).sum())
    ss_tot = float(((ys_a - ys_a.mean()) ** 2).sum()) + 1e-24
    return float(slope), float(floor), 1.0 - ss_res / ss_tot
