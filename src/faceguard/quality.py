"""Capture-quality gating — deciding when *not* to decide.

Motivation
----------
Every anti-spoofing paper reports FAR/FRR on curated datasets where the face is
well-lit, in focus and large in frame. Production is not like that: a lobby
turnstile at 7am faces a backlit doorway, a warehouse tablet has a greasy lens,
a kiosk in a stairwell runs at 20 lux, and an ATM camera occasionally serves a
half-frozen frame.

Those samples are dangerous in a *specific* way: low light kills skin
micro-texture, a smeared lens kills high-frequency energy, and a compressed
stream flattens the spectrum — all of which look exactly like the evidence a
liveness detector reads as "print attack". A system without a quality gate
therefore does not fail randomly under bad capture; it fails *biased towards
accusing innocent users*.

So FaceGuard measures capture quality **before** liveness, on physical,
model-free quantities, and when the capture cannot support a decision it
returns :attr:`~faceguard.types.FraudVerdict.INDETERMINATE` ("retry"), never
FRAUD. The report is also actionable: each failing factor maps to an operator
instruction ("move into better light", "clean the lens").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .liveness.base import as_clip, to_grayscale

def _radial_power(power: np.ndarray) -> np.ndarray:
    """Azimuthally averaged power spectrum, indexed by radius from DC."""
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    profile = np.bincount(r.ravel(), power.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    return profile[: min(cy, cx)]


#: Human-facing remediation for each quality factor that can fail.
_REMEDY = {
    "resolution": "Move closer to the camera — the face crop is too small.",
    "exposure": "Adjust lighting — the image is too dark or too bright.",
    "dynamic_range": "Increase contrast — the image is washed out or clipped.",
    "focus": "Hold still and clean the lens — the image is out of focus.",
    "motion_blur": "Hold still — the capture is smeared by motion.",
    "colour": "Use a colour camera — pulse and chroma cues need RGB.",
    "temporal": "Capture a longer clip — too few usable frames.",
    "stability": "Camera feed is unstable — frames are dropping or frozen.",
}


@dataclass
class QualityReport:
    """Per-factor capture quality, each in ``[0, 1]`` (1 = ideal)."""

    factors: Dict[str, float] = field(default_factory=dict)
    usable: bool = True
    #: Factors that fell below their minimum, worst first.
    failures: Tuple[str, ...] = ()

    @property
    def score(self) -> float:
        """Overall quality = the *weakest* factor.

        A minimum, not a mean: a perfectly lit, perfectly sharp capture of a
        12-pixel face is still unusable, and averaging would hide that.
        """
        return min(self.factors.values()) if self.factors else 0.0

    def remediation(self) -> List[str]:
        return [_REMEDY[f] for f in self.failures if f in _REMEDY]

    def summary(self) -> str:
        state = "usable" if self.usable else "unusable"
        worst = self.failures[0] if self.failures else min(self.factors, key=self.factors.get, default="-")
        return f"quality={self.score:.2f} ({state}, weakest={worst})"


class CaptureQualityGate:
    """Model-free capture assessment.

    Every factor is a physical measurement with a documented failure mode, so the
    gate is explainable to an auditor and portable across cameras.
    """

    def __init__(
        self,
        min_face_px: int = 64,
        min_quality: float = 0.25,
        require_colour: bool = False,
        require_clip: bool = False,
    ):
        self.min_face_px = int(min_face_px)
        self.min_quality = float(min_quality)
        # Advisory-by-default factors. A mono/IR camera and a single-shot
        # enrolment photo are both legitimate deployments: they simply mean the
        # chroma/temporal detectors abstain and confidence drops. Turn these on
        # for a deployment that genuinely requires an RGB video capture.
        self.require_colour = bool(require_colour)
        self.require_clip = bool(require_clip)

    def __call__(self, frames: np.ndarray) -> QualityReport:
        clip = as_clip(frames)
        t, h, w, c = clip.shape
        gray = np.stack([to_grayscale(f) for f in clip])  # (T, H, W)

        factors = {
            "resolution": self._resolution(min(h, w)),
            "exposure": self._exposure(gray),
            "dynamic_range": self._dynamic_range(gray),
            "focus": self._focus(gray),
            "colour": self._colour(clip),
            "temporal": self._temporal(t),
            "stability": self._stability(gray),
        }
        if t >= 2:
            factors["motion_blur"] = self._motion_blur(gray)

        failures = tuple(
            name for name, _ in sorted(factors.items(), key=lambda kv: kv[1])
            if factors[name] < self.min_quality
        )
        advisory = set()
        if not self.require_colour:
            advisory.add("colour")
        if not self.require_clip:
            advisory.add("temporal")
        hard_failures = tuple(f for f in failures if f not in advisory)
        return QualityReport(
            factors=factors,
            usable=not hard_failures,
            failures=hard_failures or failures,
        )

    # -- individual factors -------------------------------------------------
    def _resolution(self, side: int) -> float:
        """Saturates once the crop is comfortably above the minimum face size."""
        return float(np.clip((side - 0.5 * self.min_face_px) / self.min_face_px, 0.0, 1.0))

    def _exposure(self, gray: np.ndarray) -> float:
        """Penalises both darkness and clipping.

        Clipped pixels carry *no* information — a blown-out cheek is
        indistinguishable from a printed white patch — so the fraction of pixels
        pinned at either rail matters as much as the mean level.
        """
        mean = float(gray.mean())
        clipped = float(np.mean((gray <= 0.02) | (gray >= 0.98)))
        # Ideal mean luminance ~0.5; fall off smoothly either side.
        level = float(np.exp(-((mean - 0.5) ** 2) / (2 * 0.18**2)))
        return float(np.clip(level * (1.0 - min(1.0, clipped / 0.35)), 0.0, 1.0))

    def _dynamic_range(self, gray: np.ndarray) -> float:
        """Robust contrast: the 5-95 percentile spread of luminance."""
        lo, hi = np.percentile(gray, [5, 95])
        return float(np.clip((hi - lo) / 0.35, 0.0, 1.0))

    def _focus(self, gray: np.ndarray) -> float:
        """Ratio of high- to mid-frequency spectral energy.

        Absolute sharpness measures (Laplacian variance and friends) are
        content-dependent: a smooth, evenly-lit subject scores like a defocused
        one, and would be turned away for a fault that does not exist. Defocus
        has a specific signature instead — it attenuates *high* frequencies
        relative to mid ones — so the band ratio isolates the optical fault from
        the subject's own texture.
        """
        rep = np.median(gray, axis=0)
        h, w = rep.shape
        if min(h, w) < 16:
            return 1.0
        win = np.outer(np.hanning(h), np.hanning(w))
        power = np.abs(np.fft.fftshift(np.fft.fft2(rep * win))) ** 2
        radial = _radial_power(power)
        n = len(radial)
        if n < 8:
            return 1.0
        mid = radial[int(0.15 * n):int(0.5 * n)].sum()
        high = radial[int(0.5 * n):].sum()
        ratio = float(high / (mid + 1e-18))
        # An in-focus natural image keeps ~2% of its mid-band energy in the top
        # half of the spectrum; heavy defocus drops that by an order of magnitude.
        return float(np.clip(ratio / 0.02, 0.0, 1.0))

    def _motion_blur(self, gray: np.ndarray) -> float:
        """Intermittent smearing: how much sharpness the worst frames lose.

        Deliberately measured as a *relative* sharpness drop across the clip, not
        as inter-frame change. Content that legitimately changes fast (an
        expression, a turn) is not blur, and treating it as blur would reject
        exactly the animated captures the temporal detectors need most.
        """
        from scipy.ndimage import laplace

        sharp = np.array([float(laplace(f).std()) for f in gray])
        best = float(sharp.max())
        if best <= 1e-9:
            return 0.0
        # Median-vs-best: a couple of smeared frames are survivable, a clip that
        # is mostly smear is not.
        return float(np.clip((np.median(sharp) / best - 0.35) / 0.45, 0.0, 1.0))

    def _colour(self, clip: np.ndarray) -> float:
        """Detects a mono/IR feed replicated across three channels."""
        if clip.shape[-1] < 3:
            return 0.0
        chroma = float(np.mean(np.std(clip[..., :3], axis=-1)))
        return float(np.clip(chroma / 0.02, 0.0, 1.0))

    def _temporal(self, t: int) -> float:
        """Temporal cues (pulse, micro-motion) need a real clip, not a burst."""
        return float(np.clip(t / 15.0, 0.0, 1.0))

    def _stability(self, gray: np.ndarray) -> float:
        """Catches a frozen or duplicated stream.

        A stalled decoder that repeats one frame looks *perfectly* live to naive
        temporal checks while carrying zero temporal evidence, so a frozen feed
        must be reported as unusable rather than analysed.
        """
        if gray.shape[0] < 2:
            return 1.0
        variation = float(np.mean(np.abs(np.diff(gray, axis=0))))
        # Genuine capture noise alone puts this comfortably above 1e-4.
        return float(np.clip(variation / 5e-4, 0.0, 1.0))
