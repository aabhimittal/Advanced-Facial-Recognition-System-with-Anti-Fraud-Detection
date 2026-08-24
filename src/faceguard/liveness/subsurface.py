"""Subsurface-scattering detector — the *material* leg.

The attack it closes
--------------------
A well-made silicone or resin mask is the hardest presentation attack there is.
It is genuinely three-dimensional, so every geometric test (parallax, depth,
stereo) passes it. It is captured by a real camera, so sensor-noise and moiré
tests pass it. It moves with the wearer's head, so motion tests pass it. What it
cannot be is *skin*.

The physics
-----------
Human skin is a translucent, layered medium. Light does not simply reflect off
it — it enters the epidermis, scatters through the dermis and re-emerges
millimetres away. Crucially the mean free path is strongly wavelength-dependent:
red light penetrates several times deeper than green or blue, because
haemoglobin and melanin absorb the short wavelengths far more strongly.

The visible consequence is that **the red channel of a real face is measurably
blurrier than its green and blue channels** — fine detail (pores, stubble,
wrinkles) survives in green, and is smeared away in red. Pigmented silicone,
latex, resin, paper and pixels are all opaque or uniformly scattering: their
channels carry the *same* detail.

So the score is a ratio of high-frequency detail energy between channels — a
material property, measurable from one colour frame, and independent of pose,
motion, expression and identity.

Honest scope: a very high-quality print or replay can partially inherit the
original face's subsurface blur along with everything else it copies, so this
leg is weighted moderately rather than treated as decisive on its own. Where it
*is* decisive is the case nothing else covers — the 3-D mask.
"""

from __future__ import annotations

import numpy as np

from ..types import DetectorResult
from .base import as_clip

#: Red/green high-frequency detail ratio for real skin. 1.0 would mean the
#: channels carry identical detail, i.e. an opaque surface.
_SKIN_RATIO = 0.72


class SubsurfaceScatteringDetector:
    name = "subsurface"

    def __init__(self, min_side: int = 48):
        self.min_side = int(min_side)

    def __call__(self, frames: np.ndarray) -> DetectorResult:
        clip = as_clip(frames)
        if clip.shape[-1] < 3:
            return DetectorResult.abstain(self.name, "no_colour")
        h, w = clip.shape[1:3]
        if min(h, w) < self.min_side:
            return DetectorResult.abstain(self.name, "crop_too_small")

        # Median over time removes pulse, noise and expression, leaving a clean
        # per-channel image to compare detail in.
        rgb = np.median(clip[..., :3], axis=0)
        detail = np.array([_high_frequency_energy(rgb[..., c]) for c in range(3)])
        if detail[1] <= 1e-12 or detail[2] <= 1e-12:
            return DetectorResult.abstain(self.name, "no_detail")

        gb = float(np.sqrt(detail[1] * detail[2]))     # green/blue detail reference
        ratio = float(np.sqrt(detail[0]) / (np.sqrt(gb) + 1e-12))

        # ratio ~= _SKIN_RATIO -> translucent skin; ratio ~= 1 -> opaque material.
        translucency = float(np.clip((1.0 - ratio) / (1.0 - _SKIN_RATIO), 0.0, 1.2))
        score = float(np.clip(0.06 + 0.86 * translucency, 0.02, 0.95))

        # Detail is the measurement's substrate: a flat, textureless or
        # over-smoothed crop cannot support the comparison at all.
        reliability = float(np.clip(np.sqrt(gb) / 6e-3, 0.0, 1.0))
        return DetectorResult(
            self.name,
            score,
            reliability,
            {
                "red_detail": float(np.sqrt(detail[0])),
                "green_blue_detail": float(np.sqrt(gb)),
                "rg_detail_ratio": ratio,
                "translucency": translucency,
            },
        )


def _high_frequency_energy(channel: np.ndarray) -> float:
    """Variance of the spatially high-passed channel — its fine-detail content."""
    from scipy.ndimage import gaussian_filter

    fine = channel - gaussian_filter(channel, 1.5)
    return float((fine**2).mean())
