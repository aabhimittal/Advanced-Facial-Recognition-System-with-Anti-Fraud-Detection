"""Liveness / anti-spoofing detectors.

Each detector consumes a face crop (single image) or a short clip (stack of
frames) and returns a :class:`~faceguard.types.DetectorResult` carrying both a
liveness *score* and a *reliability* estimate. The reliability is what makes
the downstream Spectro-Temporal Liveness Fusion adaptive.
"""

from .banding import DisplayBandingDetector
from .dct import DCTDeepfakeDetector
from .motion import MotionDetector
from .parallax import ParallaxDetector
from .rppg import RPPGDetector
from .sensor import SensorNoiseDetector
from .spectral import SpectralDetector
from .subsurface import SubsurfaceScatteringDetector
from .texture import TextureDetector

__all__ = [
    "DCTDeepfakeDetector",
    "DisplayBandingDetector",
    "MotionDetector",
    "ParallaxDetector",
    "RPPGDetector",
    "SensorNoiseDetector",
    "SpectralDetector",
    "SubsurfaceScatteringDetector",
    "TextureDetector",
]
