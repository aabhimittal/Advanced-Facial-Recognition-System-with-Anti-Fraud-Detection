"""Liveness / anti-spoofing detectors.

Each detector consumes a face crop (single image) or a short clip (stack of
frames) and returns a :class:`~faceguard.types.DetectorResult` carrying both a
liveness *score* and a *reliability* estimate. The reliability is what makes
the downstream Spectro-Temporal Liveness Fusion adaptive.
"""

from .spectral import SpectralDetector
from .rppg import RPPGDetector
from .texture import TextureDetector
from .motion import MotionDetector

__all__ = ["SpectralDetector", "RPPGDetector", "TextureDetector", "MotionDetector"]
