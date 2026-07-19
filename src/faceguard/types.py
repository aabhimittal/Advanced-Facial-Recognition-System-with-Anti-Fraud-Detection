"""Shared, dependency-free data types used across FaceGuard."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


@dataclass
class DetectorResult:
    """Output of a single liveness detector.

    Attributes
    ----------
    name:
        Detector identifier (e.g. ``"spectral"``).
    score:
        Liveness score in ``[0, 1]``. Higher means *more likely a genuine,
        live face*; lower means *more likely a spoof / fraud*.
    reliability:
        Self-estimated confidence in ``[0, 1]`` that this measurement is
        trustworthy for this particular sample. A detector lowers its own
        reliability when the signal quality is poor (e.g. too few frames for
        rPPG, or too small a crop for a spectral estimate). This is the key
        input that lets the fusion stage down-weight unreliable evidence.
    detail:
        Optional human-readable / numeric diagnostics for inspection.
    """

    name: str
    score: float
    reliability: float
    detail: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = float(_clip01(self.score))
        self.reliability = float(_clip01(self.reliability))


class FraudVerdict(str, Enum):
    GENUINE = "genuine"
    SUSPICIOUS = "suspicious"
    FRAUD = "fraud"


@dataclass
class PipelineResult:
    """Full output of :class:`faceguard.pipeline.FaceGuardPipeline`."""

    liveness_score: float
    reliability: float
    verdict: FraudVerdict
    detectors: Dict[str, DetectorResult] = field(default_factory=dict)
    identity: Optional[str] = None
    match_distance: Optional[float] = None
    # Set when a SUSPICIOUS case was escalated to an active challenge-response.
    challenge_passed: Optional[bool] = None

    def is_trustworthy_match(self) -> bool:
        """True only when the face both matched an identity and passed liveness."""
        return self.verdict == FraudVerdict.GENUINE and self.identity is not None

    def summary(self) -> str:
        idt = self.identity or "unknown"
        return (
            f"verdict={self.verdict.value} live={self.liveness_score:.3f} "
            f"conf={self.reliability:.3f} identity={idt}"
        )


def _clip01(x: float) -> float:
    if x != x:  # NaN guard
        return 0.0
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
