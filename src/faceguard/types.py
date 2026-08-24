"""Shared, dependency-free data types used across FaceGuard."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for type checkers
    from .quality import QualityReport


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

    @classmethod
    def abstain(cls, name: str, reason: str) -> "DetectorResult":
        """A neutral, zero-weight result.

        Used whenever a detector cannot honestly speak: not enough frames, a
        crop too small, an unhandled exception, or a skipped run under a latency
        budget. Because ``logit(0.5) = 0`` and ``reliability = 0``, an abstention
        is mathematically inert in the fusion — a broken detector degrades the
        system's *confidence*, never its *accusation*.
        """
        return cls(name, 0.5, 0.0, {f"abstain_{reason}": 1.0})


class FraudVerdict(str, Enum):
    GENUINE = "genuine"
    SUSPICIOUS = "suspicious"
    FRAUD = "fraud"
    #: The sample was never judged: capture quality (or a detector outage) left
    #: too little trustworthy evidence to make *any* claim. Crucially this is
    #: **not** an accusation — a dark room or a smeared lens must produce a
    #: "please retry", never a fraud flag. Industrial deployments care about this
    #: distinction because a false FRAUD costs a customer, a false GENUINE costs
    #: money, and conflating "no signal" with "attack" produces both.
    INDETERMINATE = "indeterminate"


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
    # Capture-quality verdict; when unusable the pipeline returns INDETERMINATE.
    quality: Optional["QualityReport"] = None
    # Names of detectors that failed or were skipped (outage / time budget).
    degraded: Tuple[str, ...] = ()
    # Wall-clock cost of the liveness stage, for SLA monitoring.
    elapsed_ms: float = 0.0

    def is_trustworthy_match(self) -> bool:
        """True only when the face both matched an identity and passed liveness."""
        return self.verdict == FraudVerdict.GENUINE and self.identity is not None

    @property
    def needs_retry(self) -> bool:
        """True when the *system*, not the subject, failed to produce a decision."""
        return self.verdict == FraudVerdict.INDETERMINATE

    def summary(self) -> str:
        idt = self.identity or "unknown"
        deg = f" degraded={','.join(self.degraded)}" if self.degraded else ""
        return (
            f"verdict={self.verdict.value} live={self.liveness_score:.3f} "
            f"conf={self.reliability:.3f} identity={idt}{deg}"
        )


def _clip01(x: float) -> float:
    if x != x:  # NaN guard
        return 0.0
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
