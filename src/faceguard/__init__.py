"""FaceGuard — facial recognition with novel Spectro-Temporal Liveness Fusion.

Public API
----------
    from faceguard import FaceGuardPipeline, PipelineResult

The anti-fraud core (liveness detectors + fusion) depends only on numpy/scipy.
Face detection and deep embeddings are optional pluggable adapters.

Beyond the detectors, the package ships the parts a deployment actually needs:
a capture-quality gate that answers "retry" instead of "fraud" on a bad frame
(:mod:`faceguard.quality`), risk-tiered operating points
(:mod:`faceguard.policy`), sequential decisioning for live streams
(:mod:`faceguard.sequential`), cancellable templates so a gallery breach is
survivable (:mod:`faceguard.recognition.protection`), and a tamper-evident
decision log (:mod:`faceguard.audit`).
"""

from .audit import AuditLog, AuditRecord
from .challenge import Challenge, ChallengeIssuer, ChallengeResult, ChallengeType
from .config import FaceGuardConfig
from .fusion import WeightCalibrator, fit_fusion_weights
from .pipeline import FaceGuardPipeline
from .policy import RiskTier, TierPolicy, config_for, policy_for
from .quality import CaptureQualityGate, QualityReport
from .recognition.protection import ProtectedMatcher, TemplateProtector
from .sequential import SequentialDecision, SequentialState, SequentialVerifier, sequential_verdict
from .types import DetectorResult, FraudVerdict, PipelineResult

__version__ = "0.3.0"

__all__ = [
    "AuditLog",
    "AuditRecord",
    "CaptureQualityGate",
    "Challenge",
    "ChallengeIssuer",
    "ChallengeResult",
    "ChallengeType",
    "DetectorResult",
    "FaceGuardConfig",
    "FaceGuardPipeline",
    "FraudVerdict",
    "PipelineResult",
    "ProtectedMatcher",
    "QualityReport",
    "RiskTier",
    "SequentialDecision",
    "SequentialState",
    "SequentialVerifier",
    "TemplateProtector",
    "TierPolicy",
    "WeightCalibrator",
    "__version__",
    "config_for",
    "fit_fusion_weights",
    "policy_for",
    "sequential_verdict",
]
