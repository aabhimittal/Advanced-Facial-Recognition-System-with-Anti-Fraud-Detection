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

from .types import DetectorResult, FraudVerdict, PipelineResult
from .pipeline import FaceGuardPipeline
from .config import FaceGuardConfig
from .fusion import fit_fusion_weights, WeightCalibrator
from .challenge import Challenge, ChallengeIssuer, ChallengeType, ChallengeResult
from .quality import CaptureQualityGate, QualityReport
from .policy import RiskTier, TierPolicy, config_for, policy_for
from .sequential import SequentialVerifier, SequentialDecision, SequentialState, sequential_verdict
from .audit import AuditLog, AuditRecord
from .recognition.protection import ProtectedMatcher, TemplateProtector

__version__ = "0.3.0"

__all__ = [
    "FaceGuardPipeline",
    "FaceGuardConfig",
    "DetectorResult",
    "FraudVerdict",
    "PipelineResult",
    "fit_fusion_weights",
    "WeightCalibrator",
    "Challenge",
    "ChallengeIssuer",
    "ChallengeType",
    "ChallengeResult",
    "CaptureQualityGate",
    "QualityReport",
    "RiskTier",
    "TierPolicy",
    "config_for",
    "policy_for",
    "SequentialVerifier",
    "SequentialDecision",
    "SequentialState",
    "sequential_verdict",
    "AuditLog",
    "AuditRecord",
    "ProtectedMatcher",
    "TemplateProtector",
    "__version__",
]
