"""FaceGuard — facial recognition with novel Spectro-Temporal Liveness Fusion.

Public API
----------
    from faceguard import FaceGuardPipeline, PipelineResult

The anti-fraud core (liveness detectors + fusion) depends only on numpy/scipy.
Face detection and deep embeddings are optional pluggable adapters.
"""

from .types import DetectorResult, FraudVerdict, PipelineResult
from .pipeline import FaceGuardPipeline
from .config import FaceGuardConfig
from .fusion import fit_fusion_weights, WeightCalibrator
from .challenge import Challenge, ChallengeType, ChallengeResult

__version__ = "0.2.0"

__all__ = [
    "FaceGuardPipeline",
    "FaceGuardConfig",
    "DetectorResult",
    "FraudVerdict",
    "PipelineResult",
    "fit_fusion_weights",
    "WeightCalibrator",
    "Challenge",
    "ChallengeType",
    "ChallengeResult",
    "__version__",
]
