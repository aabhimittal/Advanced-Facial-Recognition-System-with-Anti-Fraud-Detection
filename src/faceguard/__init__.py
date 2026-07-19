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

__version__ = "0.1.0"

__all__ = [
    "FaceGuardPipeline",
    "FaceGuardConfig",
    "DetectorResult",
    "FraudVerdict",
    "PipelineResult",
    "__version__",
]
