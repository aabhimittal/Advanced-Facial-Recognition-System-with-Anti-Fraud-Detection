"""Central configuration for the FaceGuard pipeline.

Every tunable threshold lives here so experiments are reproducible and the
science stays separate from the wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class FaceGuardConfig:
    # --- Fusion decision thresholds (on the fused liveness score in [0,1]) ---
    genuine_threshold: float = 0.62
    fraud_threshold: float = 0.42
    # Below `fraud_threshold` -> FRAUD, above `genuine_threshold` -> GENUINE,
    # in between -> SUSPICIOUS (ask for a second sample / challenge-response).

    # If total evidence reliability is below this, never return GENUINE:
    # low-confidence "genuine" is downgraded to SUSPICIOUS.
    min_reliability_for_genuine: float = 0.35

    # --- Prior probability that an incoming face is genuine (Bayesian fusion) ---
    genuine_prior: float = 0.5

    # --- Per-detector base trust (learned/tuned weights, multiply reliability) ---
    detector_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "spectral": 1.0,
            "rppg": 1.0,
            "texture": 0.9,
            "motion": 0.8,
            "dct": 1.0,
        }
    )

    # --- Recognition ---
    match_threshold: float = 0.6  # cosine distance; lower = more similar

    def __post_init__(self) -> None:
        if not (0.0 <= self.fraud_threshold <= self.genuine_threshold <= 1.0):
            raise ValueError("Require 0 <= fraud_threshold <= genuine_threshold <= 1")
