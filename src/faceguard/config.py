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
            # Geometric certificate of 3-D structure; abstains without motion.
            "parallax": 0.9,
            # Photon-transfer check: the only leg that sees injection attacks.
            "sensor": 1.1,
            # Rolling-shutter beat: near-conclusive when present, silent when not.
            "banding": 1.0,
            # Material cue: decisive against 3-D masks, weak against prints
            # (a photo of a face inherits that face's subsurface blur).
            "subsurface": 0.8,
        }
    )

    # --- Fusion robustness ---
    # Maximum log-odds any single detector may contribute. logit(1-1e-6) ~= 13.8,
    # so without a cap one saturated cue would overrule every other. 4.0 still
    # lets a confident detector move the posterior from 0.5 to ~0.98 alone.
    max_detector_logit: float = 4.0

    # --- Capture quality gate (see faceguard.quality) ---
    # Samples below this quality are answered INDETERMINATE ("please retry"),
    # never FRAUD: bad capture must not be charged to the user as an accusation.
    min_capture_quality: float = 0.25
    min_face_px: int = 64
    require_colour: bool = False
    require_clip: bool = False

    # --- Availability / SLA ---
    # Soft wall-clock budget for the liveness stage. Detectors not yet run when
    # the budget is exhausted abstain instead of blowing the latency SLA.
    time_budget_ms: float = 0.0  # 0 = unlimited
    # Minimum fraction of configured detectors that must actually report for a
    # verdict to be issued at all; below it the result is INDETERMINATE. This is
    # what stops a half-broken deployment from quietly degrading into a
    # coin-flip that still *looks* authoritative.
    min_detector_availability: float = 0.5

    # --- Recognition ---
    match_threshold: float = 0.6  # cosine distance; lower = more similar

    def __post_init__(self) -> None:
        if not (0.0 <= self.fraud_threshold <= self.genuine_threshold <= 1.0):
            raise ValueError("Require 0 <= fraud_threshold <= genuine_threshold <= 1")
        if self.max_detector_logit <= 0:
            raise ValueError("max_detector_logit must be positive")
        if not 0.0 <= self.min_detector_availability <= 1.0:
            raise ValueError("min_detector_availability must be in [0, 1]")
        if any(w < 0 for w in self.detector_weights.values()):
            raise ValueError("detector weights must be non-negative")
