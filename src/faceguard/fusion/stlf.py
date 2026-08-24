"""Spectro-Temporal Liveness Fusion (STLF) — the novel core of FaceGuard.

The idea
--------
Any single anti-spoofing cue can be defeated by an attacker who targets it: a
high-resolution print beats naive texture checks, a good display beats simple
colour statistics, a replayed video can even carry a faint pulse. STLF's premise
is that a fraudster rarely defeats *orthogonal* cues **simultaneously**, and —
crucially — that a defender should **weight each cue by how trustworthy it is on
this specific sample**.

Most published liveness systems fuse cues with *fixed* weights (or concatenate
features into one classifier). STLF instead treats fusion as a **confidence-
weighted Bayesian log-opinion pool**: each detector reports a liveness score
*and a self-estimated reliability*, and the reliability directly scales how far
that detector is allowed to move the posterior. A detector facing a degraded
signal (too few frames for rPPG, a crop too small for a clean spectrum) reports
low reliability and *abstains automatically* — no hand-tuned gating required.

The maths
---------
Working in log-odds (logits), with prior ``p0`` and per-detector base trust
``w_i`` (config) and self-reported reliability ``r_i``::

    L = logit(p0) + Σ_i (w_i · r_i) · logit(s_i)
    p_fused = sigmoid(L)

Each per-detector term is additionally capped at ``±max_detector_logit`` so a
single saturated cue cannot act as an unearned veto over the whole pool.

Because ``logit(0.5) = 0``, a score of exactly 0.5 (pure abstention) contributes
nothing, and a detector with ``r_i = 0`` drops out entirely — the pool degrades
gracefully to whatever evidence *is* trustworthy. The fused confidence grows
with the total reliable evidence gathered::

    R = 1 - exp(-Σ_i w_i · r_i)

This makes STLF *auditable* (every cue's contribution to the log-odds is
inspectable) and *adaptive* (weights follow signal quality per sample) — the two
properties fixed-weight fusion lacks.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np

from ..config import FaceGuardConfig
from ..types import DetectorResult, FraudVerdict

_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class SpectroTemporalLivenessFusion:
    """Confidence-weighted Bayesian fusion of liveness detectors."""

    def __init__(self, config: FaceGuardConfig | None = None):
        self.config = config or FaceGuardConfig()

    def fuse(self, results: Iterable[DetectorResult]) -> Tuple[float, float, FraudVerdict, Dict[str, float]]:
        cfg = self.config
        log_odds = _logit(cfg.genuine_prior)
        total_evidence = 0.0
        contributions: Dict[str, float] = {}

        cap = cfg.max_detector_logit
        for r in results:
            w = cfg.detector_weights.get(r.name, 1.0)
            eff = w * r.reliability            # effective pooling weight
            # Saturation guard: a detector reporting exactly 0 or 1 would
            # contribute ±13.8 log-odds and single-handedly overrule every other
            # cue — a de-facto veto that no individual detector has earned, and
            # a real failure mode when a heuristic pins at its rail. Capping the
            # per-detector contribution keeps the pool a *pool*: any one leg can
            # argue forcefully, none can dictate.
            contribution = float(np.clip(eff * _logit(r.score), -cap, cap))
            log_odds += contribution
            total_evidence += eff
            contributions[r.name] = contribution

        fused_score = _sigmoid(log_odds)
        confidence = 1.0 - math.exp(-total_evidence)
        verdict = self._decide(fused_score, confidence)
        return fused_score, confidence, verdict, contributions

    def _decide(self, score: float, confidence: float) -> FraudVerdict:
        cfg = self.config
        if score <= cfg.fraud_threshold:
            return FraudVerdict.FRAUD
        if score >= cfg.genuine_threshold:
            # A confident-looking "genuine" with too little reliable evidence
            # behind it is downgraded — better to challenge than to trust blindly.
            if confidence >= cfg.min_reliability_for_genuine:
                return FraudVerdict.GENUINE
            return FraudVerdict.SUSPICIOUS
        return FraudVerdict.SUSPICIOUS


def fuse(results, config: FaceGuardConfig | None = None):
    """Convenience wrapper around :class:`SpectroTemporalLivenessFusion`."""
    return SpectroTemporalLivenessFusion(config).fuse(results)
