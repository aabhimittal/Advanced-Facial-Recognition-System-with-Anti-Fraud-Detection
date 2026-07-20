"""Learn STLF fusion weights from a labelled PAD set.

STLF is *linear in log-odds*::

    L = logit(p₀) + Σᵢ wᵢ · (rᵢ · logit(sᵢ))

so if we treat ``xᵢ = rᵢ · logit(sᵢ)`` as features and the genuine/spoof label as
the target, fitting a **logistic regression** recovers exactly the quantities the
fusion needs: the coefficients are the per-detector weights ``wᵢ`` and the
intercept is ``logit(p₀)``. No separate model, no new inference path — the
learned parameters drop straight back into :class:`FaceGuardConfig`.

By default weights are constrained to be **non-negative** (projected gradient),
preserving the STLF semantics that a detector is a unit of *trust* — a higher
liveness score must never push the fused decision toward "spoof".

Dependency-free: plain numpy gradient descent, so it runs in CI.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Mapping, Sequence

import numpy as np

from ..config import FaceGuardConfig
from ..types import DetectorResult

_EPS = 1e-6
Sample = Mapping[str, DetectorResult]  # e.g. PipelineResult.detectors


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


class WeightCalibrator:
    """Fit ``detector_weights`` and ``genuine_prior`` by logistic regression."""

    def __init__(
        self,
        base_config: FaceGuardConfig | None = None,
        l2: float = 1e-3,
        non_negative: bool = True,
        iters: int = 2000,
        lr: float = 0.2,
    ):
        self.base_config = base_config or FaceGuardConfig()
        self.l2 = l2
        self.non_negative = non_negative
        self.iters = iters
        self.lr = lr

    def fit(self, samples: Sequence[Sample], labels: Sequence[int]) -> FaceGuardConfig:
        """Return a new config whose weights/prior are learned from the data.

        ``samples[k]`` maps detector name -> its :class:`DetectorResult` for the
        k-th example; ``labels[k]`` is 1 for genuine (live), 0 for spoof.
        """
        if len(samples) != len(labels):
            raise ValueError("samples and labels must have equal length")
        names = self._detector_names(samples)
        X = self._featurize(samples, names)                 # (N, D)
        y = np.asarray(labels, dtype=np.float64)            # (N,)

        w, b = self._logistic_fit(X, y)
        weights = {name: float(w[i]) for i, name in enumerate(names)}
        prior = 1.0 / (1.0 + math.exp(-b))
        prior = float(min(max(prior, 1e-3), 1.0 - 1e-3))
        return replace(self.base_config, detector_weights=weights, genuine_prior=prior)

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _detector_names(samples: Sequence[Sample]) -> List[str]:
        names: List[str] = []
        for s in samples:
            for n in s:
                if n not in names:
                    names.append(n)
        return names

    @staticmethod
    def _featurize(samples: Sequence[Sample], names: Sequence[str]) -> np.ndarray:
        X = np.zeros((len(samples), len(names)))
        for k, s in enumerate(samples):
            for i, name in enumerate(names):
                det = s.get(name)
                if det is not None:
                    # Reliability-scaled log-odds — the exact STLF contribution term.
                    X[k, i] = det.reliability * np.clip(_logit(det.score), -8.0, 8.0)
        return X

    def _logistic_fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        for _ in range(self.iters):
            z = X @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            err = p - y
            grad_w = X.T @ err / n + self.l2 * w
            grad_b = float(err.mean())
            w -= self.lr * grad_w
            b -= self.lr * grad_b
            if self.non_negative:
                np.clip(w, 0.0, None, out=w)
        return w, b


def fit_fusion_weights(
    samples: Sequence[Sample], labels: Sequence[int], **kwargs
) -> FaceGuardConfig:
    """Convenience wrapper: fit and return a calibrated :class:`FaceGuardConfig`."""
    return WeightCalibrator(**kwargs).fit(samples, labels)
