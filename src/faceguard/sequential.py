"""Sequential verification — deciding as fast as the evidence allows.

The problem with a fixed window
-------------------------------
Every liveness system in the literature analyses a fixed capture: "record three
seconds, then decide". That is the wrong shape for a real deployment. A blatant
print attack is obvious in 300 ms, and holding the queue for another 2.7 s is
pure friction; a marginal capture in bad light is *still* marginal at three
seconds, and deciding anyway just to respect the clock is how false accepts
happen.

Wald's sequential probability ratio test (SPRT) inverts this. Instead of fixing
the sample size and accepting whatever error rate falls out, you fix the error
rates you are willing to tolerate and let the *sample size* vary::

    A = log((1 - β) / α)        upper (accept-live) boundary
    B = log(β / (1 - α))        lower (reject) boundary

with ``α`` the false-accept rate and ``β`` the false-reject rate you are
targeting. Accumulate the per-window log-likelihood ratio; cross ``A`` and you
accept, cross ``B`` and you reject, and until then you keep watching. The SPRT
is optimal in the sense that no other test achieving the same error rates needs
fewer samples on average.

Why it composes so naturally here
---------------------------------
STLF already produces its posterior in log-odds, and log-odds *is* the
log-likelihood ratio relative to a 0.5 prior. So each new window's fused score
drops straight into the accumulator with no extra modelling::

    Λ ← Λ + logit(p_window) · min(1, reliability_window)

Reliability scaling matters: a window shot while the user was walking into frame
should push the accumulator less far than a clean one. Low-quality windows
therefore slow the decision instead of corrupting it — which is exactly the
behaviour an operator wants, and the opposite of what a fixed-window system
does.

Use it for a live camera feed, for multi-attempt retries at a kiosk, or to fuse
several short bursts without ever pretending you saw one long clip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from .types import FraudVerdict, PipelineResult

_EPS = 1e-6


class SequentialState(str, Enum):
    ACCEPT = "accept"        # boundary crossed: live
    REJECT = "reject"        # boundary crossed: attack
    CONTINUE = "continue"    # keep sampling
    EXHAUSTED = "exhausted"  # ran out of windows without crossing either


@dataclass
class SequentialDecision:
    state: SequentialState
    log_likelihood_ratio: float
    windows: int
    upper: float
    lower: float
    #: Posterior probability of "live" implied by the accumulator.
    probability: float = 0.5
    history: List[float] = field(default_factory=list)

    @property
    def verdict(self) -> FraudVerdict:
        """Map the sequential state onto the pipeline's verdict vocabulary."""
        if self.state is SequentialState.ACCEPT:
            return FraudVerdict.GENUINE
        if self.state is SequentialState.REJECT:
            return FraudVerdict.FRAUD
        # Still undecided (or out of budget) is *not* an accusation.
        return FraudVerdict.SUSPICIOUS

    def summary(self) -> str:
        return (
            f"{self.state.value} after {self.windows} window(s): "
            f"LLR={self.log_likelihood_ratio:+.2f} in [{self.lower:.2f}, {self.upper:.2f}] "
            f"p_live={self.probability:.3f}"
        )


class SequentialVerifier:
    """Wald SPRT over a stream of per-window liveness posteriors.

    Parameters
    ----------
    far, frr:
        Target false-accept and false-reject rates. These are *operating
        targets*, not measured guarantees: the SPRT bounds hold under the usual
        i.i.d.-evidence assumption, which consecutive windows of one capture only
        approximately satisfy (an attacker holds the same photo up the whole
        time). Correlated evidence makes the real error rates worse than the
        nominal ones, so treat these as a knob calibrated on your own data.
    max_windows:
        Hard cap so a borderline subject cannot hold a lane open forever.
    """

    def __init__(self, far: float = 1e-3, frr: float = 0.02, max_windows: int = 12):
        if not 0.0 < far < 1.0 or not 0.0 < frr < 1.0:
            raise ValueError("far and frr must be in (0, 1)")
        if max_windows < 1:
            raise ValueError("max_windows must be >= 1")
        self.far = float(far)
        self.frr = float(frr)
        self.max_windows = int(max_windows)
        self.upper = math.log((1.0 - self.frr) / self.far)
        self.lower = math.log(self.frr / (1.0 - self.far))
        self.reset()

    def reset(self) -> None:
        self._llr = 0.0
        self._n = 0
        self._history: List[float] = []

    # -- streaming interface ------------------------------------------------
    def update(self, score: float, reliability: float = 1.0) -> SequentialDecision:
        """Fold one window's fused liveness score into the accumulator."""
        weight = min(1.0, max(0.0, float(reliability)))
        self._llr += weight * _logit(score)
        self._n += 1
        self._history.append(self._llr)
        return self._decision()

    def observe(self, result: PipelineResult) -> SequentialDecision:
        """Fold a whole :class:`PipelineResult` in, respecting its own caveats.

        An INDETERMINATE window contributes *nothing* — it is a capture failure,
        not evidence — but it still consumes a window, so a camera that never
        produces a usable frame ends in EXHAUSTED rather than looping forever.
        """
        if result.verdict is FraudVerdict.INDETERMINATE:
            self._n += 1
            self._history.append(self._llr)
            return self._decision()
        return self.update(result.liveness_score, result.reliability)

    def run(self, results) -> SequentialDecision:
        """Consume an iterable of results, stopping the moment a boundary is hit."""
        decision = self._decision()
        for result in results:
            decision = self.observe(result)
            if decision.state in (SequentialState.ACCEPT, SequentialState.REJECT):
                return decision
        if decision.state is SequentialState.CONTINUE:
            # The stream ended before either boundary was reached. Saying
            # CONTINUE to a caller with nothing left to feed us would be a lie;
            # EXHAUSTED tells them the truth: undecided, out of evidence.
            decision.state = SequentialState.EXHAUSTED
        return decision

    # -- internals ----------------------------------------------------------
    def _decision(self) -> SequentialDecision:
        if self._llr >= self.upper:
            state = SequentialState.ACCEPT
        elif self._llr <= self.lower:
            state = SequentialState.REJECT
        elif self._n >= self.max_windows:
            state = SequentialState.EXHAUSTED
        else:
            state = SequentialState.CONTINUE
        return SequentialDecision(
            state=state,
            log_likelihood_ratio=self._llr,
            windows=self._n,
            upper=self.upper,
            lower=self.lower,
            probability=_sigmoid(self._llr),
            history=list(self._history),
        )

    def expected_windows(self, mean_step_live: float, mean_step_attack: float) -> Tuple[float, float]:
        """Wald's average sample number, given measured per-window LLR steps.

        Capacity planning without a trial: feed in the mean per-window log-odds
        your own data produces for genuine and attack samples, and this returns
        how many windows the average subject of each kind will need. Both steps
        must be measured — deriving them from the error targets would be
        circular, and the resulting numbers would mean nothing.
        """
        if mean_step_live <= 0 or mean_step_attack >= 0:
            raise ValueError(
                "mean_step_live must be positive and mean_step_attack negative; "
                "if they are not, the detector ensemble is not separating classes"
            )
        return (
            min(float(self.max_windows), self.upper / mean_step_live),
            min(float(self.max_windows), self.lower / mean_step_attack),
        )


def _logit(p: float) -> float:
    p = min(max(float(p), _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def sequential_verdict(
    results, far: float = 1e-3, frr: float = 0.02, max_windows: int = 12
) -> SequentialDecision:
    """One-shot convenience wrapper around :class:`SequentialVerifier`."""
    return SequentialVerifier(far=far, frr=frr, max_windows=max_windows).run(results)


def windows(clip, size: int = 30, stride: Optional[int] = None):
    """Split a long clip into overlapping analysis windows.

    Stride defaults to half the window, so a transient event (a blink, a flicker
    of banding, a moment of parallax) cannot fall between two windows and be
    missed by both.
    """
    import numpy as np

    arr = np.asarray(clip)
    step = int(stride or max(1, size // 2))
    out = []
    for start in range(0, max(1, len(arr) - size + 1), step):
        chunk = arr[start:start + size]
        if len(chunk) >= min(size, len(arr)):
            out.append(chunk)
    return out or [arr]
