"""Risk-tiered decision policy — one detector stack, many operating points.

Why a single threshold is always wrong
--------------------------------------
Unlocking a phone to check the weather and authorising a six-figure wire
transfer are not the same decision, but a single ``genuine_threshold`` treats
them identically. Deployments respond by either running the strict policy
everywhere (and drowning support in false rejects) or the loose one everywhere
(and paying for the fraud).

The right structure is one ensemble, several **operating points**, selected per
transaction by what is at stake. This module names four tiers, states plainly
what each trades away, and derives a :class:`~faceguard.config.FaceGuardConfig`
for each so the choice is a parameter of the *request*, not a property of the
deployment.

The escalation ladder is what makes strict tiers usable: a tier does not
respond to doubt by rejecting the user, it responds by *asking for more* — a
challenge-response, a longer capture, or a second factor. Friction is spent only
on ambiguous cases, which are a small minority of traffic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

from .config import FaceGuardConfig
from .types import FraudVerdict, PipelineResult


class RiskTier(str, Enum):
    """What is at stake behind this authentication."""

    CONVENIENCE = "convenience"  # unlock a screen, resume a session
    STANDARD = "standard"        # sign in, open an account view
    ELEVATED = "elevated"        # payment, profile or credential change
    CRITICAL = "critical"        # high-value transfer, vault, physical access


@dataclass(frozen=True)
class TierPolicy:
    """The complete operating point for one risk tier."""

    tier: RiskTier
    genuine_threshold: float
    fraud_threshold: float
    min_reliability_for_genuine: float
    min_capture_quality: float
    #: Escalate to an active challenge whenever the passive verdict is SUSPICIOUS.
    challenge_on_suspicious: bool
    #: Require a challenge even when the passive verdict is already GENUINE.
    always_challenge: bool
    #: Minimum clip length worth analysing at this tier, in frames at 30 fps.
    min_frames: int
    #: One-line statement of the trade this tier makes.
    rationale: str

    def to_config(self, base: Optional[FaceGuardConfig] = None) -> FaceGuardConfig:
        """Derive a pipeline config, preserving any base tuning (e.g. learned weights)."""
        base = base or FaceGuardConfig()
        return replace(
            base,
            genuine_threshold=self.genuine_threshold,
            fraud_threshold=self.fraud_threshold,
            min_reliability_for_genuine=self.min_reliability_for_genuine,
            min_capture_quality=self.min_capture_quality,
        )

    def requires_challenge(self, result: PipelineResult) -> bool:
        """Whether this result must be escalated to an active challenge."""
        if self.always_challenge:
            return result.verdict is not FraudVerdict.FRAUD
        return self.challenge_on_suspicious and result.verdict is FraudVerdict.SUSPICIOUS


_POLICIES = {
    RiskTier.CONVENIENCE: TierPolicy(
        tier=RiskTier.CONVENIENCE,
        genuine_threshold=0.45,
        fraud_threshold=0.20,
        min_reliability_for_genuine=0.15,
        min_capture_quality=0.15,
        challenge_on_suspicious=False,
        always_challenge=False,
        min_frames=8,
        rationale=(
            "Optimised for never annoying the user. Accepts weak evidence because "
            "the cost of a false accept is a screen unlock, and the cost of a "
            "false reject is a user who stops using the feature."
        ),
    ),
    RiskTier.STANDARD: TierPolicy(
        tier=RiskTier.STANDARD,
        genuine_threshold=0.62,
        fraud_threshold=0.42,
        min_reliability_for_genuine=0.35,
        min_capture_quality=0.25,
        challenge_on_suspicious=True,
        always_challenge=False,
        min_frames=30,
        rationale=(
            "The balanced default: passive-first, escalating to a challenge only "
            "for the ambiguous middle band, so typical traffic sees no friction."
        ),
    ),
    RiskTier.ELEVATED: TierPolicy(
        tier=RiskTier.ELEVATED,
        genuine_threshold=0.78,
        fraud_threshold=0.50,
        min_reliability_for_genuine=0.60,
        min_capture_quality=0.35,
        challenge_on_suspicious=True,
        always_challenge=False,
        min_frames=60,
        rationale=(
            "Demands both a high posterior and substantial reliable evidence "
            "behind it, so a lucky capture with two working detectors cannot "
            "authorise a payment."
        ),
    ),
    RiskTier.CRITICAL: TierPolicy(
        tier=RiskTier.CRITICAL,
        genuine_threshold=0.88,
        fraud_threshold=0.55,
        min_reliability_for_genuine=0.80,
        min_capture_quality=0.45,
        challenge_on_suspicious=True,
        always_challenge=True,
        min_frames=90,
        rationale=(
            "Passive liveness alone never suffices: every non-fraud outcome is "
            "escalated to an unpredictable challenge, because a replay that "
            "defeats all passive cues still cannot answer a prompt it has never "
            "seen."
        ),
    ),
}


def policy_for(tier: RiskTier | str) -> TierPolicy:
    """Look up the operating point for a tier (accepts the enum or its value)."""
    if isinstance(tier, str):
        tier = RiskTier(tier)
    return _POLICIES[tier]


def config_for(tier: RiskTier | str, base: Optional[FaceGuardConfig] = None) -> FaceGuardConfig:
    """Shorthand for ``policy_for(tier).to_config(base)``."""
    return policy_for(tier).to_config(base)


def tiers() -> list:
    """All policies, ordered from most permissive to most stringent."""
    return [_POLICIES[t] for t in RiskTier]
