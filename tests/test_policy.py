"""Risk-tiered operating points: one ensemble, several decisions."""

import pytest

from faceguard import FaceGuardPipeline, FraudVerdict, RiskTier, config_for, policy_for
from faceguard.policy import tiers
from faceguard.types import PipelineResult
from faceguard.utils import synth_live_clip, synth_spoof_clip


def test_tiers_are_monotonically_stricter():
    ordered = tiers()
    for lower, higher in zip(ordered, ordered[1:]):
        assert higher.genuine_threshold >= lower.genuine_threshold
        assert higher.min_reliability_for_genuine >= lower.min_reliability_for_genuine
        assert higher.min_capture_quality >= lower.min_capture_quality
        assert higher.min_frames >= lower.min_frames


def test_policy_lookup_accepts_enum_or_string():
    assert policy_for("critical") is policy_for(RiskTier.CRITICAL)
    with pytest.raises(ValueError):
        policy_for("whatever-tier")


def test_config_preserves_base_tuning_such_as_learned_weights():
    from faceguard.config import FaceGuardConfig

    base = FaceGuardConfig(detector_weights={"spectral": 2.5}, match_threshold=0.42)
    cfg = config_for(RiskTier.ELEVATED, base)
    assert cfg.detector_weights == {"spectral": 2.5}
    assert cfg.match_threshold == 0.42
    assert cfg.genuine_threshold == policy_for(RiskTier.ELEVATED).genuine_threshold


@pytest.mark.parametrize("tier", list(RiskTier))
def test_every_tier_still_rejects_a_spoof(tier):
    pipe = FaceGuardPipeline.for_tier(tier)
    assert pipe.analyze(synth_spoof_clip(seed=0)).verdict is FraudVerdict.FRAUD


@pytest.mark.parametrize("tier", list(RiskTier))
def test_no_tier_rejects_a_genuine_capture(tier):
    """Stricter tiers may ask for more; they must not call an honest user a fraud."""
    pipe = FaceGuardPipeline.for_tier(tier)
    verdict = pipe.analyze(synth_live_clip(seed=0, frames=90)).verdict
    assert verdict is not FraudVerdict.FRAUD


def test_critical_tier_always_demands_a_challenge():
    genuine = PipelineResult(0.99, 0.99, FraudVerdict.GENUINE)
    assert policy_for(RiskTier.CRITICAL).requires_challenge(genuine)
    assert not policy_for(RiskTier.STANDARD).requires_challenge(genuine)
    assert not policy_for(RiskTier.CONVENIENCE).requires_challenge(
        PipelineResult(0.5, 0.5, FraudVerdict.SUSPICIOUS)
    )


def test_a_fraud_verdict_is_never_escalated_only_confirmed():
    fraud = PipelineResult(0.01, 0.9, FraudVerdict.FRAUD)
    assert not any(p.requires_challenge(fraud) for p in tiers())
