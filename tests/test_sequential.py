"""Sequential (SPRT) verification: decide as soon as the evidence allows."""

import pytest

from faceguard import FaceGuardPipeline, FraudVerdict
from faceguard.sequential import (
    SequentialDecision,
    SequentialState,
    SequentialVerifier,
    sequential_verdict,
    windows,
)
from faceguard.types import PipelineResult
from faceguard.utils import synth_live_clip, synth_spoof_clip


def _result(score, reliability=1.0, verdict=FraudVerdict.SUSPICIOUS) -> PipelineResult:
    return PipelineResult(score, reliability, verdict)


def test_obvious_attack_is_rejected_in_far_fewer_windows_than_the_budget():
    verifier = SequentialVerifier(far=1e-3, frr=0.02, max_windows=12)
    decision = verifier.run(_result(0.01) for _ in range(12))
    assert decision.state is SequentialState.REJECT
    assert decision.windows <= 2, "a blatant spoof should not cost twelve windows"


def test_clear_live_capture_is_accepted_early():
    decision = sequential_verdict((_result(0.99) for _ in range(12)))
    assert decision.state is SequentialState.ACCEPT
    assert decision.verdict is FraudVerdict.GENUINE
    assert decision.probability > 0.99


def test_ambiguous_evidence_keeps_sampling_then_exhausts():
    decision = sequential_verdict((_result(0.55) for _ in range(4)), max_windows=20)
    assert decision.state is SequentialState.EXHAUSTED
    assert decision.verdict is FraudVerdict.SUSPICIOUS, "undecided is not an accusation"


def test_low_reliability_windows_slow_the_decision_instead_of_corrupting_it():
    strong = SequentialVerifier().run(_result(0.95, reliability=1.0) for _ in range(6))
    weak = SequentialVerifier().run(_result(0.95, reliability=0.1) for _ in range(6))
    assert strong.state is SequentialState.ACCEPT
    assert weak.log_likelihood_ratio < strong.log_likelihood_ratio
    assert weak.state is not SequentialState.REJECT


def test_indeterminate_windows_contribute_nothing_but_still_consume_budget():
    verifier = SequentialVerifier(max_windows=3)
    decision = verifier.run(_result(0.5, 0.0, FraudVerdict.INDETERMINATE) for _ in range(3))
    assert decision.log_likelihood_ratio == 0.0
    assert decision.state is SequentialState.EXHAUSTED, "a dead camera must not loop forever"


def test_boundaries_follow_the_requested_error_targets():
    strict = SequentialVerifier(far=1e-6, frr=0.01)
    lax = SequentialVerifier(far=1e-2, frr=0.01)
    assert strict.upper > lax.upper, "a stricter FAR target must demand more evidence"


@pytest.mark.parametrize("bad", [{"far": 0.0}, {"frr": 1.0}, {"max_windows": 0}])
def test_invalid_operating_points_are_rejected(bad):
    with pytest.raises(ValueError):
        SequentialVerifier(**bad)


def test_expected_windows_requires_measured_steps():
    verifier = SequentialVerifier()
    with pytest.raises(ValueError):
        verifier.expected_windows(mean_step_live=-1.0, mean_step_attack=-1.0)
    live, attack = verifier.expected_windows(mean_step_live=4.0, mean_step_attack=-4.0)
    assert 0 < live <= verifier.max_windows and 0 < attack <= verifier.max_windows


def test_windows_overlap_so_a_transient_event_cannot_fall_between_them():
    clip = synth_live_clip(seed=0, frames=90)
    chunks = windows(clip, size=30)
    assert len(chunks) > 3
    assert all(len(c) == 30 for c in chunks)
    # Consecutive windows share frames.
    assert chunks[0].shape == chunks[1].shape


def test_streaming_a_real_capture_reaches_the_right_boundary():
    pipe = FaceGuardPipeline()
    for generator, expected in [
        (synth_live_clip, SequentialState.ACCEPT),
        (synth_spoof_clip, SequentialState.REJECT),
    ]:
        clip = generator(seed=0, frames=150)
        decision = SequentialVerifier().run(pipe.analyze(w) for w in windows(clip, size=45))
        assert decision.state is expected, decision.summary()
        assert isinstance(decision, SequentialDecision)
