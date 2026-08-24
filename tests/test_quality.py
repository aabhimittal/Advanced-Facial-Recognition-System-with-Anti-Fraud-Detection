"""Capture quality gating: bad capture must mean "retry", never "fraud"."""

import numpy as np
import pytest

from faceguard import CaptureQualityGate, FaceGuardPipeline, FraudVerdict
from faceguard.utils import synth_live_clip
from faceguard.utils.image_ops import blur


@pytest.fixture(scope="module")
def gate():
    return CaptureQualityGate()


def test_good_capture_is_usable(gate):
    report = gate(synth_live_clip(seed=0))
    assert report.usable and report.score > 0.3 and not report.failures


@pytest.mark.parametrize(
    "name,make",
    [
        ("exposure", lambda: synth_live_clip(seed=1) * 0.04),          # near-dark room
        ("exposure", lambda: np.clip(synth_live_clip(seed=1) * 3.0, 0, 1)),  # blown out
        ("focus", lambda: np.stack([blur(f, 2.5) for f in synth_live_clip(seed=1)])),
        ("resolution", lambda: synth_live_clip(seed=1, size=24)),      # face too far away
        ("stability", lambda: np.repeat(synth_live_clip(seed=1)[:1], 30, axis=0)),  # frozen feed
    ],
)
def test_degraded_capture_is_rejected_with_the_right_reason(gate, name, make):
    report = gate(make())
    assert not report.usable
    assert name in report.failures
    assert report.remediation()  # operators get told what to fix


def test_unusable_capture_never_produces_a_fraud_accusation():
    # The whole point of the gate: a camera fault is not the user's fault.
    pipe = FaceGuardPipeline()
    result = pipe.analyze(synth_live_clip(seed=2) * 0.03)
    assert result.verdict is FraudVerdict.INDETERMINATE
    assert result.needs_retry
    assert result.liveness_score == 0.5 and result.reliability == 0.0


def test_mono_and_single_frame_captures_are_still_usable(gate):
    """An IR camera and a single enrolment photo are configurations, not faults."""
    mono = np.repeat(synth_live_clip(seed=3).mean(axis=-1, keepdims=True), 3, axis=-1)
    assert gate(mono).usable
    assert gate(synth_live_clip(seed=3)[0]).usable


def test_requiring_colour_and_clip_turns_advisories_into_failures():
    strict = CaptureQualityGate(require_colour=True, require_clip=True)
    mono = np.repeat(synth_live_clip(seed=4).mean(axis=-1, keepdims=True), 3, axis=-1)
    assert not strict(mono).usable
    assert not strict(synth_live_clip(seed=4)[0]).usable


def test_score_is_the_weakest_factor_not_the_average(gate):
    # A pristine capture of a 20-pixel face must not average its way to "fine".
    report = gate(synth_live_clip(seed=5, size=20))
    assert report.score == min(report.factors.values())
    assert not report.usable
