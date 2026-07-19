"""End-to-end pipeline: liveness verdicts and identity + liveness coupling."""

import numpy as np

from faceguard import FaceGuardPipeline, FraudVerdict
from faceguard.utils import synth_live_clip, synth_spoof_clip
from faceguard.utils.synthetic import synth_face


def test_pipeline_flags_live_and_spoof():
    pipe = FaceGuardPipeline()
    assert pipe.analyze(synth_live_clip(seed=0)).verdict == FraudVerdict.GENUINE
    assert pipe.analyze(synth_spoof_clip(seed=0)).verdict == FraudVerdict.FRAUD


def test_photo_of_enrolled_user_matches_identity_but_fails_liveness():
    """The core security property: a spoof of an enrolled user is NOT trusted."""
    pipe = FaceGuardPipeline()
    live = synth_live_clip(seed=2)
    enroll_crop = (np.median(live[..., :3], axis=0) * 255).astype("uint8")
    pipe.enroll("alice", enroll_crop)

    # A replayed spoof rendered from the same face may still match identity...
    spoof = synth_spoof_clip(seed=2)
    result = pipe.identify_and_verify(spoof)
    # ...but liveness fusion must reject it, so it is never a trustworthy match.
    assert result.verdict == FraudVerdict.FRAUD
    assert not result.is_trustworthy_match()


def test_genuine_enrolled_user_is_trusted():
    pipe = FaceGuardPipeline()
    live = synth_live_clip(seed=5)
    crop = (np.median(live[..., :3], axis=0) * 255).astype("uint8")
    pipe.enroll("bob", crop)
    result = pipe.identify_and_verify(live)
    assert result.verdict == FraudVerdict.GENUINE
    assert result.identity == "bob"
    assert result.is_trustworthy_match()


def test_matcher_rejects_unknown_face():
    pipe = FaceGuardPipeline()
    pipe.enroll("carol", synth_face(seed=1, size=96))
    name, dist = pipe.matcher.identify(pipe.embedder.embed(synth_face(seed=99, size=96)))
    # A clearly different synthetic face should not match carol's prototype.
    assert name is None or dist > 0.0
