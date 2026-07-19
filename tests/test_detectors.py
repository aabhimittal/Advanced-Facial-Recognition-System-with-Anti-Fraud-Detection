"""Each liveness cue must, on its own, score a live sample above a spoof."""

import numpy as np
import pytest

from faceguard.liveness import MotionDetector, RPPGDetector, SpectralDetector, TextureDetector
from faceguard.utils import synth_live_clip, synth_spoof_clip
from faceguard.utils.synthetic import synth_face


def _rep(clip):
    return np.median(clip[..., :3], axis=0)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_spectral_separates(seed):
    det = SpectralDetector()
    live = det(synth_face(seed=seed, spoof=False))
    spoof = det(synth_face(seed=seed, spoof=True))
    assert live.score > spoof.score
    assert live.score > 0.5 > spoof.score


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_texture_separates(seed):
    det = TextureDetector()
    live = det(synth_face(seed=seed, spoof=False))
    spoof = det(synth_face(seed=seed, spoof=True))
    assert live.score > spoof.score


@pytest.mark.parametrize("seed", [0, 3])
def test_rppg_separates(seed):
    det = RPPGDetector(fps=30.0)
    live = det(synth_live_clip(seed=seed))
    spoof = det(synth_spoof_clip(seed=seed))
    assert live.score > 0.5 > spoof.score
    # A real clip yields a plausible resting/active heart rate.
    assert 42 <= live.detail["hr_bpm"] <= 240


@pytest.mark.parametrize("seed", [0, 3])
def test_motion_separates(seed):
    det = MotionDetector()
    live = det(synth_live_clip(seed=seed))
    spoof = det(synth_spoof_clip(seed=seed))
    assert live.score > spoof.score


def test_detectors_abstain_on_degraded_input():
    # Single frame -> temporal detectors must report zero reliability (abstain).
    frame = synth_face(size=96)
    assert RPPGDetector()(frame).reliability == 0.0
    assert MotionDetector()(frame).reliability == 0.0
    # Tiny crop -> spectral abstains rather than guessing.
    assert SpectralDetector()(np.zeros((10, 10))).reliability == 0.0


def test_scores_are_bounded():
    det = SpectralDetector()
    for arr in (np.zeros((64, 64)), np.ones((64, 64)), np.full((64, 64), np.nan)):
        r = det(arr)
        assert 0.0 <= r.score <= 1.0 and 0.0 <= r.reliability <= 1.0
