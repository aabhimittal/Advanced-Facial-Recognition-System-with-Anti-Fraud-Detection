"""The four new liveness legs, each on the attack class it exists to catch."""

import numpy as np
import pytest

from faceguard import FaceGuardPipeline, FraudVerdict
from faceguard.liveness import (
    DisplayBandingDetector,
    ParallaxDetector,
    SensorNoiseDetector,
    SubsurfaceScatteringDetector,
)
from faceguard.utils import (
    synth_injected_clip,
    synth_live_clip,
    synth_mask_clip,
    synth_replay_clip,
    synth_spoof_clip,
    synth_tilted_photo_clip,
)

SEEDS = [0, 1, 2]


# -- sensor / injection ----------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_sensor_noise_detects_injection(seed):
    """A stream that never passed through silicon has no photon shot noise."""
    det = SensorNoiseDetector()
    captured = det(synth_live_clip(seed=seed))
    injected = det(synth_injected_clip(seed=seed))
    assert captured.score > injected.score
    assert injected.score < 0.4 < captured.score


@pytest.mark.parametrize("seed", SEEDS)
def test_presentation_attacks_still_look_like_real_captures(seed):
    """A photo held up to a camera IS captured by that camera: don't double-count."""
    det = SensorNoiseDetector()
    assert det(synth_spoof_clip(seed=seed)).score > 0.5


def test_sensor_abstains_on_a_burst_too_short_to_measure_noise():
    result = SensorNoiseDetector()(synth_live_clip(seed=0, frames=4))
    assert result.reliability == 0.0 and result.score == 0.5


# -- banding / screen replay -----------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_banding_separates_screen_replay_from_print(seed):
    det = DisplayBandingDetector()
    replay = det(synth_replay_clip(seed=seed))
    printed = det(synth_spoof_clip(seed=seed))
    assert replay.score < 0.2
    assert replay.reliability > printed.reliability, "seeing the beat should be the confident case"


@pytest.mark.parametrize("seed", SEEDS)
def test_banding_does_not_accuse_a_live_face(seed):
    assert DisplayBandingDetector()(synth_live_clip(seed=seed)).score > 0.1


def test_banding_is_one_sided():
    """No banding is weak evidence: a global-shutter camera never shows any."""
    from faceguard.liveness.base import MAX_ONE_SIDED_LIVE

    assert DisplayBandingDetector()(synth_live_clip(seed=0)).score <= MAX_ONE_SIDED_LIVE


# -- parallax / planar attacks ---------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("attack", [synth_spoof_clip, synth_replay_clip])
def test_parallax_certifies_depth_and_flatness(seed, attack):
    det = ParallaxDetector()
    live = det(synth_live_clip(seed=seed))
    flat = det(attack(seed=seed))
    assert live.score > 0.5 > flat.score
    assert flat.detail["residual_ratio"] < live.detail["residual_ratio"]


def test_parallax_abstains_without_motion():
    """No movement, no baseline, no opinion."""
    still = np.repeat(synth_live_clip(seed=0)[:1], 20, axis=0)
    assert ParallaxDetector()(still).reliability == 0.0


def test_a_tilted_photo_is_still_a_plane():
    """Six degrees of freedom of hand motion do not create depth."""
    pipe = FaceGuardPipeline()
    result = pipe.analyze(synth_tilted_photo_clip(seed=0))
    assert result.verdict is FraudVerdict.FRAUD


# -- subsurface / 3-D masks ------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_subsurface_scattering_separates_skin_from_silicone(seed):
    det = SubsurfaceScatteringDetector()
    skin = det(synth_live_clip(seed=seed))
    mask = det(synth_mask_clip(seed=seed))
    assert skin.score > 0.5 > mask.score
    # Real skin blurs red more than green/blue; opaque materials do not.
    assert skin.detail["rg_detail_ratio"] < mask.detail["rg_detail_ratio"]


def test_subsurface_abstains_without_colour():
    mono = synth_live_clip(seed=0).mean(axis=-1)
    assert SubsurfaceScatteringDetector()(mono).reliability == 0.0


@pytest.mark.parametrize("seed", SEEDS)
def test_mask_attack_is_never_accepted_given_a_realistic_capture(seed):
    """A 3-D mask defeats geometry and sensor cues; pulse and material catch it.

    It needs a longer look than a print does — two seconds leaves it in the
    ambiguous band — which is exactly why the pipeline escalates rather than
    guessing.
    """
    pipe = FaceGuardPipeline()
    result = pipe.analyze(synth_mask_clip(seed=seed, frames=150))
    assert result.verdict is not FraudVerdict.GENUINE


# -- the ensemble as a whole -----------------------------------------------
@pytest.mark.parametrize(
    "generator,expected",
    [
        (synth_live_clip, FraudVerdict.GENUINE),
        (synth_spoof_clip, FraudVerdict.FRAUD),
        (synth_replay_clip, FraudVerdict.FRAUD),
        (synth_tilted_photo_clip, FraudVerdict.FRAUD),
        (synth_injected_clip, FraudVerdict.FRAUD),
    ],
)
@pytest.mark.parametrize("seed", SEEDS)
def test_end_to_end_attack_coverage(generator, expected, seed):
    result = FaceGuardPipeline().analyze(generator(seed=seed))
    assert result.verdict is expected, result.summary()
