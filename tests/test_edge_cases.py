"""Industrial edge cases: malformed input, broken detectors, degraded hardware.

Every case here is one a lab benchmark never sees and a deployment sees weekly.
The contract under test is uniform: FaceGuard always returns a decision object,
never raises for a bad *sample*, and never converts a system failure into an
accusation against the user.
"""

import numpy as np
import pytest

from faceguard import FaceGuardConfig, FaceGuardPipeline, FraudVerdict
from faceguard.liveness.base import InvalidFrameError, as_clip, to_grayscale
from faceguard.types import DetectorResult
from faceguard.utils import synth_live_clip, synth_spoof_clip


@pytest.fixture(scope="module")
def pipe():
    return FaceGuardPipeline()


# -- input normalisation ---------------------------------------------------
@pytest.mark.parametrize(
    "make,expected_scale",
    [
        (lambda: (synth_live_clip(seed=0) * 255).astype(np.uint8), "8-bit"),
        (lambda: (synth_live_clip(seed=0) * 1023).astype(np.uint16), "10-bit"),
        (lambda: (synth_live_clip(seed=0) * 4095).astype(np.uint16), "12-bit"),
        (lambda: (synth_live_clip(seed=0) * 65535).astype(np.uint16), "16-bit"),
    ],
)
def test_high_bit_depth_cameras_are_scaled_not_saturated(make, expected_scale):
    """A 12-bit machine-vision camera must not be turned into a white rectangle."""
    clip = as_clip(make())
    assert clip.max() <= 1.0
    assert 0.2 < clip.mean() < 0.9, f"{expected_scale} input collapsed to {clip.mean():.3f}"


def test_nan_and_inf_frames_are_scrubbed():
    raw = synth_live_clip(seed=1).copy()
    raw[3, :10, :10] = np.nan
    raw[4, :10, :10] = np.inf
    assert np.isfinite(as_clip(raw)).all()


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((0, 32, 32, 3)),         # empty clip from a dropped buffer
        np.zeros((32, 0, 3)),             # zero-width ROI from a bad crop
        np.array([1.0, 2.0, 3.0]),        # not an image at all
        np.array([np.zeros((4, 4)), np.zeros((5, 5))], dtype=object),  # ragged frames
    ],
)
def test_degenerate_input_raises_a_typed_error(bad):
    with pytest.raises(InvalidFrameError):
        as_clip(bad)


def test_pipeline_converts_malformed_input_into_a_retry_not_a_crash(pipe):
    result = pipe.analyze(np.array([1.0, 2.0, 3.0]))
    assert result.verdict is FraudVerdict.INDETERMINATE
    assert result.degraded  # the reason is recorded


@pytest.mark.parametrize(
    "shape",
    [(64, 64), (64, 64, 1), (64, 64, 3), (64, 64, 4), (10, 64, 64), (1, 10, 64, 64, 3)],
)
def test_every_plausible_layout_normalises_to_a_clip(shape):
    clip = as_clip(np.full(shape, 0.5))
    assert clip.ndim == 4 and clip.shape[-1] >= 1


def test_grayscale_conversion_handles_alpha_and_single_channel():
    rgba = np.dstack([np.full((16, 16), 0.5)] * 4)
    assert to_grayscale(rgba).shape == (16, 16)
    assert to_grayscale(np.full((16, 16, 1), 0.25)).shape == (16, 16)


# -- fault isolation -------------------------------------------------------
class _BrokenDetector:
    name = "spectral"

    def __call__(self, _):
        raise RuntimeError("model file corrupted on disk")


def test_a_crashing_detector_degrades_confidence_not_correctness(pipe):
    """One broken leg must not take down authentication for the whole estate."""
    healthy = pipe.analyze(synth_live_clip(seed=2))

    broken = FaceGuardPipeline()
    broken.spectral = _BrokenDetector()
    degraded = broken.analyze(synth_live_clip(seed=2))

    assert "spectral" in degraded.degraded
    assert degraded.detectors["spectral"].reliability == 0.0
    assert degraded.verdict is FraudVerdict.GENUINE          # still decides
    assert degraded.reliability <= healthy.reliability       # but less sure


def test_mass_detector_outage_produces_indeterminate_not_a_coin_flip():
    pipe = FaceGuardPipeline()
    for name in ("spectral", "texture", "dct", "rppg", "motion", "parallax"):
        setattr(pipe, name, _named_broken(name))
    result = pipe.analyze(synth_live_clip(seed=3))
    assert result.verdict is FraudVerdict.INDETERMINATE
    assert len(result.degraded) >= 6


def _named_broken(name):
    broken = _BrokenDetector()
    broken.name = name
    return broken


def test_latency_budget_makes_late_detectors_abstain():
    """A turnstile queue would rather have a fast, honest answer than a late one."""
    pipe = FaceGuardPipeline(config=FaceGuardConfig(time_budget_ms=1e-6))
    result = pipe.analyze(synth_live_clip(seed=4))
    assert result.degraded  # something was skipped
    skipped = [n for n in result.degraded if n in result.detectors]
    assert all(result.detectors[n].reliability == 0.0 for n in skipped)


# -- fusion robustness -----------------------------------------------------
def test_no_single_detector_can_veto_the_ensemble():
    """A saturated cue may argue forcefully; it may not dictate."""
    from faceguard.fusion import fuse

    honest = [DetectorResult(n, 0.1, 1.0) for n in ("spectral", "texture", "rppg", "motion")]
    saturated = DetectorResult("dct", 1.0, 1.0)  # claims certainty
    score, _, verdict, contributions = fuse(honest + [saturated])
    assert abs(contributions["dct"]) <= FaceGuardConfig().max_detector_logit + 1e-9
    assert verdict is FraudVerdict.FRAUD, "four honest detectors were overruled by one"


def test_abstentions_are_mathematically_inert():
    from faceguard.fusion import fuse

    base = [DetectorResult("spectral", 0.9, 0.8)]
    with_abstainers = base + [DetectorResult.abstain(n, "detector_error") for n in ("rppg", "dct")]
    assert abs(fuse(base)[0] - fuse(with_abstainers)[0]) < 1e-12


# -- capture edge cases ----------------------------------------------------
def test_constant_and_extreme_frames_never_crash(pipe):
    for frames in (np.zeros((20, 64, 64, 3)), np.ones((20, 64, 64, 3)), np.full((20, 64, 64, 3), 0.5)):
        result = pipe.analyze(frames)
        assert result.verdict in set(FraudVerdict)
        assert 0.0 <= result.liveness_score <= 1.0


def test_batch_analysis_isolates_a_bad_sample(pipe):
    results = pipe.analyze_batch([synth_live_clip(seed=5), np.array([1.0]), synth_spoof_clip(seed=5)])
    assert results[0].verdict is FraudVerdict.GENUINE
    assert results[1].verdict is FraudVerdict.INDETERMINATE
    assert results[2].verdict is FraudVerdict.FRAUD
