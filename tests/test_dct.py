"""DCT deepfake head: separates real photos from GAN-fingerprinted ones."""

import numpy as np
import pytest

from faceguard import FaceGuardPipeline, FraudVerdict
from faceguard.liveness import DCTDeepfakeDetector
from faceguard.utils import synth_deepfake_clip, synth_deepfake_face, synth_face


@pytest.mark.parametrize("seed", [0, 1, 5])
def test_dct_flags_deepfake_fingerprint(seed):
    det = DCTDeepfakeDetector()
    real = det(synth_face(seed=seed))
    fake = det(synth_deepfake_face(seed=seed))
    assert real.score > 0.5 > fake.score
    assert real.score > fake.score
    assert fake.detail["corner"] > real.detail["corner"]  # Nyquist checkerboard


def test_dct_abstains_on_small_crop():
    assert DCTDeepfakeDetector()(np.zeros((16, 16))).reliability == 0.0


def test_pipeline_flags_deepfake_clip():
    # A deepfake that looks live (texture + motion) but carries the GAN fingerprint
    # and no pulse must NOT be accepted as genuine — the DCT head is decisive here.
    pipe = FaceGuardPipeline()
    result = pipe.analyze(synth_deepfake_clip(seed=0))
    assert result.verdict != FraudVerdict.GENUINE
    assert result.detectors["dct"].score < 0.5
