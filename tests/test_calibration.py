"""Learning STLF fusion weights from a labelled PAD set."""

import math

import pytest

from faceguard import FaceGuardPipeline, FraudVerdict
from faceguard.config import FaceGuardConfig
from faceguard.fusion import fit_fusion_weights, fuse
from faceguard.utils import synth_deepfake_clip, synth_live_clip, synth_spoof_clip

_GENS = [(synth_live_clip, 1), (synth_spoof_clip, 0), (synth_deepfake_clip, 0)]


def _dataset(pipe, seeds):
    samples, labels = [], []
    for seed in seeds:
        for gen, label in _GENS:
            samples.append(pipe.analyze(gen(seed=seed)).detectors)
            labels.append(label)
    return samples, labels


def _accuracy(pipe, config, seeds):
    ok = total = 0
    for seed in seeds:
        for gen, label in _GENS:
            dets = pipe.analyze(gen(seed=seed)).detectors
            _, _, verdict, _ = fuse(dets.values(), config)
            pred = 1 if verdict == FraudVerdict.GENUINE else 0
            ok += pred == label
            total += 1
    return ok / total


@pytest.fixture(scope="module")
def pipe():
    return FaceGuardPipeline()


def test_learned_weights_are_valid(pipe):
    samples, labels = _dataset(pipe, range(8))
    cfg = fit_fusion_weights(samples, labels)
    assert set(cfg.detector_weights) == set(FaceGuardPipeline().detectors)
    for w in cfg.detector_weights.values():
        assert w >= 0.0 and math.isfinite(w)   # non-negativity is enforced
    assert 0.0 < cfg.genuine_prior < 1.0


def test_learning_improves_heldout_accuracy(pipe):
    train, labels = _dataset(pipe, range(8))
    learned = fit_fusion_weights(train, labels)
    heldout = range(100, 108)
    assert _accuracy(pipe, learned, heldout) >= _accuracy(pipe, FaceGuardConfig(), heldout)
    assert _accuracy(pipe, learned, heldout) >= 0.9


def test_fit_rejects_mismatched_lengths(pipe):
    samples, labels = _dataset(pipe, range(2))
    with pytest.raises(ValueError):
        fit_fusion_weights(samples, labels[:-1])
