"""Properties of the Spectro-Temporal Liveness Fusion."""

from faceguard.config import FaceGuardConfig
from faceguard.fusion import fuse
from faceguard.types import DetectorResult, FraudVerdict


def _r(name, score, rel):
    return DetectorResult(name, score, rel)


def test_zero_reliability_detector_is_ignored():
    """A detector with reliability 0 must not move the posterior at all."""
    base = [_r("spectral", 0.9, 0.8)]
    with_abstainer = [*base, _r("rppg", 0.01, 0.0)]  # screaming "spoof" but abstaining
    s1, _, _, _ = fuse(base)
    s2, _, _, _ = fuse(with_abstainer)
    assert abs(s1 - s2) < 1e-9


def test_reliable_evidence_moves_posterior_more():
    low = fuse([_r("spectral", 0.05, 0.2)])[0]
    high = fuse([_r("spectral", 0.05, 1.0)])[0]
    assert high < low < 0.5  # more reliable spoof evidence -> lower liveness


def test_unanimous_live_is_genuine():
    dets = [_r("spectral", 0.9, 1.0), _r("texture", 0.85, 1.0), _r("rppg", 0.9, 0.8)]
    score, _, verdict, _ = fuse(dets)
    assert verdict == FraudVerdict.GENUINE and score > 0.6


def test_unanimous_spoof_is_fraud():
    dets = [_r("spectral", 0.1, 1.0), _r("rppg", 0.05, 0.9), _r("motion", 0.2, 1.0)]
    _, _, verdict, _ = fuse(dets)
    assert verdict == FraudVerdict.FRAUD


def test_low_confidence_genuine_is_downgraded():
    """A high score backed by almost no reliable evidence must not be trusted."""
    cfg = FaceGuardConfig(min_reliability_for_genuine=0.5)
    dets = [_r("spectral", 0.999, 0.08)]  # very high score, tiny reliability
    score, conf, verdict, _ = fuse(dets, cfg)
    # Score clears the genuine bar, but low confidence forces a downgrade.
    assert score > cfg.genuine_threshold
    assert conf < cfg.min_reliability_for_genuine
    assert verdict == FraudVerdict.SUSPICIOUS


def test_conflicting_evidence_is_suspicious():
    dets = [_r("spectral", 0.9, 0.5), _r("rppg", 0.1, 0.5)]
    _, _, verdict, _ = fuse(dets)
    assert verdict == FraudVerdict.SUSPICIOUS
