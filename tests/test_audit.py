"""Tamper-evident decision logging."""

import json

import pytest

from faceguard import AuditLog, FaceGuardPipeline, FraudVerdict
from faceguard.types import PipelineResult
from faceguard.utils import synth_live_clip, synth_spoof_clip


@pytest.fixture()
def log():
    counter = iter(range(1000))
    return AuditLog(salt=b"audit-salt", clock=lambda: 1_700_000_000 + next(counter))


def _result(verdict=FraudVerdict.GENUINE, score=0.9):
    return PipelineResult(score, 0.8, verdict)


def test_an_intact_chain_verifies(log):
    for _ in range(5):
        log.append(_result())
    assert log.verify() is None and len(log) == 5


def test_editing_a_past_record_is_detected_at_its_index(log):
    for _ in range(5):
        log.append(_result())
    log.records  # a copy: mutating it must not affect the log
    log._records[2].liveness_score = 0.01
    assert log.verify() == 2


def test_deleting_a_record_breaks_the_chain(log):
    for _ in range(4):
        log.append(_result())
    del log._records[1]
    assert log.verify() == 1


def test_reordering_records_is_detected(log):
    for i in range(4):
        log.append(_result(score=0.5 + i / 10))
    log._records[1], log._records[2] = log._records[2], log._records[1]
    assert log.verify() is not None


def test_the_log_stores_no_biometric_data(log):
    pipe = FaceGuardPipeline()
    log.append(pipe.analyze(synth_live_clip(seed=0)), subject="alice", lane="gate-1")
    record = log.records[0]
    body = json.loads(json.dumps(record.payload()))
    assert "alice" not in body
    assert record.subject_digest and record.subject_digest != "alice"
    assert record.detector_scores and record.context == {"lane": "gate-1"}


def test_a_subject_history_is_queryable_with_the_salt(log):
    log.append(_result(), subject="alice")
    log.append(_result(), subject="bob")
    log.append(_result(FraudVerdict.FRAUD), subject="alice")
    assert len(log.for_subject("alice")) == 2
    assert len(log.for_subject("nobody")) == 0


def test_verdict_rates_expose_a_failing_camera(log):
    for _ in range(3):
        log.append(_result(FraudVerdict.INDETERMINATE, 0.5))
    log.append(_result())
    rates = log.rates()
    assert rates["indeterminate"] == pytest.approx(0.75)
    assert sum(rates.values()) == pytest.approx(1.0)


def test_head_advances_and_pins_the_state(log):
    genesis = log.head
    log.append(_result())
    assert log.head != genesis
    assert log.head == log.records[-1].entry_hash


def test_export_is_one_json_object_per_line(log):
    pipe = FaceGuardPipeline()
    log.append(pipe.analyze(synth_live_clip(seed=1)), subject="alice")
    log.append(pipe.analyze(synth_spoof_clip(seed=1)), subject="mallory")
    lines = log.to_jsonl().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["entry_hash"] for line in lines)
