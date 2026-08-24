"""Tamper-evident audit trail for biometric decisions.

Why a plain log is not enough
-----------------------------
When a biometric decision is disputed — a customer denies making the transfer,
a regulator asks why an account was locked, an investigator asks whether the
system was tricked — the log *is* the evidence. And a log that anyone with write
access can quietly edit is not evidence of anything.

So each record commits to its predecessor: ``hash_i = H(hash_{i-1} || record_i)``.
Altering, deleting or reordering any past entry breaks every hash that follows,
and :meth:`AuditLog.verify` finds the exact index where the chain diverges. It is
the same primitive as a blockchain without the distributed consensus — which is
the part you do not need when you control the store and only want detection.

Privacy first
-------------
An audit trail is *not* a place to keep faces. Records hold decisions,
per-detector scores, and a salted digest of the identity — never an image, never
an embedding, never a raw name. That makes the trail safe to retain for the
years a compliance regime demands, while the biometric data itself is deleted on
whatever much shorter schedule policy requires. Verifying a specific subject's
history is still possible: recompute the digest with the same salt and filter.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .types import FraudVerdict, PipelineResult

_GENESIS = "0" * 64


@dataclass
class AuditRecord:
    """One immutable decision entry."""

    timestamp: float
    verdict: str
    liveness_score: float
    reliability: float
    subject_digest: Optional[str]
    detector_scores: Dict[str, float] = field(default_factory=dict)
    quality: Optional[float] = None
    degraded: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    challenge_passed: Optional[bool] = None
    context: Dict[str, Any] = field(default_factory=dict)
    prev_hash: str = _GENESIS
    entry_hash: str = ""

    def payload(self) -> str:
        """Canonical JSON of everything except the entry's own hash."""
        body = {k: v for k, v in asdict(self).items() if k != "entry_hash"}
        return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)

    def compute_hash(self) -> str:
        return hashlib.sha256(self.payload().encode("utf-8")).hexdigest()


class AuditLog:
    """An append-only, hash-chained decision log.

    Parameters
    ----------
    salt:
        Secret used to derive subject digests. Keep it out of the log store: with
        the salt an auditor can confirm "was this decision about this person?",
        without it the trail cannot be mined for who was seen where.
    """

    def __init__(self, salt: bytes = b"", clock=time.time):
        self.salt = bytes(salt)
        self._clock = clock
        self._records: List[AuditRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> List[AuditRecord]:
        return list(self._records)

    @property
    def head(self) -> str:
        """Current chain head — publish or countersign it to pin the log's state.

        Pinning the head somewhere the log's own operator cannot rewrite (a
        witness service, a signed nightly export) is what upgrades this from
        "detects outside tampering" to "detects tampering by the operator too".
        """
        return self._records[-1].entry_hash if self._records else _GENESIS

    def append(
        self,
        result: PipelineResult,
        subject: Optional[str] = None,
        **context: Any,
    ) -> AuditRecord:
        """Record one decision and extend the chain."""
        record = AuditRecord(
            timestamp=float(self._clock()),
            verdict=result.verdict.value,
            liveness_score=round(float(result.liveness_score), 6),
            reliability=round(float(result.reliability), 6),
            subject_digest=self.digest(subject) if subject else None,
            detector_scores={
                name: round(float(d.score), 6) for name, d in sorted(result.detectors.items())
            },
            quality=round(float(result.quality.score), 6) if result.quality else None,
            degraded=list(result.degraded),
            elapsed_ms=round(float(result.elapsed_ms), 3),
            challenge_passed=result.challenge_passed,
            context=dict(context),
            prev_hash=self.head,
        )
        record.entry_hash = record.compute_hash()
        self._records.append(record)
        return record

    def digest(self, subject: str) -> str:
        """Salted, non-reversible identifier for a subject."""
        return hmac.new(self.salt, subject.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify(self) -> Optional[int]:
        """Return the index of the first corrupted record, or ``None`` if intact."""
        prev = _GENESIS
        for i, rec in enumerate(self._records):
            if rec.prev_hash != prev or rec.entry_hash != rec.compute_hash():
                return i
            prev = rec.entry_hash
        return None

    # -- reporting ----------------------------------------------------------
    def to_jsonl(self) -> str:
        """Export as JSON Lines — one record per line, ready to ship to a SIEM."""
        return "\n".join(json.dumps(asdict(r), sort_keys=True, default=str) for r in self._records)

    def rates(self) -> Dict[str, float]:
        """Verdict mix over the log.

        The operational dashboard nobody builds until an incident: a rising
        INDETERMINATE rate means a camera is failing, and a rising FRAUD rate on
        one lane usually means an attack campaign rather than a run of bad luck.
        """
        if not self._records:
            return {v.value: 0.0 for v in FraudVerdict}
        n = float(len(self._records))
        return {
            v.value: sum(1 for r in self._records if r.verdict == v.value) / n
            for v in FraudVerdict
        }

    def for_subject(self, subject: str) -> List[AuditRecord]:
        """Every decision recorded about one subject (needs the salt)."""
        target = self.digest(subject)
        return [r for r in self._records if r.subject_digest == target]
