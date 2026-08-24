"""A deployment-shaped walkthrough: quality gate, risk tiers, SPRT, audit trail.

Run with:  python examples/production.py

`quickstart.py` shows the detection idea. This one shows the parts a real
integration has to get right — deciding when *not* to decide, choosing an
operating point per transaction, deciding as fast as the evidence allows, and
leaving behind a record that survives a dispute.
"""

import numpy as np

from faceguard import (
    AuditLog,
    CaptureQualityGate,
    ChallengeIssuer,
    FaceGuardPipeline,
    FraudVerdict,
    ProtectedMatcher,
    RiskTier,
    SequentialVerifier,
    TemplateProtector,
    policy_for,
)
from faceguard.recognition import FaceEmbedder
from faceguard.sequential import windows
from faceguard.utils import (
    synth_challenge_clip,
    synth_face,
    synth_injected_clip,
    synth_live_clip,
    synth_mask_clip,
    synth_spoof_clip,
)


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# --------------------------------------------------------------------------
section("1. Capture quality: a bad frame is a retry, not an accusation")
gate = CaptureQualityGate()
pipe = FaceGuardPipeline()

for label, clip in [
    ("well-lit capture", synth_live_clip(seed=0)),
    ("near-dark room", synth_live_clip(seed=0) * 0.04),
    ("frozen video feed", np.repeat(synth_live_clip(seed=0)[:1], 30, axis=0)),
]:
    report = gate(clip)
    result = pipe.analyze(clip)
    print(f"\n{label:20} {report.summary()}")
    print(f"{'':20} verdict={result.verdict.value}")
    for hint in report.remediation():
        print(f"{'':20} -> {hint}")

# --------------------------------------------------------------------------
section("2. Risk tiers: one ensemble, several operating points")
for tier in RiskTier:
    policy = policy_for(tier)
    tiered = FaceGuardPipeline.for_tier(tier)
    verdicts = {
        name: tiered.analyze(clip).verdict.value
        for name, clip in [
            ("live", synth_live_clip(seed=1, frames=90)),
            ("mask", synth_mask_clip(seed=1, frames=90)),
            ("print", synth_spoof_clip(seed=1, frames=90)),
        ]
    }
    print(f"\n{tier.value:12} {verdicts}")
    print(f"{'':12} {policy.rationale}")

# --------------------------------------------------------------------------
section("3. Sequential decisioning: stop as soon as the evidence allows")
for label, clip in [
    ("live", synth_live_clip(seed=2, frames=180)),
    ("injected stream", synth_injected_clip(seed=2, frames=180)),
]:
    verifier = SequentialVerifier(far=1e-3, frr=0.02, max_windows=12)
    decision = verifier.run(pipe.analyze(w) for w in windows(clip, size=45))
    print(f"{label:16} {decision.summary()}")

# --------------------------------------------------------------------------
section("4. Challenge-response: signed, expiring, single-use")
issuer = ChallengeIssuer(secret=b"keep-this-in-an-hsm-not-in-git!!")
guarded = FaceGuardPipeline(issuer=issuer)
challenge = guarded.issue_challenge()
print(f"prompt: {challenge.instruction}  (ttl={challenge.ttl_s}s)")

response = synth_challenge_clip(challenge.kind.value, respond=True, seed=3)
print(f"first submission : {guarded.verify_challenge(response, challenge).passed}")
replayed = guarded.verify_challenge(response, challenge)
print(f"same clip again  : passed={replayed.passed} reason={replayed.reason!r}")

# --------------------------------------------------------------------------
section("5. Cancellable templates: surviving a gallery breach")
embedder = FaceEmbedder()
protector = TemplateProtector(key=b"deployment-key-from-your-kms!!!!", bits=512)
gallery = ProtectedMatcher(protector, match_threshold=0.6)
gallery.enroll("alice", embedder.embed(synth_face(seed=7)))
print("identify:", gallery.identify(embedder.embed(synth_face(seed=7))))

stolen = protector.protect(embedder.embed(synth_face(seed=7)), b"old-salt")
reissued = protector.reissue(embedder.embed(synth_face(seed=7)), b"new-salt")
print(f"stolen vs reissued template distance: {protector.distance(stolen, reissued):.3f}")
print("(≈0.5 means the leaked template no longer matches anything)")

# --------------------------------------------------------------------------
section("6. Audit trail: tamper-evident, and free of biometric data")
log = AuditLog(salt=b"audit-salt-from-your-kms")
for subject, clip in [("alice", synth_live_clip(seed=4)), ("mallory", synth_spoof_clip(seed=4))]:
    log.append(pipe.analyze(clip), subject=subject, lane="turnstile-3")

print(f"records={len(log)}  chain intact={log.verify() is None}  head={log.head[:16]}...")
print(f"verdict mix: {log.rates()}")
print(f"stored fields: {sorted(vars(log.records[0]))}")

log._records[0].verdict = FraudVerdict.FRAUD.value  # someone edits history
print(f"after tampering, first corrupted index: {log.verify()}")
