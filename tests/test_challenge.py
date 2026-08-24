"""Active challenge-response for SUSPICIOUS cases."""

import pytest

from faceguard import FaceGuardPipeline, FraudVerdict
from faceguard.challenge import ChallengeType, ChallengeVerifier, issue_challenge
from faceguard.types import PipelineResult
from faceguard.utils import synth_challenge_clip

_KINDS = ["blink", "turn_left", "turn_right", "nod"]


@pytest.mark.parametrize("kind", _KINDS)
def test_matching_response_passes(kind):
    v = ChallengeVerifier()
    r = v.verify(synth_challenge_clip(kind, respond=True, seed=0), issue_challenge(ChallengeType(kind)))
    assert r.passed and r.confidence > 0.0


@pytest.mark.parametrize("kind", _KINDS)
def test_non_response_fails(kind):
    # A live clip that does NOT perform the action must fail the challenge.
    v = ChallengeVerifier()
    clip = synth_challenge_clip("blink", respond=False, seed=0)
    assert not v.verify(clip, issue_challenge(ChallengeType(kind))).passed


def test_no_false_accepts_across_mismatched_pairs():
    """The crux of replay resistance: a response to a *different* prompt is rejected."""
    v = ChallengeVerifier()
    for asked in _KINDS:
        for did in _KINDS:
            if asked == did:
                continue
            clip = synth_challenge_clip(did, respond=True, seed=0)
            assert not v.verify(clip, issue_challenge(ChallengeType(asked))).passed


def test_issue_challenge_is_deterministic_by_nonce():
    assert issue_challenge(nonce=0).kind == issue_challenge(nonce=0).kind
    assert issue_challenge(nonce=1).instruction  # has human-readable text


def test_pipeline_resolves_suspicious_case():
    pipe = FaceGuardPipeline()
    challenge = pipe.issue_challenge(ChallengeType.NOD, nonce=3)

    passed = PipelineResult(0.5, 0.9, FraudVerdict.SUSPICIOUS)
    passed = pipe.resolve_suspicious(passed, synth_challenge_clip("nod", seed=1), challenge)
    assert passed.verdict == FraudVerdict.GENUINE and passed.challenge_passed

    failed = PipelineResult(0.5, 0.9, FraudVerdict.SUSPICIOUS)
    failed = pipe.resolve_suspicious(failed, synth_challenge_clip("turn_left", seed=1), challenge)
    assert failed.verdict == FraudVerdict.FRAUD and failed.challenge_passed is False


def test_resolve_leaves_decided_verdicts_untouched():
    pipe = FaceGuardPipeline()
    ch = pipe.issue_challenge(ChallengeType.BLINK, nonce=1)
    genuine = PipelineResult(0.9, 0.9, FraudVerdict.GENUINE)
    out = pipe.resolve_suspicious(genuine, synth_challenge_clip("blink", seed=1), ch)
    assert out.verdict == FraudVerdict.GENUINE and out.challenge_passed is None


# -- freshness, signing and replay protection ------------------------------
def test_a_stale_challenge_is_refused_before_any_analysis():
    """A response captured minutes ago must not satisfy today's prompt."""
    from faceguard.challenge import Challenge, ChallengeType, ChallengeVerifier

    challenge = Challenge(ChallengeType.BLINK, nonce=7, issued_at=1000.0, ttl_s=30.0)
    outcome = ChallengeVerifier().verify(
        synth_challenge_clip("blink", seed=0), challenge, now=1100.0
    )
    assert not outcome.passed and outcome.reason == "expired"


def test_an_unsigned_challenge_is_refused_when_an_issuer_is_configured():
    """The client does not get to choose which prompt it answers."""
    from faceguard.challenge import Challenge, ChallengeIssuer, ChallengeType, ChallengeVerifier

    issuer = ChallengeIssuer(secret=b"server-side-secret-key-material!")
    verifier = ChallengeVerifier(issuer=issuer)
    forged = Challenge(ChallengeType.BLINK, nonce=1)
    outcome = verifier.verify(synth_challenge_clip("blink", seed=0), forged)
    assert not outcome.passed and outcome.reason == "bad_signature"


def test_a_response_cannot_be_submitted_twice():
    from faceguard.challenge import ChallengeIssuer, ChallengeType, ChallengeVerifier

    issuer = ChallengeIssuer(secret=b"server-side-secret-key-material!")
    verifier = ChallengeVerifier(issuer=issuer)
    challenge = issuer.issue(ChallengeType.BLINK)
    clip = synth_challenge_clip("blink", seed=0)

    assert verifier.verify(clip, challenge).passed
    replayed = verifier.verify(clip, challenge)
    assert not replayed.passed and replayed.reason == "replayed"


def test_issued_nonces_are_unpredictable():
    from faceguard.challenge import issue_challenge

    nonces = {issue_challenge().nonce for _ in range(64)}
    assert len(nonces) == 64, "a guessable nonce is a pre-recordable prompt"


def test_issuer_requires_real_key_material():
    from faceguard.challenge import ChallengeIssuer

    with pytest.raises(ValueError):
        ChallengeIssuer(secret=b"tooshort")


def test_a_genuine_user_who_misses_the_prompt_is_asked_again_not_accused():
    """Missing a challenge is a usability event; forging one is an attack."""
    from faceguard import FaceGuardPipeline, FraudVerdict
    from faceguard.challenge import ChallengeType
    from faceguard.types import PipelineResult

    pipe = FaceGuardPipeline()
    challenge = pipe.issue_challenge(ChallengeType.NOD, nonce=3)
    passive_genuine = PipelineResult(0.95, 0.9, FraudVerdict.GENUINE)
    out = pipe.enforce_challenge(
        passive_genuine, synth_challenge_clip("nod", respond=False, seed=2), challenge
    )
    assert out.verdict is FraudVerdict.SUSPICIOUS and out.challenge_passed is False
