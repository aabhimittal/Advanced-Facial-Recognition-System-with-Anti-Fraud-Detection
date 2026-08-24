"""Active challenge-response liveness for SUSPICIOUS cases.

Passive liveness (STLF) decides from whatever the camera happens to capture. When
the evidence is inconclusive (`SUSPICIOUS`), we escalate to an *active* check: the
system issues a randomly chosen challenge — blink, turn your head, nod — and
verifies that the response actually happens, in the requested way, within a short
window. Because the challenge is unpredictable, a pre-recorded replay cannot
satisfy it: this is what defeats the replay attacks passive cues find hardest.
"""

from .challenge import (
    Challenge,
    ChallengeIssuer,
    ChallengeType,
    ChallengeResult,
    ChallengeVerifier,
    issue_challenge,
)

__all__ = [
    "Challenge",
    "ChallengeIssuer",
    "ChallengeType",
    "ChallengeResult",
    "ChallengeVerifier",
    "issue_challenge",
]
