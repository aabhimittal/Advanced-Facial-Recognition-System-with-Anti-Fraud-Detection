"""End-to-end FaceGuard pipeline: detect -> liveness -> fuse -> recognise.

    pipeline = FaceGuardPipeline()
    pipeline.enroll("alice", alice_crop)
    result = pipeline.identify_and_verify(clip)   # clip: (T, H, W, 3)
    if result.is_trustworthy_match():
        grant_access(result.identity)

A "trustworthy match" requires BOTH a recognised identity AND a GENUINE liveness
verdict — recognition alone can be fooled by a photo of an authorised user; the
Spectro-Temporal Liveness Fusion is what closes that hole.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .challenge import Challenge, ChallengeResult, ChallengeType, ChallengeVerifier, issue_challenge
from .config import FaceGuardConfig
from .detection import FaceDetector
from .fusion import SpectroTemporalLivenessFusion
from .liveness import (
    DCTDeepfakeDetector,
    MotionDetector,
    RPPGDetector,
    SpectralDetector,
    TextureDetector,
)
from .liveness.base import as_clip
from .recognition import FaceEmbedder, FaceMatcher
from .types import FraudVerdict, PipelineResult


class FaceGuardPipeline:
    def __init__(
        self,
        config: Optional[FaceGuardConfig] = None,
        embedder: Optional[FaceEmbedder] = None,
        matcher: Optional[FaceMatcher] = None,
        detector: Optional[FaceDetector] = None,
        fps: float = 30.0,
    ):
        self.config = config or FaceGuardConfig()
        self.detector = detector or FaceDetector()
        self.embedder = embedder or FaceEmbedder()
        self.matcher = matcher or FaceMatcher(self.config.match_threshold)
        self.fusion = SpectroTemporalLivenessFusion(self.config)

        self.spectral = SpectralDetector()
        self.texture = TextureDetector()
        self.rppg = RPPGDetector(fps=fps)
        self.motion = MotionDetector()
        self.dct = DCTDeepfakeDetector()
        self.challenge_verifier = ChallengeVerifier()

    # -- enrolment / recognition -------------------------------------------
    def enroll(self, name: str, face: np.ndarray) -> None:
        crop = self._largest_crop(face)
        self.matcher.enroll(name, self.embedder.embed(crop))

    # -- liveness ----------------------------------------------------------
    def analyze(self, frames: np.ndarray) -> PipelineResult:
        """Run liveness fusion on an image or a clip (no recognition)."""
        clip = as_clip(frames)
        rep = self._representative(clip)  # single frame for image-based cues

        results = {
            self.spectral.name: self.spectral(rep),
            self.texture.name: self.texture(rep),
            self.rppg.name: self.rppg(clip),
            self.motion.name: self.motion(clip),
            self.dct.name: self.dct(rep),
        }
        score, confidence, verdict, _ = self.fusion.fuse(results.values())
        return PipelineResult(
            liveness_score=score,
            reliability=confidence,
            verdict=verdict,
            detectors=results,
        )

    def identify_and_verify(self, frames: np.ndarray) -> PipelineResult:
        """Full pipeline: liveness fusion + identity match on the same input."""
        result = self.analyze(frames)
        rep = self._representative(as_clip(frames))
        crop = self._largest_crop((rep * 255).astype("uint8"))
        name, dist = self.matcher.identify(self.embedder.embed(crop))
        result.identity = name
        result.match_distance = dist
        return result

    # -- active challenge-response (for SUSPICIOUS cases) ------------------
    def issue_challenge(
        self, kind: Optional[ChallengeType] = None, nonce: int = 0
    ) -> Challenge:
        """Issue an (ideally random-nonce) challenge to escalate a SUSPICIOUS case."""
        return issue_challenge(kind=kind, nonce=nonce)

    def verify_challenge(self, response_frames: np.ndarray, challenge: Challenge) -> ChallengeResult:
        """Check that a captured clip performs the requested action."""
        return self.challenge_verifier.verify(response_frames, challenge)

    def resolve_suspicious(
        self,
        result: PipelineResult,
        response_frames: np.ndarray,
        challenge: Challenge,
    ) -> PipelineResult:
        """Escalate a SUSPICIOUS result via challenge-response.

        A passed challenge upgrades the verdict to GENUINE; a failed one is treated
        as an attack and downgraded to FRAUD. GENUINE/FRAUD results are returned
        unchanged — the challenge only adjudicates the ambiguous middle band.
        """
        if result.verdict != FraudVerdict.SUSPICIOUS:
            return result
        outcome = self.verify_challenge(response_frames, challenge)
        result.challenge_passed = outcome.passed
        result.verdict = FraudVerdict.GENUINE if outcome.passed else FraudVerdict.FRAUD
        return result

    # -- helpers -----------------------------------------------------------
    def _representative(self, clip: np.ndarray) -> np.ndarray:
        # Median over time is robust to per-frame jitter and pulse modulation.
        return np.median(clip[..., :3], axis=0)

    def _largest_crop(self, image: np.ndarray) -> np.ndarray:
        box = self.detector.detect_largest(np.asarray(image))
        crop = box.crop(np.asarray(image))
        return crop if crop.size else np.asarray(image)
