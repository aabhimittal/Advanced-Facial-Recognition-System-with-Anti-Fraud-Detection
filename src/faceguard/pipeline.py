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

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .challenge import (
    Challenge,
    ChallengeIssuer,
    ChallengeResult,
    ChallengeType,
    ChallengeVerifier,
    issue_challenge,
)
from .config import FaceGuardConfig
from .detection import FaceDetector
from .fusion import SpectroTemporalLivenessFusion
from .liveness import (
    DCTDeepfakeDetector,
    DisplayBandingDetector,
    MotionDetector,
    ParallaxDetector,
    RPPGDetector,
    SensorNoiseDetector,
    SpectralDetector,
    SubsurfaceScatteringDetector,
    TextureDetector,
)
from .liveness.base import InvalidFrameError, as_clip
from .quality import CaptureQualityGate, QualityReport
from .recognition import FaceEmbedder, FaceMatcher
from .types import DetectorResult, FraudVerdict, PipelineResult


class FaceGuardPipeline:
    @classmethod
    def for_tier(cls, tier, base: Optional[FaceGuardConfig] = None, **kwargs) -> "FaceGuardPipeline":
        """Build a pipeline at the operating point of a :class:`~faceguard.policy.RiskTier`."""
        from .policy import config_for

        return cls(config=config_for(tier, base), **kwargs)

    def __init__(
        self,
        config: Optional[FaceGuardConfig] = None,
        embedder: Optional[FaceEmbedder] = None,
        matcher: Optional[FaceMatcher] = None,
        detector: Optional[FaceDetector] = None,
        fps: float = 30.0,
        quality_gate: Optional[CaptureQualityGate] = None,
        issuer: Optional[ChallengeIssuer] = None,
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
        self.parallax = ParallaxDetector()
        self.sensor = SensorNoiseDetector()
        self.banding = DisplayBandingDetector()
        self.subsurface = SubsurfaceScatteringDetector()
        self.issuer = issuer
        self.challenge_verifier = ChallengeVerifier(issuer=issuer)
        self.quality_gate = quality_gate or CaptureQualityGate(
            min_face_px=self.config.min_face_px,
            min_quality=self.config.min_capture_quality,
            require_colour=self.config.require_colour,
            require_clip=self.config.require_clip,
        )

    @property
    def detectors(self) -> Dict[str, object]:
        """Every liveness leg, keyed by name. Image-only legs take one frame."""
        return {
            "spectral": self.spectral,
            "texture": self.texture,
            "dct": self.dct,
            "rppg": self.rppg,
            "motion": self.motion,
            "parallax": self.parallax,
            "sensor": self.sensor,
            "banding": self.banding,
            "subsurface": self.subsurface,
        }

    # -- enrolment / recognition -------------------------------------------
    def enroll(self, name: str, face: np.ndarray) -> None:
        crop = self._largest_crop(face)
        self.matcher.enroll(name, self.embedder.embed(crop))

    # -- liveness ----------------------------------------------------------
    #: Legs that consume a single representative frame rather than the clip.
    _IMAGE_LEGS = ("spectral", "texture", "dct")

    def analyze(self, frames: np.ndarray) -> PipelineResult:
        """Run liveness fusion on an image or a clip (no recognition).

        Never raises for a bad *sample*: malformed input, unusable capture and
        detector crashes all resolve to an INDETERMINATE result carrying the
        reason. Production callers get a decision object on every path, which is
        what lets a kiosk say "please try again" instead of returning a 500.
        """
        started = time.perf_counter()
        try:
            clip = as_clip(frames)
        except InvalidFrameError as exc:
            return self._indeterminate(str(exc), started)

        quality = self.quality_gate(clip)
        if not quality.usable:
            # Refuse to judge rather than judge badly: an unusable capture is a
            # capture problem, and reporting it as FRAUD would blame the user for
            # the camera's failure.
            return self._indeterminate(
                "unusable_capture", started, quality=quality,
                degraded=quality.failures,
            )

        rep = self._representative(clip)
        results, degraded = self._run_detectors(clip, rep)

        available = sum(1 for r in results.values() if r.reliability > 0.0)
        if available < max(1, round(self.config.min_detector_availability * len(results))):
            # Too much of the ensemble is silent. The remaining evidence may look
            # decisive, but a decision made on a crippled ensemble is not one an
            # auditor could defend.
            return self._indeterminate(
                "insufficient_detectors", started, quality=quality,
                degraded=tuple(degraded), detectors=results,
            )

        score, confidence, verdict, _ = self.fusion.fuse(results.values())
        return PipelineResult(
            liveness_score=score,
            reliability=confidence,
            verdict=verdict,
            detectors=results,
            quality=quality,
            degraded=tuple(degraded),
            elapsed_ms=(time.perf_counter() - started) * 1e3,
        )

    def analyze_batch(self, clips: Sequence[np.ndarray]) -> List[PipelineResult]:
        """Analyse many samples, isolating failures to the sample that caused them."""
        return [self.analyze(c) for c in clips]

    # -- detector execution ------------------------------------------------
    def _run_detectors(self, clip: np.ndarray, rep: np.ndarray):
        """Run every leg under fault isolation and a soft latency budget.

        Two industrial realities are handled here. First, one detector's bug
        must not take down authentication for the whole estate — a crash becomes
        an abstention, which the fusion already knows how to absorb. Second, a
        queue at a turnstile cares more about answering in time than about
        squeezing in the last cue, so legs that would blow the budget abstain
        too. Both are *recorded*, so a monitoring system sees a degraded mode
        instead of an unexplained accuracy drop.
        """
        budget_s = self.config.time_budget_ms / 1e3
        started = time.perf_counter()
        results: Dict[str, DetectorResult] = {}
        degraded: List[str] = []

        for name, det in self.detectors.items():
            if budget_s > 0 and (time.perf_counter() - started) > budget_s:
                results[name] = DetectorResult.abstain(name, "time_budget_exhausted")
                degraded.append(name)
                continue
            sample = rep if name in self._IMAGE_LEGS else clip
            try:
                results[name] = det(sample)
            except Exception:  # noqa: BLE001 - deliberate isolation boundary
                results[name] = DetectorResult.abstain(name, "detector_error")
                degraded.append(name)
        return results, degraded

    def _indeterminate(
        self,
        reason: str,
        started: float,
        quality: Optional[QualityReport] = None,
        degraded=(),
        detectors: Optional[Dict[str, DetectorResult]] = None,
    ) -> PipelineResult:
        return PipelineResult(
            liveness_score=0.5,
            reliability=0.0,
            verdict=FraudVerdict.INDETERMINATE,
            detectors=detectors or {},
            quality=quality,
            degraded=tuple(degraded) or (reason,),
            elapsed_ms=(time.perf_counter() - started) * 1e3,
        )

    def identify_and_verify(self, frames: np.ndarray) -> PipelineResult:
        """Full pipeline: liveness fusion + identity match on the same input."""
        result = self.analyze(frames)
        if result.verdict == FraudVerdict.INDETERMINATE and not result.detectors:
            # The sample never got as far as liveness; matching it would only
            # produce an identity nobody should act on.
            return result
        rep = self._representative(as_clip(frames))
        crop = self._largest_crop((rep * 255).astype("uint8"))
        name, dist = self.matcher.identify(self.embedder.embed(crop))
        result.identity = name
        result.match_distance = dist
        return result

    # -- active challenge-response (for SUSPICIOUS cases) ------------------
    def issue_challenge(
        self, kind: Optional[ChallengeType] = None, nonce: Optional[int] = None
    ) -> Challenge:
        """Issue a challenge to escalate a SUSPICIOUS case.

        With an issuer configured the challenge is signed, time-limited and
        single-use; without one it is an unsigned prompt with a TTL, which is
        fine for a demo and not fine for production.
        """
        if self.issuer is not None and nonce is None:
            return self.issuer.issue(kind=kind)
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

        A passed challenge upgrades the verdict to GENUINE; a failed one is
        treated as an attack and downgraded to FRAUD. GENUINE/FRAUD results are
        returned unchanged — the challenge only adjudicates the ambiguous middle
        band. To demand a challenge of an already-GENUINE result (the CRITICAL
        tier's posture), call :meth:`enforce_challenge` instead.
        """
        if result.verdict is not FraudVerdict.SUSPICIOUS:
            return result
        return self._adjudicate(result, response_frames, challenge)

    def enforce_challenge(
        self,
        result: PipelineResult,
        response_frames: np.ndarray,
        challenge: Challenge,
    ) -> PipelineResult:
        """Require a challenge even for a passively-GENUINE result.

        This is what the CRITICAL risk tier calls (see
        :attr:`~faceguard.policy.TierPolicy.always_challenge`): for a vault door
        or a high-value transfer, passive evidence alone is never enough, because
        the one thing no recording can do is answer a prompt it has not seen.
        A genuine user who simply misses the prompt drops to SUSPICIOUS and may
        retry — only a forged or replayed response is treated as an attack.
        """
        if result.verdict is FraudVerdict.FRAUD:
            return result
        return self._adjudicate(result, response_frames, challenge)

    def _adjudicate(
        self,
        result: PipelineResult,
        response_frames: np.ndarray,
        challenge: Challenge,
    ) -> PipelineResult:
        was_genuine = result.verdict is FraudVerdict.GENUINE
        outcome = self.verify_challenge(response_frames, challenge)
        result.challenge_passed = outcome.passed
        if outcome.passed:
            result.verdict = FraudVerdict.GENUINE
        elif was_genuine and outcome.reason in ("expired", "action_not_performed"):
            # A passively-genuine subject who simply missed the prompt (looked
            # away, took too long) is not an attacker. Downgrade to SUSPICIOUS
            # and let them try again; reserve FRAUD for a response that was
            # actively wrong — a replayed or forged nonce.
            result.verdict = FraudVerdict.SUSPICIOUS
        else:
            result.verdict = FraudVerdict.FRAUD
        return result

    # -- helpers -----------------------------------------------------------
    def _representative(self, clip: np.ndarray) -> np.ndarray:
        # Median over time is robust to per-frame jitter and pulse modulation.
        return np.median(clip[..., :3], axis=0)

    def _largest_crop(self, image: np.ndarray) -> np.ndarray:
        box = self.detector.detect_largest(np.asarray(image))
        crop = box.crop(np.asarray(image))
        return crop if crop.size else np.asarray(image)
