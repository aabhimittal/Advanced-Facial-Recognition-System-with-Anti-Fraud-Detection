"""Challenge issuing and response verification.

The verifier reconstructs the head-motion *trajectory* over the clip (via
low-pass phase correlation between the first frame and every later frame) plus an
illumination-normalised *eye-region* signal, then checks whether the trajectory /
signal matches the specific challenge that was issued. Everything runs on
numpy/scipy, so it is fully testable against the synthetic response generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from ..liveness.base import as_clip, to_grayscale


class ChallengeType(str, Enum):
    BLINK = "blink"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    NOD = "nod"


_INSTRUCTIONS = {
    ChallengeType.BLINK: "Please blink now.",
    ChallengeType.TURN_LEFT: "Please turn your head to the left.",
    ChallengeType.TURN_RIGHT: "Please turn your head to the right.",
    ChallengeType.NOD: "Please nod your head.",
}


@dataclass
class Challenge:
    kind: ChallengeType
    nonce: int  # unpredictable id, ties a response to this specific prompt

    @property
    def instruction(self) -> str:
        return _INSTRUCTIONS[self.kind]


@dataclass
class ChallengeResult:
    challenge: Challenge
    passed: bool
    confidence: float
    detail: Dict[str, float] = field(default_factory=dict)


def issue_challenge(kind: Optional[ChallengeType] = None, nonce: int = 0) -> Challenge:
    """Issue a challenge. Pass ``kind`` for determinism; otherwise derive one from
    the nonce (callers supply an unpredictable nonce per authentication attempt)."""
    if kind is None:
        kind = list(ChallengeType)[nonce % len(ChallengeType)]
    return Challenge(kind=kind, nonce=nonce)


class ChallengeVerifier:
    def __init__(self, motion_thresh: float = 2.0, blink_prominence: float = 3.0):
        self.motion_thresh = motion_thresh          # pixels of net head travel
        self.blink_prominence = blink_prominence     # dip vs baseline noise (sigmas)

    def verify(self, frames: np.ndarray, challenge: Challenge) -> ChallengeResult:
        clip = as_clip(frames)
        dy, dx = _trajectory(clip)                  # per-frame shift vs frame 0
        blink = _blink_signal(clip)

        x_range = float(dx.max() - dx.min())
        y_range = float(dy.max() - dy.min())
        net_dx = float(dx[len(dx) // 2:].mean())    # net horizontal travel (signed)
        blink_prom = float(blink)

        k = challenge.kind
        if k == ChallengeType.BLINK:
            passed = blink_prom >= self.blink_prominence
            conf = _ratio(blink_prom, self.blink_prominence)
            detail = {"blink_prominence": blink_prom}
        elif k in (ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT):
            want_sign = -1.0 if k == ChallengeType.TURN_LEFT else 1.0
            directional = net_dx * want_sign        # positive if turned the right way
            passed = directional >= self.motion_thresh and x_range > y_range
            conf = _ratio(directional, self.motion_thresh)
            detail = {"net_dx": net_dx, "x_range": x_range, "y_range": y_range}
        else:  # NOD
            passed = y_range >= self.motion_thresh and y_range > x_range
            conf = _ratio(y_range, self.motion_thresh)
            detail = {"y_range": y_range, "x_range": x_range}

        return ChallengeResult(challenge, bool(passed), float(np.clip(conf, 0, 1)), detail)


def _ratio(value: float, thresh: float) -> float:
    return max(0.0, value) / (2.0 * thresh)


def _trajectory(clip: np.ndarray):
    """Signed (dy, dx) head shift of every frame relative to frame 0, in pixels."""
    gray = [gaussian_filter(to_grayscale(f), 2.0) for f in clip]
    ref = gray[0]
    max_shift = min(ref.shape) // 3
    dy = np.zeros(len(gray))
    dx = np.zeros(len(gray))
    for t in range(1, len(gray)):
        dy[t], dx[t] = _pairwise_shift(ref, gray[t], max_shift)
    return dy, dx


def _pairwise_shift(ref: np.ndarray, img: np.ndarray, max_shift: int):
    """Signed translation of ``img`` relative to ``ref`` via phase correlation."""
    fr = np.fft.fft2(ref)
    fi = np.fft.fft2(img)
    cross = fr * np.conj(fi)
    cross /= np.abs(cross) + 1e-9
    corr = np.fft.ifft2(cross).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy = peak[0] if peak[0] <= ref.shape[0] // 2 else peak[0] - ref.shape[0]
    dx = peak[1] if peak[1] <= ref.shape[1] // 2 else peak[1] - ref.shape[1]
    # Negate so the sign matches image-space translation (content moved right -> +dx).
    return float(np.clip(-dy, -max_shift, max_shift)), float(np.clip(-dx, -max_shift, max_shift))


def _blink_signal(clip: np.ndarray) -> float:
    """Prominence of a transient darkening in the eye band, normalised by lighting.

    Subtracting a lower-face reference cancels global illumination changes, so only
    a *localised* eye event (a blink) produces a large negative excursion.
    """
    h = clip.shape[1]
    eye = clip[:, int(0.2 * h):int(0.45 * h)].mean(axis=(1, 2, 3))
    cheek = clip[:, int(0.6 * h):int(0.85 * h)].mean(axis=(1, 2, 3))
    diff = eye - cheek
    diff = diff - np.median(diff)
    dip = -diff.min()                       # blink -> eyes darken -> negative diff
    return float(dip / (diff.std() + 1e-6))
