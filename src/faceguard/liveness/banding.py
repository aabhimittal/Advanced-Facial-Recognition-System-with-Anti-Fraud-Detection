"""Rolling-shutter banding detector — the *screen replay* leg.

The physics
-----------
Almost every camera in the field uses a rolling shutter: rows are exposed
sequentially, not simultaneously. Almost every display refreshes its backlight
or pixels periodically (60/120 Hz LCD backlight PWM, per-row OLED scan-out).
Photograph a display with a rolling shutter and the two periodicities beat
against each other, producing **horizontal luminance bands that drift vertically
from frame to frame**.

That drift is the signature this detector reads, and it is remarkably hard to
suppress: it is created by the *capture geometry*, not by the content, so it
survives the attacker's best efforts at a high-resolution, glare-free,
moiré-suppressed replay. It is also what distinguishes a screen replay from a
printed photo, which the texture and spectral legs treat identically.

The method
----------
1. Collapse each frame to a row-luminance profile, cancelling content by
   subtracting the temporal mean of each row (a static face contributes nothing).
2. Take the spatial FFT down the row axis: banding concentrates energy in a
   narrow band of vertical spatial frequencies, whereas real illumination change
   is broadband and low-frequency.
3. Require *temporal phase drift* at that frequency — a genuine beat marches
   steadily up or down the frame, while a static shadow does not move.

A live face scores high; a screen replay scores low. When no periodic row
structure is present at all, the detector abstains rather than voting "live",
because absence of banding is only weak evidence (a global-shutter camera, or a
display in phase with the sensor, produces none).
"""

from __future__ import annotations

import numpy as np

from ..types import DetectorResult
from .base import as_clip, one_sided_score, to_grayscale


class DisplayBandingDetector:
    name = "banding"

    def __init__(self, min_frames: int = 10, min_rows: int = 48):
        self.min_frames = int(min_frames)
        self.min_rows = int(min_rows)

    def __call__(self, frames: np.ndarray) -> DetectorResult:
        clip = as_clip(frames)
        t, h = clip.shape[0], clip.shape[1]
        if t < self.min_frames or h < self.min_rows:
            return DetectorResult.abstain(self.name, "insufficient_capture")

        gray = np.stack([to_grayscale(f) for f in clip])       # (T, H, W)
        rows = gray.mean(axis=2)                               # (T, H) row profiles
        # Cancel static content and global brightness drift: what remains is the
        # time-varying *row-wise* modulation, i.e. banding if any exists.
        rows = rows - rows.mean(axis=0, keepdims=True)
        rows = rows - rows.mean(axis=1, keepdims=True)

        win = np.hanning(h)[None, :]
        spec = np.fft.rfft(rows * win, axis=1)                 # (T, F)
        power = (np.abs(spec) ** 2).mean(axis=0)
        power[0] = 0.0                                         # drop DC
        if power.sum() <= 0:
            return DetectorResult.abstain(self.name, "no_row_signal")

        # Ignore the lowest frequencies: a real shadow gradient across the face
        # is a 1-2 cycle phenomenon, banding sits well above it.
        lo = max(2, h // 64)
        band = power[lo:]
        if band.size < 4:
            return DetectorResult.abstain(self.name, "no_row_signal")
        k = int(np.argmax(band)) + lo
        peak_ratio = float(band.max() / (np.median(band) + 1e-18))

        # Phase at the dominant frequency must advance monotonically in time for
        # a true rolling-shutter beat (the bands scroll).
        phase = np.unwrap(np.angle(spec[:, k]))
        drift = float(np.abs(np.polyfit(np.arange(t), phase, 1)[0]))    # rad/frame
        steadiness = float(1.0 / (1.0 + np.std(np.diff(phase))))

        # Banding evidence needs all three: a sharp spatial peak, a real drift
        # rate, and a steady one.
        peak_evidence = float(np.clip((peak_ratio - 8.0) / 40.0, 0.0, 1.0))
        drift_evidence = float(np.clip(drift / 0.25, 0.0, 1.0)) * steadiness
        evidence = peak_evidence * drift_evidence

        # One-sided: seeing the beat is near-conclusive, not seeing it proves
        # little (a global-shutter sensor, or a display in phase with it, shows
        # no banding at all).
        score = one_sided_score(evidence, scale=0.2)
        # We are confident when we *see* banding; absence is weaker evidence, so
        # reliability is asymmetric by design.
        reliability = float(np.clip(0.2 + 0.8 * evidence, 0.0, 1.0)) * float(
            np.clip((t - self.min_frames) / 20.0, 0.3, 1.0)
        )
        return DetectorResult(
            self.name,
            score,
            reliability,
            {
                "band_freq_cycles": float(k),
                "peak_ratio": peak_ratio,
                "phase_drift_rad_per_frame": drift,
                "evidence": evidence,
            },
        )
