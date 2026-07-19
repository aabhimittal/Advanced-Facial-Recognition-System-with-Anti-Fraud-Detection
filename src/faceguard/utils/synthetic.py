"""Synthetic live-vs-spoof face generator.

This is what lets the whole project — demos *and* the test suite — run with only
numpy/scipy, no camera and no dataset. The generator deliberately bakes in the
*physical* differences the detectors look for, so a "live" sample really does
carry a pulse and rich micro-texture while a "spoof" really does carry a display
grid and no pulse. It is a teaching/verification tool, **not** a substitute for
evaluating on real anti-spoofing datasets (see docs/EVALUATION.md).

Live sample encodes:  natural 1/f spectrum · rich micro-texture · a periodic
                      rPPG pulse in the green channel · non-rigid micro-motion.
Spoof sample encodes: display pixel grid (periodic HF peaks) · blurred/uniform
                      micro-texture · no pulse · rigid global motion only.
"""

from __future__ import annotations

import numpy as np

from .image_ops import blur, face_blob, pink_field


def synth_face(seed: int = 0, size: int = 128, spoof: bool = False) -> np.ndarray:
    """A single RGB face crop (H, W, 3) in [0, 1]."""
    rng = np.random.default_rng(seed)
    return _base_face(rng, size, spoof)


def synth_live_clip(seed: int = 0, size: int = 96, frames: int = 60, fps: float = 30.0) -> np.ndarray:
    """A live-face clip (T, H, W, 3) with pulse and non-rigid micro-motion."""
    rng = np.random.default_rng(seed)
    base = _base_face(rng, size, spoof=False)
    hr_hz = 1.2  # ~72 bpm
    clip = np.empty((frames, size, size, 3))
    # Slowly varying "expression" field -> non-rigid residual motion.
    expr = pink_field(rng, size, size, beta=2.0)[..., None]
    for t in range(frames):
        phase = 2 * np.pi * hr_hz * t / fps
        frame = base.copy()
        # rPPG: subtle periodic green-channel modulation (blood volume pulse).
        frame[..., 1] *= 1.0 + 0.03 * np.sin(phase)
        # Non-rigid micro-motion: small time-varying local deformation of brightness.
        frame += 0.02 * np.sin(phase * 0.7) * expr
        # Tiny global jitter (sub-2px).
        frame = np.roll(frame, rng.integers(-1, 2), axis=0)
        clip[t] = np.clip(frame, 0, 1)
    return clip


def synth_spoof_clip(seed: int = 0, size: int = 96, frames: int = 60) -> np.ndarray:
    """A replay/print spoof clip (T, H, W, 3): grid artefact, no pulse, rigid motion."""
    rng = np.random.default_rng(seed)
    base = _base_face(rng, size, spoof=True)
    clip = np.empty((frames, size, size, 3))
    for t in range(frames):
        # Rigid hand-held motion: whole "photo" translates together, no deformation.
        dy, dx = int(2 * np.sin(t / 8.0)), int(2 * np.cos(t / 11.0))
        clip[t] = np.clip(np.roll(np.roll(base, dy, 0), dx, 1), 0, 1)
    return clip


def _base_face(rng: np.random.Generator, size: int, spoof: bool) -> np.ndarray:
    struct = face_blob(size, size)
    skin = 0.55 + 0.35 * struct  # base luminance
    texture = pink_field(rng, size, size, beta=1.0)  # rich natural micro-texture
    img = skin + 0.18 * (texture - 0.5)
    rgb = np.stack([img * 1.02, img * 0.85, img * 0.78], axis=-1)  # warm skin tone

    if spoof:
        # 1) Print/display blur removes genuine skin micro-texture.
        rgb = blur(rgb, sigma=1.6)
        # 2) Overlay a regular display pixel grid -> periodic high-frequency peaks.
        yy, xx = np.mgrid[0:size, 0:size]
        grid = (np.sin(2 * np.pi * xx / 4.0) * np.sin(2 * np.pi * yy / 4.0))
        rgb *= (1.0 + 0.22 * grid[..., None])

    return np.clip(rgb, 0, 1)
