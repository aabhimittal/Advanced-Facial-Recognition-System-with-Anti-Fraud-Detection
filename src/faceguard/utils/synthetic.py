"""Synthetic live-vs-spoof face generator.

This is what lets the whole project — demos *and* the test suite — run with only
numpy/scipy, no camera and no dataset. The generator deliberately bakes in the
*physical* differences the detectors look for, so a "live" sample really does
carry a pulse and rich micro-texture while a "spoof" really does carry a display
grid and no pulse. It is a teaching/verification tool, **not** a substitute for
evaluating on real anti-spoofing datasets (see docs/EVALUATION.md).

Live sample encodes:    natural 1/f spectrum · rich micro-texture · a periodic
                        rPPG pulse in the green channel · non-rigid micro-motion.
Spoof sample encodes:   display pixel grid (periodic HF peaks) · blurred/uniform
                        micro-texture · no pulse · rigid global motion only.
Deepfake sample encodes: GAN upsampling checkerboard (block-DCT fingerprint) ·
                        rich texture & synthesised motion (so it *looks* live) ·
                        no coherent pulse — the case the DCT head is built for.
"""

from __future__ import annotations

import numpy as np

from .image_ops import blur, face_blob, pink_field, warp_depth, warp_planar

#: Conversion gain of the simulated 8-bit sensor: photon shot-noise variance is
#: ``gain * signal``, which is exactly the relation SensorNoiseDetector fits.
_SENSOR_GAIN = 8e-6
_READ_NOISE = 4e-4


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
    # A face is a 3-D object: depth drives parallax when the head yaws, which is
    # what ParallaxDetector measures and no flat surface can imitate.
    depth = face_blob(size, size)
    for t in range(frames):
        phase = 2 * np.pi * hr_hz * t / fps
        frame = base.copy()
        # rPPG: subtle periodic green-channel modulation (blood volume pulse).
        frame[..., 1] *= 1.0 + 0.03 * np.sin(phase)
        # Non-rigid micro-motion: small time-varying local deformation of brightness.
        frame += 0.02 * np.sin(phase * 0.7) * expr
        # Slow head yaw -> depth-dependent horizontal displacement (parallax).
        yaw = 1.6 * np.sin(2 * np.pi * t / max(frames, 2))
        frame = warp_depth(frame, depth, yaw)
        # Tiny global jitter (sub-2px).
        frame = np.roll(frame, rng.integers(-1, 2), axis=0)
        clip[t] = np.clip(_sensor_noise(rng, frame), 0, 1)
    return clip


def synth_spoof_clip(seed: int = 0, size: int = 96, frames: int = 60) -> np.ndarray:
    """A replay/print spoof clip (T, H, W, 3): grid artefact, no pulse, rigid motion."""
    rng = np.random.default_rng(seed)
    base = _base_face(rng, size, spoof=True)
    clip = np.empty((frames, size, size, 3))
    for t in range(frames):
        # Rigid hand-held motion: whole "photo" translates together, no deformation.
        dy, dx = int(2 * np.sin(t / 8.0)), int(2 * np.cos(t / 11.0))
        moved = np.roll(np.roll(base, dy, 0), dx, 1)
        # The replay is still captured by a real camera, so it carries genuine
        # sensor noise: presentation attacks are NOT injection attacks.
        clip[t] = np.clip(_sensor_noise(rng, moved), 0, 1)
    return clip


def synth_deepfake_face(seed: int = 0, size: int = 128) -> np.ndarray:
    """A single RGB face carrying a GAN upsampling fingerprint (H, W, 3) in [0, 1]."""
    rng = np.random.default_rng(seed)
    return _gan_fingerprint(_base_face(rng, size, spoof=False))


def synth_deepfake_clip(seed: int = 0, size: int = 96, frames: int = 60) -> np.ndarray:
    """A deepfake video (T, H, W, 3): looks live (texture + motion) but has the GAN
    block-DCT fingerprint and no coherent blood-volume pulse."""
    rng = np.random.default_rng(seed)
    base = _gan_fingerprint(_base_face(rng, size, spoof=False))
    clip = np.empty((frames, size, size, 3))
    expr = pink_field(rng, size, size, beta=2.0)[..., None]
    for t in range(frames):
        # Synthesised non-rigid motion (a generated video is not a still photo)...
        # ...driven by BROADBAND random amplitude, so there is no single-frequency
        # in-band peak -> no coherent pulse for rPPG to lock onto.
        frame = base + rng.normal(0.0, 0.02) * expr
        clip[t] = np.clip(np.roll(frame, rng.integers(-1, 2), axis=0), 0, 1)
    return clip


def synth_challenge_clip(
    kind: str, respond: bool = True, seed: int = 0, size: int = 96, frames: int = 60
) -> np.ndarray:
    """A live-face clip that performs (``respond=True``) or omits the requested
    action. ``kind`` is a ``ChallengeType`` value: 'blink' | 'turn_left' |
    'turn_right' | 'nod'. A non-responding clip should fail the matching check —
    which is exactly how a pre-recorded replay behaves against a random prompt."""
    rng = np.random.default_rng(seed)
    base = _base_face(rng, size, spoof=False)
    clip = np.empty((frames, size, size, 3))
    mid = frames // 2
    for t in range(frames):
        frame = base.copy()
        if respond:
            f = t / (frames - 1)
            if kind == "turn_left":
                frame = np.roll(frame, -round(12 * f), axis=1)
            elif kind == "turn_right":
                frame = np.roll(frame, round(12 * f), axis=1)
            elif kind == "nod":
                frame = np.roll(frame, round(8 * np.sin(np.pi * f)), axis=0)
            elif kind == "blink" and abs(t - mid) <= 2:
                # Briefly darken the eye band (rows ~0.2-0.45 of height).
                frame[int(0.2 * size):int(0.45 * size)] *= 0.5
        # A little involuntary jitter regardless, so it is a real (live) clip.
        frame = np.roll(frame, rng.integers(-1, 2), axis=0)
        clip[t] = np.clip(_sensor_noise(rng, frame), 0, 1)
    return clip


def synth_tilted_photo_clip(seed: int = 0, size: int = 96, frames: int = 60) -> np.ndarray:
    """A print attack with full six-degree-of-freedom hand motion.

    The attacker does what any human holding a photo does: tilts, rotates and
    advances it, rather than sliding it flat across the frame. The motion is
    richer, but the geometry is not — every displacement it can produce is still
    explained by a single homography, which is exactly what the parallax test
    certifies and no amount of hand movement can change.
    """
    rng = np.random.default_rng(seed)
    base = _base_face(rng, size, spoof=True)
    clip = np.empty((frames, size, size, 3))
    for t in range(frames):
        f = t / max(frames - 1, 1)
        frame = warp_planar(
            base,
            angle_deg=3.0 * np.sin(2 * np.pi * f),
            scale=1.0 + 0.02 * f,
            dy=1.5 * np.sin(2 * np.pi * f),
            dx=2.0 * f,
        )
        clip[t] = np.clip(_sensor_noise(rng, frame), 0, 1)
    return clip


def synth_replay_clip(
    seed: int = 0, size: int = 96, frames: int = 60, band_cycles: float = 6.0
) -> np.ndarray:
    """A screen replay seen through a rolling shutter.

    Adds the beat artefact a printed photo can never have: horizontal luminance
    bands that scroll vertically as the display refresh drifts against the
    sensor's row read-out.
    """
    rng = np.random.default_rng(seed)
    base = _base_face(rng, size, spoof=True)
    rows = np.arange(size)[:, None, None] / size
    clip = np.empty((frames, size, size, 3))
    for t in range(frames):
        # Phase advances every frame -> the bands scroll down the image.
        phase = 2 * np.pi * (band_cycles * rows + 0.11 * t)
        banding = 1.0 + 0.05 * np.sin(phase)
        dy, dx = int(2 * np.sin(t / 8.0)), int(2 * np.cos(t / 11.0))
        frame = np.roll(np.roll(base, dy, 0), dx, 1) * banding
        clip[t] = np.clip(_sensor_noise(rng, frame), 0, 1)
    return clip


def synth_injected_clip(seed: int = 0, size: int = 96, frames: int = 60) -> np.ndarray:
    """A deepfake fed straight into the capture pipeline (virtual camera).

    There is no print, no screen and no moiré — every *presentation* cue is
    clean. What it cannot fake is silicon: the stream carries no
    intensity-dependent photon shot noise, only the flat, uniform noise a
    generator or codec leaves behind. That is the photon-transfer tell.
    """
    rng = np.random.default_rng(seed)
    base = _gan_fingerprint(_base_face(rng, size, spoof=False))
    depth = face_blob(size, size)
    clip = np.empty((frames, size, size, 3))
    for t in range(frames):
        yaw = 1.6 * np.sin(2 * np.pi * t / max(frames, 2))   # rendered 3-D motion
        frame = warp_depth(base, depth, yaw)
        # Uniform, intensity-independent noise: the signature of synthesis, not
        # of a sensor (a real sensor's noise grows with brightness).
        frame = frame + rng.normal(0.0, 6e-4, frame.shape)
        clip[t] = np.clip(frame, 0, 1)
    return clip


def synth_mask_clip(seed: int = 0, size: int = 96, frames: int = 60) -> np.ndarray:
    """A 3-D silicone/resin mask attack.

    The hardest presentation attack: it is genuinely three-dimensional, so it
    produces real parallax and defeats every planar-geometry test. It is caught
    on the *material* axis instead — cast silicone has no blood-volume pulse and
    a smoother, more uniform micro-texture than skin.
    """
    rng = np.random.default_rng(seed)
    # Cast silicone keeps the face's shape and sharpness but not its micro-relief:
    # pores, fine wrinkles and capillary mottling are simply absent.
    base = _base_face(rng, size, spoof=False, texture_amp=0.04, subsurface=False)
    base = np.clip(base * 1.02 - 0.01, 0, 1)        # slightly waxy, flat tone
    depth = face_blob(size, size)
    clip = np.empty((frames, size, size, 3))
    for t in range(frames):
        yaw = 1.6 * np.sin(2 * np.pi * t / max(frames, 2))
        frame = warp_depth(base, depth, yaw)        # a worn mask really does move in 3-D
        frame = np.roll(frame, rng.integers(-1, 2), axis=0)
        clip[t] = np.clip(_sensor_noise(rng, frame), 0, 1)   # captured by a real camera
    return clip


def _sensor_noise(rng: np.random.Generator, rgb: np.ndarray) -> np.ndarray:
    """Add physically-shaped sensor noise: ``var = gain * signal + read_noise**2``."""
    signal = np.clip(rgb, 0.0, 1.0)
    sigma = np.sqrt(_SENSOR_GAIN * signal + _READ_NOISE**2)
    return rgb + rng.normal(0.0, 1.0, rgb.shape) * sigma


def _gan_fingerprint(rgb: np.ndarray, strength: float = 0.06) -> np.ndarray:
    """Inject a transposed-convolution "checkerboard" fingerprint.

    Downsample-then-nearest-upsample creates period-2 block structure, and an
    explicit Nyquist checkerboard adds the tell-tale high-frequency corner energy
    that block-DCT analysis detects.
    """
    h, w, _ = rgb.shape
    small = rgb[::2, ::2]
    up = np.kron(small, np.ones((2, 2, 1)))[:h, :w]
    out = 0.5 * rgb + 0.5 * up
    yy, xx = np.mgrid[0:h, 0:w]
    checker = ((xx + yy) % 2) * 2.0 - 1.0  # ±1 at period 2 (Nyquist)
    out = out * (1.0 + strength * checker[..., None])
    return np.clip(out, 0, 1)


def _base_face(
    rng: np.random.Generator,
    size: int,
    spoof: bool,
    texture_amp: float = 0.18,
    subsurface: bool = True,
) -> np.ndarray:
    struct = face_blob(size, size)
    skin = 0.55 + 0.35 * struct  # base luminance
    texture = pink_field(rng, size, size, beta=1.0)  # rich natural micro-texture
    img = skin + texture_amp * (texture - 0.5)
    rgb = np.stack([img * 1.02, img * 0.85, img * 0.78], axis=-1)  # warm skin tone

    if subsurface:
        # Skin is translucent and red light penetrates deepest, so fine detail
        # survives in green/blue and is smeared away in red. Opaque materials
        # (silicone, resin, paper, pixels) carry identical detail in all three.
        rgb[..., 0] = blur(rgb[..., 0], sigma=1.4)

    if spoof:
        # 1) Print/display blur removes genuine skin micro-texture.
        rgb = blur(rgb, sigma=1.6)
        # 2) Overlay a regular display pixel grid -> periodic high-frequency peaks.
        yy, xx = np.mgrid[0:size, 0:size]
        grid = (np.sin(2 * np.pi * xx / 4.0) * np.sin(2 * np.pi * yy / 4.0))
        rgb *= (1.0 + 0.22 * grid[..., None])

    return np.clip(rgb, 0, 1)
