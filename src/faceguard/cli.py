"""Command-line entry point: ``faceguard demo`` / ``faceguard analyze``."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .audit import AuditLog
from .pipeline import FaceGuardPipeline
from .policy import RiskTier, policy_for, tiers
from .sequential import SequentialVerifier, windows
from .utils import (
    synth_challenge_clip,
    synth_deepfake_clip,
    synth_injected_clip,
    synth_live_clip,
    synth_mask_clip,
    synth_replay_clip,
    synth_spoof_clip,
    synth_tilted_photo_clip,
)

#: Every attack class the synthetic bench can produce, and what should happen.
_ATTACKS = [
    ("live face", synth_live_clip, "genuine"),
    ("print / replay", synth_spoof_clip, "fraud"),
    ("screen replay (rolling-shutter banding)", synth_replay_clip, "fraud"),
    ("tilted photo (6-DoF hand motion)", synth_tilted_photo_clip, "fraud"),
    ("deepfake video", synth_deepfake_clip, "fraud"),
    ("injected stream (virtual camera)", synth_injected_clip, "fraud"),
    ("3-D silicone mask", synth_mask_clip, "not-genuine"),
]


def _print_result(label: str, result) -> None:
    print(f"\n=== {label} ===")
    print(f"  verdict          : {result.verdict.value.upper()}")
    print(f"  liveness score   : {result.liveness_score:.3f}")
    print(f"  fused confidence : {result.reliability:.3f}")
    if result.quality is not None:
        print(f"  capture quality  : {result.quality.summary()}")
    if result.degraded:
        print(f"  degraded         : {', '.join(result.degraded)}")
    for name, det in result.detectors.items():
        print(f"    - {name:<11} score={det.score:.3f}  reliability={det.reliability:.3f}")
    for hint in (result.quality.remediation() if result.quality else []):
        print(f"  ! {hint}")


def _cmd_demo(args: argparse.Namespace) -> int:
    pipe = FaceGuardPipeline()
    live = pipe.analyze(synth_live_clip(seed=args.seed))
    spoof = pipe.analyze(synth_spoof_clip(seed=args.seed))
    deepfake = pipe.analyze(synth_deepfake_clip(seed=args.seed))
    _print_result("LIVE face (genuine)", live)
    _print_result("SPOOF face (replay/print)", spoof)
    _print_result("DEEPFAKE face (GAN fingerprint)", deepfake)
    ok = (
        live.verdict.value == "genuine"
        and spoof.verdict.value == "fraud"
        and deepfake.verdict.value != "genuine"
    )
    print("\nSTLF separated live from spoof + deepfake:", "YES ✅" if ok else "NO ❌")

    # Challenge-response: escalate a SUSPICIOUS case with a random nonce.
    challenge = pipe.issue_challenge(nonce=args.seed)
    print(f"\n--- Challenge-response (for SUSPICIOUS cases) ---\nPrompt: {challenge.instruction}")
    good = pipe.verify_challenge(synth_challenge_clip(challenge.kind.value, respond=True), challenge)
    replay = pipe.verify_challenge(synth_challenge_clip("blink", respond=False), challenge)
    print(f"  matching response : passed={good.passed} (conf={good.confidence:.2f})")
    print(f"  replay / no action: passed={replay.passed}")
    return 0 if ok else 1


def _cmd_attacks(args: argparse.Namespace) -> int:
    """Score every synthetic attack class the library knows how to generate."""
    pipe = FaceGuardPipeline()
    print(f"{'attack class':<42} {'verdict':<14} {'live':>6} {'conf':>6}   expected")
    print("-" * 88)
    failures = 0
    for label, generator, expected in _ATTACKS:
        result = pipe.analyze(generator(seed=args.seed, frames=args.frames))
        got = result.verdict.value
        ok = got == expected if expected != "not-genuine" else got != "genuine"
        failures += not ok
        mark = " " if ok else "  <-- MISS"
        print(
            f"{label:<42} {got:<14} {result.liveness_score:>6.3f} "
            f"{result.reliability:>6.3f}   {expected}{mark}"
        )
    print(f"\n{len(_ATTACKS) - failures}/{len(_ATTACKS)} attack classes handled as expected")
    return 0 if failures == 0 else 1


def _cmd_policy(args: argparse.Namespace) -> int:
    """Show what each risk tier trades away."""
    for policy in tiers():
        print(f"\n=== {policy.tier.value.upper()} ===")
        print(f"  genuine >= {policy.genuine_threshold}   fraud <= {policy.fraud_threshold}")
        print(f"  min reliability {policy.min_reliability_for_genuine}   "
              f"min quality {policy.min_capture_quality}   min frames {policy.min_frames}")
        print(f"  challenge: {'always' if policy.always_challenge else 'on suspicion'}")
        print(f"  {policy.rationale}")
    return 0


def _cmd_stream(args: argparse.Namespace) -> int:
    """Sequential (SPRT) decisioning over a rolling window of a long capture."""
    pipe = FaceGuardPipeline.for_tier(RiskTier(args.tier))
    verifier = SequentialVerifier(far=args.far, frr=args.frr, max_windows=args.max_windows)
    log = AuditLog(salt=b"cli-demo-salt")

    generators = {"live": synth_live_clip, "spoof": synth_spoof_clip, "mask": synth_mask_clip}
    clip = generators[args.sample](seed=args.seed, frames=args.frames)

    print(f"tier={args.tier} sample={args.sample}  target FAR={args.far} FRR={args.frr}")
    decision = None
    for i, window in enumerate(windows(clip, size=args.window), start=1):
        result = pipe.analyze(window)
        log.append(result, subject=args.sample, window=i)
        decision = verifier.observe(result)
        print(f"  window {i:>2}: live={result.liveness_score:.3f} -> {decision.summary()}")
        if decision.state.value in ("accept", "reject"):
            break
    print(f"\nfinal: {decision.verdict.value.upper()}  ({decision.summary()})")
    print(f"audit chain intact: {log.verify() is None}  head={log.head[:16]}...")
    print(f"policy note: {policy_for(args.tier).rationale}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    try:
        import cv2
    except Exception:
        print("Reading image/video files needs OpenCV: pip install '.[full]'", file=sys.stderr)
        return 2
    pipe = FaceGuardPipeline()
    cap = cv2.VideoCapture(args.path)
    frames = []
    while len(frames) < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        print(f"No frames read from {args.path}", file=sys.stderr)
        return 2
    result = pipe.analyze(np.stack(frames).astype(np.float64) / 255.0)
    _print_result(args.path, result)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="faceguard", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the synthetic live-vs-spoof demo")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=_cmd_demo)

    at = sub.add_parser("attacks", help="score every synthetic attack class")
    at.add_argument("--seed", type=int, default=0)
    at.add_argument("--frames", type=int, default=90)
    at.set_defaults(func=_cmd_attacks)

    po = sub.add_parser("policy", help="show the risk-tier operating points")
    po.set_defaults(func=_cmd_policy)

    st = sub.add_parser("stream", help="sequential (SPRT) decisioning over a rolling window")
    st.add_argument("--sample", choices=("live", "spoof", "mask"), default="live")
    st.add_argument("--tier", choices=[t.value for t in RiskTier], default="standard")
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--frames", type=int, default=180)
    st.add_argument("--window", type=int, default=45)
    st.add_argument("--far", type=float, default=1e-3)
    st.add_argument("--frr", type=float, default=0.02)
    st.add_argument("--max-windows", type=int, default=12)
    st.set_defaults(func=_cmd_stream)

    a = sub.add_parser("analyze", help="analyze an image or video file (needs OpenCV)")
    a.add_argument("path")
    a.add_argument("--max-frames", type=int, default=90)
    a.set_defaults(func=_cmd_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
