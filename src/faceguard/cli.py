"""Command-line entry point: ``faceguard demo`` / ``faceguard analyze``."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .pipeline import FaceGuardPipeline
from .utils import synth_challenge_clip, synth_deepfake_clip, synth_live_clip, synth_spoof_clip


def _print_result(label: str, result) -> None:
    print(f"\n=== {label} ===")
    print(f"  verdict          : {result.verdict.value.upper()}")
    print(f"  liveness score   : {result.liveness_score:.3f}")
    print(f"  fused confidence : {result.reliability:.3f}")
    for name, det in result.detectors.items():
        print(f"    - {name:<9} score={det.score:.3f}  reliability={det.reliability:.3f}")


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

    a = sub.add_parser("analyze", help="analyze an image or video file (needs OpenCV)")
    a.add_argument("path")
    a.add_argument("--max-frames", type=int, default=90)
    a.set_defaults(func=_cmd_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
