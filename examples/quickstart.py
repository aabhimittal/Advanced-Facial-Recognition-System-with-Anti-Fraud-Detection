"""Minimal end-to-end example — runs with just numpy + scipy.

    python examples/quickstart.py
"""

import numpy as np

from faceguard import FaceGuardPipeline
from faceguard.utils import synth_live_clip, synth_spoof_clip


def main() -> None:
    pipe = FaceGuardPipeline()

    # Enrol an authorised user from a genuine live clip.
    live = synth_live_clip(seed=42)
    reference = (np.median(live[..., :3], axis=0) * 255).astype("uint8")
    pipe.enroll("authorised_user", reference)

    print("1) Genuine live user tries to authenticate:")
    genuine = pipe.identify_and_verify(live)
    print("   ", genuine.summary())
    print("    trustworthy match?", genuine.is_trustworthy_match())

    print("\n2) Attacker replays a spoof of the same user:")
    spoof = pipe.identify_and_verify(synth_spoof_clip(seed=42))
    print("   ", spoof.summary())
    print("    trustworthy match?", spoof.is_trustworthy_match(),
          "  <- identity may match, but liveness rejects it")

    print("\nPer-cue breakdown for the spoof (contribution audit):")
    for name, det in spoof.detectors.items():
        print(f"   {name:<9} score={det.score:.3f} reliability={det.reliability:.3f} {det.detail}")


if __name__ == "__main__":
    main()
