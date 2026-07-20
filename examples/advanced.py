"""Advanced example: DCT deepfake head, learned weights, challenge-response.

    python examples/advanced.py

Runs with just numpy + scipy.
"""

from faceguard import ChallengeType, FaceGuardPipeline, FraudVerdict
from faceguard.config import FaceGuardConfig
from faceguard.fusion import fit_fusion_weights, fuse
from faceguard.utils import (
    synth_challenge_clip,
    synth_deepfake_clip,
    synth_live_clip,
    synth_spoof_clip,
)

_GENS = [(synth_live_clip, 1), (synth_spoof_clip, 0), (synth_deepfake_clip, 0)]


def main() -> None:
    pipe = FaceGuardPipeline()

    print("1) DCT deepfake head catches a face that otherwise looks live:")
    df = pipe.analyze(synth_deepfake_clip(seed=0))
    print("   ", df.summary(), " dct_score=%.3f" % df.detectors["dct"].score)

    print("\n2) Learn fusion weights from a labelled PAD set:")
    samples, labels = [], []
    for seed in range(8):
        for gen, label in _GENS:
            samples.append(pipe.analyze(gen(seed=seed)).detectors)
            labels.append(label)
    learned = fit_fusion_weights(samples, labels)
    print("   learned weights:", {k: round(v, 2) for k, v in learned.detector_weights.items()})

    def accuracy(cfg):
        ok = n = 0
        for seed in range(100, 106):
            for gen, label in _GENS:
                dets = pipe.analyze(gen(seed=seed)).detectors
                _, _, verdict, _ = fuse(dets.values(), cfg)
                ok += (verdict == FraudVerdict.GENUINE) == bool(label)
                n += 1
        return ok / n

    print("   held-out accuracy  default -> %.2f" % accuracy(FaceGuardConfig()))
    print("   held-out accuracy  learned -> %.2f" % accuracy(learned))

    print("\n3) Challenge-response resolves a SUSPICIOUS case:")
    challenge = pipe.issue_challenge(ChallengeType.TURN_RIGHT, nonce=7)
    print("   prompt:", challenge.instruction)
    from faceguard.types import PipelineResult

    good = pipe.resolve_suspicious(
        PipelineResult(0.5, 0.9, FraudVerdict.SUSPICIOUS),
        synth_challenge_clip("turn_right", seed=1),
        challenge,
    )
    replay = pipe.resolve_suspicious(
        PipelineResult(0.5, 0.9, FraudVerdict.SUSPICIOUS),
        synth_challenge_clip("nod", seed=1),  # wrong action = replay
        challenge,
    )
    print("   correct response -> %s" % good.verdict.value)
    print("   replay (wrong)   -> %s" % replay.verdict.value)


if __name__ == "__main__":
    main()
