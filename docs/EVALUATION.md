# Evaluating FaceGuard honestly

The numbers printed by `faceguard demo` are on **synthetic** data whose physics
we control. They demonstrate that the pipeline *wires up and separates* clearly
live vs clearly spoofed inputs — they are **not** an accuracy claim on real
attacks. This page describes how to get real numbers.

## Metrics (ISO/IEC 30107-3)

Report presentation-attack detection with the standard error rates:

- **APCER** — Attack Presentation Classification Error Rate: fraction of spoofs
  wrongly accepted as genuine (a *security* failure). Report the worst APCER
  across attack types.
- **BPCER** — Bona-fide Presentation Classification Error Rate: fraction of
  genuine users wrongly rejected (a *usability* failure).
- **ACER** = (APCER + BPCER) / 2.

Also report a **DET curve** (APCER vs BPCER as the `genuine_threshold` sweeps)
and, for the fusion score, ROC-AUC.

## Public datasets to evaluate on

| Dataset | Attack types | Notes |
|---------|--------------|-------|
| Replay-Attack (Idiap) | print, mobile/HD replay | classic baseline |
| CASIA-FASD | print, cut-photo, replay | small, quick to iterate |
| OULU-NPU | print, replay | 4 protocols, cross-device/illumination |
| CASIA-SURF | print + depth/IR | multi-modal; use RGB stream here |

These require registration/licences; FaceGuard does **not** redistribute them.

## Suggested harness (sketch)

```python
from faceguard import FaceGuardPipeline, FraudVerdict

pipe = FaceGuardPipeline()
y_true, y_score = [], []
for clip, is_genuine in load_your_dataset():     # clip: (T,H,W,3) RGB in [0,1]
    r = pipe.analyze(clip)
    y_true.append(is_genuine)
    y_score.append(r.liveness_score)             # continuous score for ROC/DET

# APCER/BPCER at a chosen operating threshold:
def rates(y_true, y_score, thr):
    apcer = sum(s >= thr for s, t in zip(y_score, y_true) if not t) / max(1, sum(not t for t in y_true))
    bpcer = sum(s <  thr for s, t in zip(y_score, y_true) if t)     / max(1, sum(y_true))
    return apcer, bpcer, (apcer + bpcer) / 2
```

## Tuning workflow

1. Split into train/dev/test by **subject** (never leak a subject across splits).
2. On train, fit `detector_weights` (logistic regression of the label on each
   cue's `logit(sᵢ)`) — STLF is linear in log-odds, so the coefficients are the
   weights.
3. Pick `genuine_threshold` / `fraud_threshold` on dev at your target APCER.
4. Report APCER/BPCER/ACER + DET on the untouched test set, per attack type.

## Known limitations to state in any write-up

- rPPG needs enough frames and reasonable lighting; it abstains otherwise (by
  design), so short clips lean on the image cues.
- The synthetic generator is not a domain-realistic simulator; do not quote its
  numbers as field performance.
- The default numpy embedder is for demos; use the ArcFace backend for
  recognition benchmarks.
