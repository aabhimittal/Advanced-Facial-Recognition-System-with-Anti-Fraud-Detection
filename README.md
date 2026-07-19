# FaceGuard 🛡️ — Facial Recognition with Spectro-Temporal Liveness Fusion

> Recognising *who* a face belongs to is a solved-ish problem. Knowing whether
> there is a **real, live human** in front of the camera — and not a photo, a
> phone screen, a printed mask, or a deepfake — is where fraud actually happens.
> FaceGuard is a small, readable, from-scratch research framework built around a
> novel anti-fraud method: **Spectro-Temporal Liveness Fusion (STLF)**.

[![CI](https://github.com/aabhimittal/advanced-facial-recognition-system-with-anti-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/aabhimittal/advanced-facial-recognition-system-with-anti-fraud-detection/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.8%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## Why this project exists

Face recognition alone is *insecure*: it will happily match a **photo** of an
authorised user and unlock the door. The interesting, unsolved, actively-
researched problem is **presentation-attack detection (PAD) / liveness** —
telling a genuine live face apart from a spoof. FaceGuard is built to *teach*
that problem end-to-end and to propose a genuinely novel fusion idea, with code
you can read in an afternoon.

**Design goals**

- **Runs anywhere, instantly.** The entire anti-fraud core depends only on
  `numpy` + `scipy`. `python -m faceguard.cli demo` works with no camera, no
  dataset, no GPU, no model downloads.
- **Novel, defensible contribution.** Not "yet another CNN" — a fusion
  *algorithm* (STLF) with a clear probabilistic formulation (below).
- **Honest.** We make **no** unverified benchmark claims. What we assert, the
  test suite proves on synthetic data; how to evaluate on *real* PAD datasets is
  documented in [`docs/EVALUATION.md`](docs/EVALUATION.md).
- **Pluggable.** Deep face detectors (RetinaFace) and embeddings (ArcFace) drop
  in as optional adapters without touching the anti-fraud logic.

---

## The novel idea: Spectro-Temporal Liveness Fusion (STLF)

The premise: **a fraudster rarely defeats *orthogonal* physical cues at the same
time.** A high-res print beats a naive texture check; a good display beats a
colour check; a replayed video may even carry a faint pulse. So FaceGuard runs
four cues that fail *independently* and fuses them — but with a twist most
published systems lack.

| Cue | Physical signal it exploits | Spoofs it catches |
|-----|------------------------------|-------------------|
| **Spectral** | Natural `1/f` image spectrum | screens, prints, GAN deepfakes (periodic HF peaks) |
| **rPPG pulse** | Heartbeat colour change in skin | prints, masks, still images (no pulse) |
| **Micro-texture** | Stochastic, aperiodic skin detail | display grids & halftone (periodic), matte prints (smooth) |
| **Micro-motion** | Non-rigid 3-D facial deformation | a waved flat photo (rigid motion only) |

**The twist — confidence-weighted Bayesian fusion.** Every detector returns not
just a liveness score `sᵢ ∈ [0,1]` but a **self-estimated reliability**
`rᵢ ∈ [0,1]`. rPPG on a 4-frame clip *knows* it can't be trusted and reports
`rᵢ ≈ 0`; the spectral cue on a 20-pixel crop does the same. Fusion happens in
log-odds space, and reliability directly scales how far each cue can move the
posterior:

```
L  = logit(p₀) + Σᵢ (wᵢ · rᵢ) · logit(sᵢ)
p  = sigmoid(L)                       # fused liveness probability
R  = 1 − exp(−Σᵢ wᵢ · rᵢ)            # fused confidence
```

Because `logit(0.5) = 0`, a detector that abstains (`sᵢ=0.5` or `rᵢ=0`)
contributes *nothing* — the pool degrades gracefully to whatever evidence is
actually trustworthy on **this** sample. This makes STLF:

- **Adaptive** — weights follow signal quality per-sample, no hand-tuned gating.
- **Auditable** — every cue's contribution to the log-odds is inspectable.
- **Safe by default** — a high score backed by little reliable evidence is
  *downgraded* to `SUSPICIOUS` rather than trusted (`min_reliability_for_genuine`).

Full derivation and the "why not fixed weights / why not one big classifier"
discussion: [`docs/NOVEL_TECHNIQUE.md`](docs/NOVEL_TECHNIQUE.md).

---

## Quickstart

```bash
git clone https://github.com/aabhimittal/advanced-facial-recognition-system-with-anti-fraud-detection.git
cd advanced-facial-recognition-system-with-anti-fraud-detection
pip install -e .              # numpy + scipy only
python -m faceguard.cli demo  # synthetic live-vs-spoof, no camera needed
```

Example output:

```
=== LIVE face (genuine) ===
  verdict          : GENUINE
  liveness score   : 1.000
  fused confidence : 0.918
    - spectral  score=1.000  reliability=0.600
    - texture   score=0.852  reliability=1.000
    - rppg      score=0.985  reliability=0.200
    - motion    score=0.571  reliability=1.000

=== SPOOF face (replay/print) ===
  verdict          : FRAUD
  liveness score   : 0.066
  ...
```

### Use it in code

```python
from faceguard import FaceGuardPipeline
from faceguard.utils import synth_live_clip

pipe = FaceGuardPipeline()
clip = synth_live_clip()          # or your own (T, H, W, 3) array of RGB frames

pipe.enroll("alice", clip[0])     # enrol a reference face
result = pipe.identify_and_verify(clip)

print(result.summary())           # verdict=genuine live=0.99 conf=0.91 identity=alice
if result.is_trustworthy_match(): # True only if identity matched AND liveness passed
    print("Access granted to", result.identity)
```

The security property in one test: a **spoof of an enrolled user still matches
their identity but is rejected by liveness**, so `is_trustworthy_match()` is
`False` — see `tests/test_pipeline.py`.

### Real webcam / ArcFace (optional)

```bash
pip install -e '.[full]'          # opencv + insightface + onnxruntime
python -m faceguard.cli analyze path/to/video.mp4
```

With these installed, real Haar detection and ArcFace embeddings are used
automatically — the STLF core is unchanged.

---

## Architecture

```
frame(s) ──▶ FaceDetector ──▶ crop
                                │
        ┌───────────────┬──────┴──────┬────────────────┐
        ▼               ▼             ▼                ▼
   SpectralDetector TextureDetector RPPGDetector  MotionDetector
    (FFT 1/f)      (LBP + period.)  (POS pulse)   (non-rigid flow)
        │  (score, reliability) for each              │
        └───────────────┬─────────────────────────────┘
                        ▼
        Spectro-Temporal Liveness Fusion  (Bayesian log-odds pool)
                        ▼
             verdict ∈ {GENUINE, SUSPICIOUS, FRAUD}  +  confidence
                        │
                        ▼  (if GENUINE)
              FaceEmbedder + FaceMatcher ──▶ identity
```

Each box is one small, independently testable module under `src/faceguard/`.
See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## What's honest about this project

- The **algorithms are real** and the physics is sound (spectral fingerprints of
  displays, rPPG, LBP, non-rigid motion are all established PAD signals).
- The **synthetic generator** deliberately bakes in those physical differences so
  the pipeline is verifiable offline — it is a **teaching/CI tool, not a claim of
  field accuracy**. Numbers from `demo` are on synthetic data.
- The default **numpy embedder** is a simple gradient descriptor for demos; use
  the ArcFace adapter for real recognition.
- To measure real performance (APCER/BPCER/ACER), run against public PAD datasets
  — protocol and hooks in [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Repository layout

```
src/faceguard/
  liveness/    spectral · texture · rppg · motion detectors  (the physics)
  fusion/      stlf.py — the novel confidence-weighted Bayesian pool
  recognition/ embedder (numpy | ArcFace) + matcher
  detection/   face detector (centre-crop | OpenCV Haar)
  utils/       synthetic live/spoof generator (numpy/scipy only)
  pipeline.py  end-to-end orchestration
tests/         22 tests, run with just numpy + scipy + pytest
docs/          NOVEL_TECHNIQUE · ARCHITECTURE · EVALUATION
```

## Roadmap / good first issues

- [ ] rPPG: swap POS for CHROM and add a signal-quality index feeding reliability
- [ ] Learn `detector_weights` on a labelled set (logistic regression on log-odds)
- [ ] Add a DCT-based GAN-fingerprint head for deepfake-specific detection
- [ ] Challenge-response mode (blink / head-turn prompts) for `SUSPICIOUS` cases
- [ ] Real-dataset evaluation harness (Replay-Attack / CASIA-SURF / OULU-NPU)

## References

Key ideas build on: ArcFace (Deng et al. 2019) for embeddings; POS rPPG
(Wang et al. 2017); LBP face anti-spoofing (Määttä et al. 2011); frequency-domain
GAN-fingerprint detection (Frank et al. 2020). Full notes in the docs.

## License

MIT — see [LICENSE](LICENSE). Educational/research use; not certified for
production biometric access control without independent evaluation.
