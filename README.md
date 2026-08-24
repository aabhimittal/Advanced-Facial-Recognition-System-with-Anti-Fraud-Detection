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
eight cues that fail *independently* and fuses them — but with a twist most
published systems lack.

| Cue | Physical signal it exploits | Spoofs it catches |
|-----|------------------------------|-------------------|
| **Spectral** | Natural `1/f` image spectrum | screens, prints (periodic HF peaks) |
| **rPPG pulse** | Heartbeat colour change in skin, *phase-coherent across the face* | prints, masks, still images (no pulse); rhythmic motion posing as one |
| **Micro-texture** | Stochastic, aperiodic skin detail | display grids & halftone (periodic), matte prints (smooth) |
| **Micro-motion** | Non-rigid 3-D facial deformation | a waved flat photo (rigid motion only) |
| **DCT deepfake** | Block-DCT upsampling fingerprint | GAN/diffusion deepfakes (Nyquist checkerboard) |
| **Parallax** 🆕 | Motion field that only *depth* can produce | any planar attack — print, phone, tablet — under any hand motion |
| **Sensor noise** 🆕 | Photon-transfer law: `var = g·signal` | **injection attacks** — a virtual camera or tampered SDK, where no presentation cue exists at all |
| **Display banding** 🆕 | Rolling-shutter beat against a screen's refresh | screen replay, separating it from print |
| **Subsurface scattering** 🆕 | Red light penetrates skin deeper than green | **3-D silicone masks**, which pass every geometric and sensor test |

The last four exist because the first four share a blind spot: they all assume
the attacker must hold *something* up to a real camera. Parallax turns the
attacker's own hand motion into a geometric proof of flatness; the sensor leg
notices when no camera was involved at all; banding separates a screen from
paper; and subsurface scattering asks the one question a perfectly-shaped mask
cannot answer — *is this material skin?*

For the ambiguous middle (`SUSPICIOUS`), FaceGuard escalates to **active
challenge-response** — a randomly chosen blink / head-turn / nod prompt that a
pre-recorded replay cannot satisfy. Challenges are HMAC-signed, time-limited and
single-use, so a captured genuine response cannot be replayed either.

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
- **Un-vetoable** — each cue's contribution is capped at `max_detector_logit`, so
  one saturated heuristic cannot overrule four honest ones.
- **Asymmetric where the physics is asymmetric** — artefact detectors (spectral,
  DCT, banding) can only ever *see* an artefact, so a clean reading tops out at
  0.65 rather than 1.0: the absence of moiré is not proof of life, because a good
  attack leaves none either.

### Deciding responsibly, not just accurately

Detection accuracy is the part everyone benchmarks. These are the parts that
decide whether a deployment survives contact with reality:

| Concern | What FaceGuard does |
|---|---|
| Bad capture (dark lobby, greasy lens, frozen feed) | A **capture-quality gate** answers `INDETERMINATE` — "please retry" — never `FRAUD`. Degraded capture looks exactly like a print attack, so without this, systems fail *biased towards accusing innocent users*. |
| One detector crashes in production | Fault-isolated: a crash becomes an *abstention*, which the fusion absorbs mathematically. Confidence drops; correctness does not. |
| Half the ensemble is broken | `min_detector_availability` refuses to issue a verdict at all rather than quietly degrade to a coin-flip that still looks authoritative. |
| A queue at the turnstile | `time_budget_ms` makes late detectors abstain instead of blowing the latency SLA. |
| One threshold for a screen unlock *and* a wire transfer | Four **risk tiers** (`faceguard.policy`), each a documented operating point, selected per transaction. |
| Fixed 3-second capture for every user | **Sequential (SPRT) decisioning**: fix the error rates, let the sample size vary. Blatant spoofs are rejected in one window; ambiguous cases get more looks. |
| The gallery leaks | **Cancellable templates** (BioHashing): store key-dependent bit strings, rotate the key after a breach. You cannot reissue somebody's face. |
| "Prove the system wasn't tricked" | **Hash-chained audit log** — editing, deleting or reordering any past decision is detectable, and no biometric data is stored in it. |

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

Other commands:

```bash
python -m faceguard.cli attacks   # score every synthetic attack class
python -m faceguard.cli policy    # show what each risk tier trades away
python -m faceguard.cli stream    # sequential (SPRT) decisioning, window by window
```

`faceguard attacks` output (synthetic bench, 90-frame clips):

```
attack class                               verdict          live   conf   expected
----------------------------------------------------------------------------------
live face                                  genuine         1.000  0.996   genuine
print / replay                             fraud           0.027  0.995   fraud
screen replay (rolling-shutter banding)    fraud           0.000  0.998   fraud
tilted photo (6-DoF hand motion)           fraud           0.001  0.996   fraud
deepfake video                             fraud           0.000  0.995   fraud
injected stream (virtual camera)           fraud           0.000  0.996   fraud
3-D silicone mask                          suspicious      0.589  0.992   not-genuine

7/7 attack classes handled as expected
```

The mask row is the interesting one: it is genuinely 3-D and genuinely captured,
so it defeats every geometric and sensor cue and lands in `SUSPICIOUS` — which is
the *correct* answer for two seconds of evidence. Give it a five-second clip, or
escalate it to a challenge, and it resolves to `FRAUD`. A system that returned a
confident verdict there would be guessing.

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

### Learn the fusion weights from a labelled PAD set

STLF is linear in log-odds, so learning the weights is just a logistic
regression on the reliability-scaled per-cue contributions:

```python
from faceguard.fusion import fit_fusion_weights

# samples[k] = PipelineResult.detectors for example k; labels[k] = 1 genuine / 0 spoof
samples = [pipe.analyze(clip).detectors for clip in clips]
config = fit_fusion_weights(samples, labels)   # -> calibrated FaceGuardConfig
tuned = FaceGuardPipeline(config=config)        # weights + prior now data-driven
```

On the bundled synthetic set this lifts held-out accuracy from ~0.63 to 1.0 and
sensibly upweights the most discriminative cues (texture, rPPG).

### Escalate SUSPICIOUS cases with challenge-response

```python
from faceguard import ChallengeType

result = pipe.analyze(clip)
if result.verdict.value == "suspicious":
    challenge = pipe.issue_challenge(nonce=session_nonce)   # random, unpredictable
    print(challenge.instruction)                            # "Please turn your head to the left."
    response_clip = capture_from_webcam()                   # user reacts
    result = pipe.resolve_suspicious(result, response_clip, challenge)
    # passed -> GENUINE, failed/replayed -> FRAUD
```

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
frame(s) ──▶ CaptureQualityGate ──▶ unusable? ──▶ INDETERMINATE ("retry", never fraud)
                   │ usable
                   ▼
             FaceDetector ──▶ crop
                   │
   ┌─────────┬─────────┬─────────┬────┴────┬──────────┬─────────┬────────────┐
   ▼         ▼         ▼         ▼         ▼          ▼         ▼            ▼
Spectral  Texture    RPPG     Motion   DCT-fake   Parallax   Sensor      Subsurface
(FFT 1/f)(LBP+per.)(POS+coh.)(non-rig)(blockDCT) (geometry)(photon-xfer)(skin optics)
   │   (score, reliability) each · crash or timeout ⇒ abstain, never a wrong vote
   └─────────┬──────────────────────────────────────────────────────────────┘
             ▼
   Spectro-Temporal Liveness Fusion  (Bayesian log-odds pool, per-cue cap)
   weights learnable via logistic regression on a labelled PAD set
             ▼
   verdict ∈ {GENUINE, SUSPICIOUS, FRAUD, INDETERMINATE} + confidence
      │              │                        │
      │              │ SUSPICIOUS ──▶ signed, single-use challenge ──▶ GENUINE|FRAUD
      │              ▼
      │        SequentialVerifier (SPRT) ──▶ accept / reject / keep watching
      ▼ (if GENUINE)
 FaceEmbedder ──▶ FaceMatcher | ProtectedMatcher (cancellable templates) ──▶ identity
      │
      └──▶ AuditLog (hash-chained, no biometric data)
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
- The **SPRT error targets** (`far`, `frr`) are operating knobs, not guarantees:
  Wald's bounds assume independent evidence, and consecutive windows of one
  capture are correlated (the attacker holds the same photo the whole time).
  Calibrate them on your own data.
- **Cancellable templates are not encryption.** They are lossy and revocable, and
  an attacker holding both the key and a template can still approximate the
  embedding's direction. Keep the key in an HSM/KMS, separate from the store.
- To measure real performance (APCER/BPCER/ACER), run against public PAD datasets
  — protocol and hooks in [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Repository layout

```
src/faceguard/
  liveness/    spectral · texture · rppg · motion · dct · parallax · sensor ·
               banding · subsurface            (the physics, one file per cue)
  fusion/      stlf.py — the novel Bayesian pool · calibration.py — learned weights
  quality.py   capture-quality gate: when NOT to decide
  policy.py    risk-tiered operating points (convenience → critical)
  sequential.py  Wald SPRT for streams and multi-attempt captures
  audit.py     hash-chained, privacy-preserving decision log
  challenge/   signed, expiring, single-use blink/turn/nod challenges
  recognition/ embedder (numpy | ArcFace) + matcher + protection.py (cancellable)
  detection/   face detector (centre-crop | OpenCV Haar)
  utils/       synthetic generators: live · print · screen replay · tilted photo ·
               deepfake · injected stream · 3-D mask · challenge responses
  pipeline.py  end-to-end orchestration with fault isolation and a latency budget
tests/         178 tests, run with just numpy + scipy + pytest
docs/          NOVEL_TECHNIQUE · ARCHITECTURE · EVALUATION
```

## Roadmap / good first issues

- [x] Learn `detector_weights` on a labelled set (logistic regression on log-odds)
- [x] Add a DCT-based GAN-fingerprint head for deepfake-specific detection
- [x] Challenge-response mode (blink / head-turn prompts) for `SUSPICIOUS` cases
- [x] Capture-quality gate so bad capture returns "retry", not "fraud"
- [x] Injection-attack detection (photon-transfer / sensor-noise head)
- [x] Planar-attack geometry (depth-parallax head) and 3-D mask material cue
- [x] Sequential (SPRT) decisioning, risk tiers, cancellable templates, audit log
- [ ] rPPG: swap POS for CHROM and add a signal-quality index feeding reliability
- [ ] Landmark-based challenge verification (replace region heuristics with mesh)
- [ ] Real-dataset evaluation harness (Replay-Attack / CASIA-SURF / OULU-NPU)

## References

Key ideas build on: ArcFace (Deng et al. 2019) for embeddings; POS rPPG
(Wang et al. 2017); LBP face anti-spoofing (Määttä et al. 2011); frequency-domain
GAN-fingerprint detection (Frank et al. 2020). Full notes in the docs.

## License

MIT — see [LICENSE](LICENSE). Educational/research use; not certified for
production biometric access control without independent evaluation.
