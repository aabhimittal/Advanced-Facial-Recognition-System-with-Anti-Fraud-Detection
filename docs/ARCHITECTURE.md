# Architecture

FaceGuard is deliberately split into small, single-responsibility modules so the
novel fusion logic stays isolated from the (swappable) detection/recognition
backends.

## Module map

| Module | Responsibility | Depends on |
|--------|----------------|-----------|
| `types.py` | `DetectorResult`, `FraudVerdict`, `PipelineResult` | stdlib only |
| `config.py` | all tunable thresholds & weights | stdlib only |
| `liveness/base.py` | grayscale / clip normalisation | numpy |
| `liveness/spectral.py` | FFT `1/f` fingerprint cue | numpy |
| `liveness/rppg.py` | POS pulse extraction + band-SNR | numpy, scipy |
| `liveness/texture.py` | LBP entropy + autocorrelation periodicity | numpy, scipy |
| `liveness/motion.py` | non-rigid micro-motion via aligned residual | numpy, scipy |
| `liveness/dct.py` | block-DCT GAN/deepfake fingerprint | numpy, scipy |
| `fusion/stlf.py` | **Spectro-Temporal Liveness Fusion** | stdlib only |
| `fusion/calibration.py` | learn weights/prior by logistic regression | numpy |
| `challenge/challenge.py` | active blink/turn/nod challenge-response | numpy, scipy |
| `recognition/embedder.py` | face embedding (numpy \| ArcFace adapter) | numpy (+ insightface) |
| `recognition/matcher.py` | enrol / identify gallery, cosine distance | numpy |
| `detection/detector.py` | face box (centre-crop \| OpenCV Haar) | numpy (+ opencv) |
| `pipeline.py` | wires detect → liveness → fuse → recognise | all of the above |
| `utils/synthetic.py` | live/spoof generator for demos & CI | numpy, scipy |

## The detector contract

Every liveness detector is a callable returning a `DetectorResult`:

```python
DetectorResult(name: str, score: float, reliability: float, detail: dict)
```

- `score ∈ [0,1]` — higher = more likely a genuine live face.
- `reliability ∈ [0,1]` — the detector's confidence that *this* measurement is
  trustworthy for *this* sample. Return `0.0` to abstain (e.g. temporal
  detectors on a single frame). This is the only coupling between a detector and
  the fusion stage, which is what keeps detectors independently testable and the
  fusion backend-agnostic.

Image-based cues (spectral, texture) receive a single representative frame (the
temporal median of the clip); temporal cues (rPPG, motion) receive the full
clip. A single still image is treated as a 1-frame clip, so temporal cues abstain
and STLF decides from the image cues alone.

## Data flow

```
FaceGuardPipeline.analyze(frames)
  clip      = as_clip(frames)                 # (T, H, W, C)
  rep       = median over time                # (H, W, 3)
  results   = { spectral(rep), texture(rep), rppg(clip), motion(clip), dct(rep) }
  score, confidence, verdict = STLF.fuse(results.values())
  -> PipelineResult

FaceGuardPipeline.identify_and_verify(frames)
  = analyze(frames)  +  matcher.identify(embedder.embed(crop(rep)))

FaceGuardPipeline.resolve_suspicious(result, response_clip, challenge)
  # only acts on SUSPICIOUS: passed challenge -> GENUINE, failed -> FRAUD
```

## Learned fusion weights

`fusion/calibration.py` turns a labelled PAD set into a calibrated
`FaceGuardConfig`. Because STLF is linear in log-odds, the features are exactly
`xᵢ = rᵢ·logit(sᵢ)` and a logistic regression's coefficients *are* the weights
(intercept → prior). Weights are constrained non-negative so a cue can only ever
add trust, never invert the decision.

## Challenge-response

`challenge/` escalates the ambiguous `SUSPICIOUS` band. `issue_challenge(nonce)`
picks an unpredictable action; `ChallengeVerifier` reconstructs the head-motion
trajectory (phase correlation vs frame 0) plus an illumination-normalised
eye-region signal and checks the *specific* requested action happened. A replay
of a different action — or of no action — is rejected, which is what makes it
resistant to pre-recorded attacks.

## Swapping backends

- **Detection**: replace `FaceDetector` with RetinaFace/MTCNN — must return
  `FaceBox` objects.
- **Recognition**: `FaceEmbedder(backend="arcface")` (auto-selected when
  `insightface` is installed) or any object exposing `.embed(crop) -> np.ndarray`.
- **Thresholds / weights**: everything lives in `FaceGuardConfig`; nothing is
  hard-coded in the fusion maths.

## Testing strategy

`tests/` runs on numpy + scipy + pytest alone:

- `test_detectors.py` — each cue separates live vs spoof and abstains on
  degraded input; scores stay bounded (incl. NaN input).
- `test_fusion.py` — abstention is a no-op, reliability scales influence, the
  low-confidence downgrade fires, conflicting evidence → `SUSPICIOUS`.
- `test_pipeline.py` — end-to-end verdicts, and the key security property that a
  spoof of an enrolled user matches identity but fails liveness.
- `test_dct.py` — the DCT head separates real from GAN-fingerprinted faces and is
  decisive on a deepfake clip the other cues would only mark `SUSPICIOUS`.
- `test_calibration.py` — learned weights are valid/non-negative and improve
  held-out accuracy over the defaults.
- `test_challenge.py` — matching responses pass, non/mismatched responses fail
  (zero false-accepts across all pairs), and the pipeline resolves `SUSPICIOUS`.
