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
| `fusion/stlf.py` | **Spectro-Temporal Liveness Fusion** | stdlib only |
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
  results   = { spectral(rep), texture(rep), rppg(clip), motion(clip) }
  score, confidence, verdict = STLF.fuse(results.values())
  -> PipelineResult

FaceGuardPipeline.identify_and_verify(frames)
  = analyze(frames)  +  matcher.identify(embedder.embed(crop(rep)))
```

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
