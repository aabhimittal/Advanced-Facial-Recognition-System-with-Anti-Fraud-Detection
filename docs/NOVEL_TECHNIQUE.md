# Spectro-Temporal Liveness Fusion (STLF)

This document explains the novel contribution of FaceGuard in depth: what it is,
why it is different from prior fusion approaches, and the exact maths.

## 1. The problem with single-cue and fixed-weight liveness

Face **presentation-attack detection** (PAD) systems try to separate a genuine
live face from a spoof (print, replay on a screen, 3-D mask, deepfake). Two
common design patterns each have a structural weakness:

1. **Single strong cue** (e.g. one CNN trained on texture). Attackers adapt to
   whatever the model keys on; a cue that works on prints often fails on a
   high-quality display, and vice-versa.
2. **Fixed-weight fusion** of several cues. Better, but the weights are constant
   regardless of whether a cue's signal is even *measurable* on the current
   sample. rPPG on a 6-frame clip is noise, yet a fixed-weight pool still trusts
   it at full strength. Concatenating features into one classifier has the same
   problem — it cannot say "ignore this input, it's unreliable *right now*".

## 2. The STLF premise

> A fraudster rarely defeats **orthogonal** physical cues **simultaneously**, and
> a defender should weight each cue by **how trustworthy it is on this exact
> sample**.

Two ideas, combined:

- **Orthogonality.** We pick cues whose failure modes are independent:
  - *Spectral* — global frequency structure (`1/f` vs periodic display/GAN peaks)
  - *rPPG* — a physiological pulse that only living skin has
  - *Micro-texture* — local, stochastic, aperiodic skin detail
  - *Micro-motion* — non-rigid 3-D deformation a flat spoof cannot produce
  A print kills the pulse; a screen adds a periodic spectrum and texture; a
  waved photo has only rigid motion. No single attack satisfies all four.

- **Per-sample reliability.** Each detector returns `(score sᵢ, reliability rᵢ)`.
  Reliability is a *self-assessment of signal quality*: rPPG lowers it when there
  are too few frames; spectral/texture lower it on tiny crops. This is the piece
  fixed-weight fusion lacks.

## 3. The maths

We fuse in **log-odds (logit) space** — the natural space for combining
independent probabilistic evidence.

Let `p₀` be the prior probability that a face is genuine, `wᵢ` a per-detector
base trust (config, optionally learned), and `(sᵢ, rᵢ)` the detector outputs.

```
logit(x) = ln( x / (1−x) )

L      = logit(p₀) + Σᵢ (wᵢ · rᵢ) · logit(sᵢ)      (fused log-odds)
p_live = sigmoid(L)                                  (fused probability)
R      = 1 − exp( −Σᵢ wᵢ · rᵢ )                      (fused confidence in [0,1])
```

This is a **log-linear (logarithmic) opinion pool** with the pooling weights set
to the per-sample effective reliabilities `βᵢ = wᵢ·rᵢ`.

### Why this specific form has the properties we want

- **Abstention is free.** `logit(0.5) = 0`, so a detector that says "I don't
  know" (`sᵢ = 0.5`) adds nothing. A detector with `rᵢ = 0` also adds nothing —
  it drops out of the pool entirely. The system **degrades gracefully** to the
  evidence that is actually reliable.
- **Reliability scales influence smoothly.** Doubling a cue's reliability doubles
  its pull on the log-odds — no discontinuous gating thresholds to tune.
- **Confidence is separate from score.** `R` grows only as *reliable* evidence
  accumulates. A single flukey high score does not produce high confidence.
- **Auditable.** Each term `(wᵢ·rᵢ)·logit(sᵢ)` is the signed contribution of one
  cue to the decision. `fuse()` returns these contributions for inspection.

### Decision rule (with a safety downgrade)

```
if p_live ≤ fraud_threshold:                       FRAUD
elif p_live ≥ genuine_threshold:
        if R ≥ min_reliability_for_genuine:        GENUINE
        else:                                      SUSPICIOUS   # not enough reliable evidence
else:                                              SUSPICIOUS   # send to challenge-response
```

The `SUSPICIOUS` band is a feature: rather than force a binary decision on weak
or conflicting evidence, the system asks for a second sample or a challenge
(blink / head-turn), which is exactly how robust real-world systems behave.

## 4. Relationship to prior work

- **ArcFace** (Deng et al., 2019) — identity embeddings; orthogonal to liveness.
- **rPPG / POS** (Wang et al., 2017) — pulse extraction robust to illumination;
  we use POS and add a band-SNR → reliability mapping.
- **LBP anti-spoofing** (Määttä et al., 2011) — micro-texture; we extend it with
  an autocorrelation *periodicity* penalty so it also flags display grids.
- **GAN fingerprints in frequency** (Frank et al., 2020) — motivates the
  spectral cue's sensitivity to periodic upsampling artefacts.

**What's new here** is not any single cue but the *fusion contract*: detectors
must report calibrated reliability, and fusion is a confidence-weighted Bayesian
pool with an explicit low-confidence downgrade. That combination — adaptive,
auditable, abstention-aware — is the contribution.

## 5. How to push it further

- **Learn `wᵢ`** by logistic regression of the label on the per-cue `logit(sᵢ)`
  features over a labelled PAD set — STLF is already linear in log-odds, so the
  learned coefficients *are* the weights.
- **Calibrate `sᵢ`** per detector (Platt / isotonic) so `logit(sᵢ)` is a true
  log-likelihood-ratio; then the pool is exactly a naive-Bayes combination.
- **Reliability from signal quality indices** (rPPG SNR, crop resolution, blur)
  rather than the current monotone proxies.
