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
  - *Spectral* — global frequency structure (`1/f` vs periodic display peaks)
  - *rPPG* — a physiological pulse that only living skin has
  - *Micro-texture* — local, stochastic, aperiodic skin detail
  - *Micro-motion* — non-rigid 3-D deformation a flat spoof cannot produce
  - *DCT deepfake* — the block-DCT upsampling fingerprint of generated imagery
  A print kills the pulse; a screen adds a periodic spectrum and texture; a
  waved photo has only rigid motion; a deepfake leaves a Nyquist checkerboard in
  the block-DCT even when it *looks* live. No single attack satisfies all five.

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

## 5. Learned weights & active escalation (implemented)

Two extensions ship in the library and make the contract concrete:

- **Learned `wᵢ` (`fusion/calibration.py`).** Because STLF is linear in log-odds,
  the per-cue features are exactly `xᵢ = rᵢ·logit(sᵢ)` and a logistic regression's
  coefficients *are* the weights (its intercept is `logit(p₀)`). Weights are
  constrained non-negative, preserving "a cue is a unit of trust". On the bundled
  synthetic set this lifts held-out accuracy from ~0.63 to 1.0.
- **Challenge-response (`challenge/`).** The `SUSPICIOUS` band is not a dead end:
  the system issues a random blink / head-turn / nod prompt and verifies the
  *specific* action from the response clip's motion trajectory and eye signal.
  Unpredictability is the security property — a pre-recorded replay of a
  different action (or no action) is rejected, with zero false-accepts across all
  mismatched action pairs in the test suite.

## 6. How to push it further

- **Calibrate `sᵢ`** per detector (Platt / isotonic) so `logit(sᵢ)` is a true
  log-likelihood-ratio; then the pool is exactly a naive-Bayes combination.
- **Reliability from signal quality indices** (rPPG SNR, crop resolution, blur)
  rather than the current monotone proxies.
- **Landmark-based challenge verification** — replace the region heuristics with a
  face mesh for finer actions (gaze direction, mouth shape).


---

## Extensions since the original write-up

The fusion mathematics above is unchanged; three refinements make it hold up
outside a benchmark.

### 1. One-sided evidence, honestly encoded

Artefact detectors (spectral, DCT, banding) can only ever *observe* an artefact.
A clean reading is not evidence of life — the cleanest possible image is exactly
what a sufficiently good attack produces. Encoding those cues on a symmetric
`[0, 1]` scale silently treats "no moiré" as proof of a heartbeat.

So one-sided cues are capped at `MAX_ONE_SIDED_LIVE = 0.65`:

```
score = floor + (0.65 − floor) · exp(−evidence)
```

Presence of an artefact still drives the score to ~0 and dominates the pool;
absence contributes a mild `logit(0.65) ≈ +0.62`. Certainty of liveness has to be
*earned* by two-sided cues that measure a positive physiological signal — pulse,
parallax, subsurface scattering, sensor noise.

### 2. No cue may veto the pool

`logit(1 − 10⁻⁶) ≈ 13.8`. A single detector pinned at its rail could therefore
overrule every other cue combined — a de-facto veto no individual heuristic has
earned, and a routine failure mode when a threshold-based cue saturates. Each
per-detector term is clipped to `±max_detector_logit` (default 4.0), which still
lets one confident cue move the posterior from 0.5 to ~0.98 on its own, but never
past a consensus of its peers.

### 3. Sequential extension: the log-odds pool *is* the SPRT statistic

Because STLF already produces its posterior in log-odds, and log-odds relative to
a 0.5 prior is exactly a log-likelihood ratio, per-window results accumulate into
Wald's sequential probability ratio test with no additional modelling:

```
Λ ← Λ + min(1, R_window) · logit(p_window)
accept if Λ ≥ log((1−β)/α)      reject if Λ ≤ log(β/(1−α))
```

Reliability scaling carries over from the single-shot case with the same meaning:
a poor window *slows* the decision rather than corrupting it. This is what turns
a fixed-window classifier into a streaming decision rule — and its cost model is
the honest one, since a blatant attack is rejected in one window while a marginal
subject is simply looked at for longer.

The caveat, stated plainly: Wald's error bounds assume independent evidence, and
successive windows of one capture are correlated (the attacker is holding the
same photo the whole time). Treat `far`/`frr` as operating knobs calibrated on
your own data, not as guarantees.
