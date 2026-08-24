# Industrial edge cases: what breaks a face-liveness system in the field

Benchmarks measure a system on curated captures of cooperative subjects. A
deployment is measured by everything else. This document is the catalogue of
"everything else" — each entry names a real failure mode, the FaceGuard
behaviour that addresses it, and the test that pins it down.

The organising principle throughout:

> **A system failure must never be charged to the user as an accusation.**

Degraded capture and degraded software both look, to a liveness detector, exactly
like a presentation attack: low high-frequency energy, no pulse, no motion. A
system without an explicit "I cannot tell" state does not fail randomly under
stress — it fails *biased towards calling innocent people frauds*. That is why
`FraudVerdict.INDETERMINATE` exists and why so much of the code below is about
knowing when to stay silent.

---

## 1. Capture-side edge cases

| Situation | Why it breaks naive systems | FaceGuard behaviour | Test |
|---|---|---|---|
| Dark lobby at 7am, 20-lux stairwell | Low light destroys micro-texture → reads as "print" | `exposure` factor fails → `INDETERMINATE` + "adjust lighting" | `test_quality.py::test_degraded_capture_is_rejected_with_the_right_reason` |
| Backlit doorway, blown-out cheek | Clipped pixels carry no information at all | `exposure` counts pixels pinned at either rail | same |
| Greasy or defocused lens | Blur removes exactly the high frequencies the spectral cue reads | `focus` factor via **band-ratio**, not absolute sharpness | same |
| Smooth, evenly-lit subject | Absolute sharpness measures call them "blurry" | Band-ratio focus is content-independent | `test_new_detectors.py` (mask clips pass the gate) |
| Subject far from camera | A 20-px crop cannot support a spectrum | `resolution` factor; detectors also self-abstain | `test_quality.py::test_score_is_the_weakest_factor_not_the_average` |
| Frozen / stalled decoder | A repeated frame looks *perfectly* stable to temporal cues | `stability` factor detects zero inter-frame variation | `test_quality.py` |
| Mono / IR camera | Chroma cues are impossible, not fraudulent | Advisory factor by default; rPPG and subsurface abstain | `test_quality.py::test_mono_and_single_frame_captures_are_still_usable` |
| Single enrolment photo | Temporal cues impossible | Advisory; `require_clip=True` makes it strict when policy demands | same |

Quality is scored as the **weakest** factor, never the mean: a pristine capture
of a 12-pixel face is unusable, and averaging would hide that.

## 2. Input-format edge cases

Real capture stacks deliver things no dataset contains.

| Input | Naive result | FaceGuard behaviour | Test |
|---|---|---|---|
| 10/12/16-bit machine-vision frames | `/255` saturates everything to white → bogus "spoof" | Full-scale detected and scaled correctly | `test_edge_cases.py::test_high_bit_depth_cameras_are_scaled_not_saturated` |
| NaN/Inf from a dropped buffer | Propagates silently into every score | Scrubbed at the boundary | `test_nan_and_inf_frames_are_scrubbed` |
| Zero-width ROI, empty clip, ragged frame list | `IndexError` deep inside a detector | Typed `InvalidFrameError`, surfaced as `INDETERMINATE` | `test_degenerate_input_raises_a_typed_error` |
| BGRA, grayscale clips, stray batch dimension | Shape errors or silent misreads | Normalised in one place (`liveness/base.py`) | `test_every_plausible_layout_normalises_to_a_clip` |
| All-black / all-white / constant frames | Divide-by-zero, NaN scores | Detectors abstain; verdict still well-formed | `test_constant_and_extreme_frames_never_crash` |

## 3. Availability and latency edge cases

| Situation | FaceGuard behaviour | Test |
|---|---|---|
| One detector throws (corrupt model file, bad build) | Crash → **abstention**. Fusion is unaffected in direction, only in confidence. | `test_a_crashing_detector_degrades_confidence_not_correctness` |
| Most of the ensemble is broken | `min_detector_availability` → `INDETERMINATE`. A decision from a crippled ensemble is one no auditor could defend. | `test_mass_detector_outage_produces_indeterminate_not_a_coin_flip` |
| Queue forming at a turnstile | `time_budget_ms`: not-yet-run legs abstain rather than blow the SLA — and the skip is *recorded* | `test_latency_budget_makes_late_detectors_abstain` |
| One saturated heuristic pins at 1.0 | `max_detector_logit` caps its contribution: any cue may argue, none may dictate | `test_no_single_detector_can_veto_the_ensemble` |
| A bad sample inside a batch | Isolated to that sample | `test_batch_analysis_isolates_a_bad_sample` |

Every degradation is *named* in `PipelineResult.degraded`, so monitoring sees a
degraded mode instead of an unexplained accuracy drift.

## 4. Attack-side edge cases

| Attack | Why the classic cue set misses it | The cue that catches it |
|---|---|---|
| **Injection** — virtual camera, tampered SDK, rooted HAL | There is no presentation at all: no print, no screen, no moiré | `sensor` — photon-transfer law; synthetic frames carry no intensity-dependent temporal noise |
| **Screen replay** with glare/moiré suppression | Looks like a print to texture and spectral cues | `banding` — rolling-shutter beat against the display refresh |
| **Tilted / advanced photo** (6-DoF hand motion) | Richer motion than a straight wave | `parallax` — the field is still explained by one homography |
| **3-D silicone mask** | Real depth, real camera, real motion | `subsurface` — red light penetrates skin, not silicone |
| **Rhythmic motion posing as a pulse** | A shaken photo produces an in-band spectral peak | rPPG **spatial phase coherence**: one heart drives the whole face in phase, motion does not |
| **Replay of a genuine challenge response** | An unpredictable prompt is not enough on its own | Challenges are HMAC-signed, TTL-bounded and single-use |

## 5. Human edge cases

Not every failure is an attack, and treating it as one is its own failure.

| Situation | Correct behaviour | Where |
|---|---|---|
| User looks away and misses the prompt | `SUSPICIOUS` + retry, **not** `FRAUD` | `pipeline.enforce_challenge`, `test_challenge.py::test_a_genuine_user_who_misses_the_prompt_is_asked_again_not_accused` |
| Ambiguous evidence after several windows | `EXHAUSTED` → `SUSPICIOUS`; undecided is not an accusation | `test_sequential.py::test_ambiguous_evidence_keeps_sampling_then_exhausts` |
| Camera never yields a usable frame | Windows are consumed, accumulator untouched, loop terminates | `test_indeterminate_windows_contribute_nothing_but_still_consume_budget` |
| A blatant spoof in a 12-window budget | Rejected in one window — don't hold the queue | `test_obvious_attack_is_rejected_in_far_fewer_windows_than_the_budget` |

## 6. Data-governance edge cases

| Situation | Behaviour | Test |
|---|---|---|
| Gallery database is exfiltrated | Only cancellable bit-templates leak; rotate the key and reissue | `test_protection.py::test_a_different_key_invalidates_every_stored_template` |
| The same person enrolled in two systems | Per-subject salts make the templates unlinkable | `test_a_different_salt_makes_the_same_face_unlinkable` |
| Right-to-erasure request | `ProtectedMatcher.revoke` actually forgets | `test_revoke_actually_forgets_the_subject` |
| Disputed decision, months later | Hash-chained audit record; tampering localises to an index | `test_audit.py::test_editing_a_past_record_is_detected_at_its_index` |
| Audit log itself becomes a biometric database | It stores decisions and salted digests — never images, embeddings or names | `test_the_log_stores_no_biometric_data` |

---

## What is still not covered

Stated plainly, because a threat list that claims completeness is worse than
useless:

- **Adversarial perturbations** crafted against these specific detectors. The
  ensemble raises the bar (an attack must fool orthogonal physics simultaneously)
  but nothing here is certified robust.
- **A twin, a sibling, or a very good makeup impersonation.** That is a
  *recognition* problem, not a liveness one; liveness will happily certify that a
  live human is present.
- **Coercion.** A user forced to authenticate passes every check in this library,
  because they are genuinely alive and genuinely themselves. Duress needs
  behavioural signals and out-of-band controls.
- **Real-world error rates.** All numbers in this repository come from the
  synthetic bench, which bakes in the very physics the detectors look for. It
  proves the code implements the intended physics; it does not predict APCER or
  BPCER on real attacks. See [`EVALUATION.md`](EVALUATION.md).
