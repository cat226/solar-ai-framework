# Training-duration ablation — `condition_yolo11n_4class_baseline_v1` (v2, 100ep) vs `condition_yolo11n_4class_200ep_v1` (v3, 200ep) — Phase 15B.1

**This is a controlled ablation, not a new baseline.** ONLY `epochs`
(100→200) and `patience` (20→40) changed. Dataset, architecture,
augmentation, optimizer strategy, and class taxonomy are byte-for-byte
identical to [[phase15b_four_class_condition_baseline]] — verified below,
not assumed. Purpose: determine whether v2's weak recall was caused by
under-training (fixable with more epochs) or a more fundamental
data/architecture limitation. **Neither checkpoint is promoted to
production.**

## 1. Config diff — v2 vs v3

| | v2 (`condition_yolo11n_4class_baseline_v1`) | v3 (`condition_yolo11n_4class_200ep_v1`) | changed? |
|---|---|---|---|
| epochs (budget) | 100 | 200 | **yes — the ablated variable** |
| patience | 20 | 40 | **yes — the ablated variable** |
| base model | yolo11n.pt (COCO) | yolo11n.pt (COCO) | no |
| dataset | `edithstark/solar-ai-ground-condition-4class-research-v1` | same | no |
| dataset content SHA-256 | `6637e6361343e7bb17d1703e705bffc0cb733a5428f48ce681fa28bc8826bf92` | same | no |
| split manifest SHA-256 | `600752137a4cbf8b24f506339b662ea0d0dbb67d1fc897f52fef83d6eb1d893e` | same | no |
| imgsz | 640 | 640 | no |
| batch | -1 (auto) | -1 (auto) | no |
| optimizer | auto | auto | no |
| cos_lr | true | true | no |
| seed | 42 | 42 | no |
| augmentation (12 params) | hsv/degrees/translate/scale/shear/perspective/flip/mosaic/mixup/copy_paste | identical values | no |
| Python | 3.12.13 | 3.12.13 | no |
| PyTorch | 2.7.1+cu126 | 2.7.1+cu126 | no |
| torchvision | 0.22.1+cu126 | 0.22.1+cu126 | no |
| Ultralytics | 8.4.138 | 8.4.138 | no |
| CUDA | 12.6 | 12.6 | no |
| GPU | Tesla P100-PCIE-16GB | Tesla P100-PCIE-16GB | no |

**No unexpected environment-driven changes.** The generated v3 notebook
asserted `DATASET_HASHES == V2_DATASET_HASHES` before training could
start (a hard fail-closed check, not a post-hoc comparison), and the
retrieved manifest confirms every non-ablated field — including the full
environment stack — matches v2 exactly. This is a clean, single-variable
ablation.

**Training duration**: v2 2,797.4 s (~46.6 min); v3 5,504.6 s (~91.7 min,
≈1.97×) — scaled almost exactly linearly with the doubled epoch budget, no
per-epoch slowdown.

## 2. Best epoch and training-curve interpretation

| | v2 | v3 |
|---|---|---|
| best epoch | 97 / 100 | 189 / 200 |
| patience triggered? | no (budget exhausted first) | no (budget exhausted first) |
| best val mAP50-95 | 0.238 (epoch 97) | 0.292 (epoch 189) |

Neither run's `patience` ever triggered — both trained to their full
epoch budget without an early stop. **Epoch 200 is not simply assumed
best**: `best_epoch=189` was Ultralytics' own best-fitness checkpoint
selection (independently verified by argmax of the `results.csv`
mAP50-95 column, which agrees), 11 epochs before the run ended — `best.pt`
already reflects the best validation performance, not the final epoch.

Full per-epoch trace (`results.csv`), key checkpoints:

| epoch | mAP50-95 (val) | mAP50 (val) | val_box_loss | train_box_loss |
|---|---|---|---|---|
| 100 | 0.238 | 0.443 | 1.901 | — |
| 120 | 0.261 | 0.476 | 1.843 | — |
| 140 | 0.275 | 0.491 | 1.841 | — |
| 150 | 0.284 | 0.503 | 1.819 | — |
| 160 | 0.288 | 0.505 | 1.821 | — |
| 170 | 0.289 | 0.509 | 1.820 | 1.228 |
| 180 | 0.289 | 0.508 | 1.814 | 1.213 |
| **189 (best)** | **0.292** | **0.511** | 1.810 | 1.210 |
| 200 | 0.289 | 0.509 | 1.814 | 1.149 |

**Interpretation — a real answer, not a hedge:**

1. **v2 was genuinely under-trained.** From epoch 100 → ~150, val
   mAP50-95 rose substantially (0.238 → 0.284, +19% relative), confirming
   v2's own observation that mAP was still climbing at its cutoff. This
   directly supports explanation **(A)**.
2. **But the improvement plateaus hard after ~epoch 150-160.** From
   epoch 160 → 200 — the entire back half of the extended budget —
   mAP50-95 only oscillates in a narrow 0.284–0.292 band (noise-level,
   not a trend). `val_box_loss` is flat (1.81–1.82) across the same
   range while `train_box_loss` keeps slowly falling (1.23 → 1.15) — a
   mild train/val divergence (early overfitting *onset*), though val
   metrics themselves never degrade, so this is not yet damaging
   overfitting, just diminishing returns. This is explanation **(B)**
   territory for the second half of the run.
3. **Combined verdict**: v2 was under-trained, and extending training
   recovered real, measurable gains — but only up to roughly epoch
   150-160. Continuing all the way to 200 (the full ablated budget)
   contributed almost nothing beyond that point. A future run with
   `epochs≈160, patience≈30-40` would likely reach the same result for
   much less compute. **Do not conclude "always train longer"** — the
   evidence supports "v2's specific 100-epoch cutoff was premature," not
   "more epochs are unboundedly beneficial."

## 3. Overall test-set metrics — v2 vs v3 vs delta

Standard reporting (conf floor 0.001, threshold-independent mAP), test
set touched exactly once for each checkpoint:

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| v2 | 0.5921 | 0.4405 | 0.4438 | 0.2361 |
| v3 | 0.5848 | 0.4607 | 0.4634 | 0.2546 |
| **Δ (v3 − v2)** | **−0.0073** | **+0.0202** | **+0.0196** | **+0.0185** |

Recall and both mAP metrics improved modestly (~2 points absolute);
precision is essentially flat (a small, likely noise-level decline).

## 4. Per-class test-set metrics — v2 vs v3 vs delta

At `conf=0.001` (default/threshold-independent, `test_default`):

| class | metric | v2 | v3 | Δ (v3−v2) |
|---|---|---|---|---|
| CLEAN | Precision | 0.6273 | 0.6138 | −0.0135 |
| CLEAN | Recall | 0.5184 | 0.5331 | +0.0147 |
| CLEAN | AP50 | 0.5475 | 0.5500 | +0.0025 |
| CLEAN | AP50-95 | 0.3475 | 0.3687 | **+0.0212** |
| DUSTY | Precision | 0.4670 | 0.5476 | **+0.0806** |
| DUSTY | Recall | 0.3244 | 0.3882 | **+0.0638** |
| DUSTY | AP50 | 0.3422 | 0.3933 | **+0.0511** |
| DUSTY | AP50-95 | 0.1503 | 0.1936 | **+0.0433** |
| BIRD_DROP | Precision | 0.5952 | 0.5923 | −0.0028 |
| BIRD_DROP | Recall | 0.3421 | 0.3574 | +0.0153 |
| BIRD_DROP | AP50 | 0.3601 | 0.3927 | **+0.0326** |
| BIRD_DROP | AP50-95 | 0.1478 | 0.1758 | **+0.0280** |
| PHYSICAL_DAMAGE | Precision | 0.6789 | 0.5854 | **−0.0936** |
| PHYSICAL_DAMAGE | Recall | 0.5769 | 0.5641 | −0.0128 |
| PHYSICAL_DAMAGE | AP50 | 0.5253 | 0.5174 | −0.0079 |
| PHYSICAL_DAMAGE | AP50-95 | 0.2988 | 0.2803 | −0.0185 |

**DUSTY improved the most** (the previously-worst-localized class, AP50-95
up ~29% relative) and **BIRD_DROP improved meaningfully** (AP50-95 up
~19% relative). **CLEAN improved slightly.** **PHYSICAL_DAMAGE — the
rarest class and the strongest performer in v2 — regressed**, most
notably in precision (−0.094). With only 78 test instances / 18 unique
families for this class, this could plausibly be sampling variance rather
than a genuine capability loss, but it must be reported as-is: more
training did not uniformly help every class.

## 5. Threshold sweep — v2 vs v3

Both swept conf ∈ {0.20 … 0.60} on **validation only**, test untouched
until the final single evaluation.

| conf | v2 P | v2 R | v2 F1 | v3 P | v3 R | v3 F1 |
|---|---|---|---|---|---|---|
| 0.20 | 0.551 | 0.467 | 0.505 | 0.611 | 0.492 | 0.545 |
| 0.25 | 0.598 | 0.434 | 0.503 | 0.603 | 0.497 | 0.545 |
| 0.30 | 0.638 | 0.409 | 0.499 | 0.635 | 0.480 | **0.547** |
| 0.35 | 0.690 | 0.385 | 0.494 | 0.673 | 0.453 | 0.541 |
| 0.40 | 0.721 | 0.356 | 0.477 | 0.705 | 0.437 | 0.540 |
| 0.45 | 0.756 | 0.336 | 0.465 | 0.739 | 0.413 | 0.530 |
| 0.50 | 0.780 | 0.313 | 0.447 | 0.766 | 0.391 | 0.518 |
| 0.55 | 0.803 | 0.285 | 0.421 | 0.794 | 0.369 | 0.504 |
| 0.60 | 0.833 | 0.261 | 0.394 | 0.821 | 0.344 | 0.485 |

**v2's F1 was strictly monotonically decreasing as conf rose** — best F1
at the lowest tested value (0.20), meaning the true optimum could lie
below the tested range (an explicit limitation flagged in v2's own
report).

**v3 is genuinely different: F1 is NOT monotonic.** It rises slightly
from 0.20→0.30 (0.545→0.547) before declining — a real, if very shallow,
interior peak. This is recorded programmatically in v3's manifest as
`f1_monotonic_toward_lowest_threshold: false`. **However**, the sweep
selection rule (smallest conf whose F1 is within 2% of the sweep's best)
still resolves to **0.20** for v3 too, because the curve is nearly flat
across 0.20-0.40 (all within 2% of the 0.547 peak) — so the *chosen*
operating threshold happens to be numerically identical between v2 and
v3 (0.20), but for a different reason: v2 picked the edge of a
monotonic decline, v3 picked the (flat) start of a real, shallow peak.
**Do not read "same threshold" as "same curve shape."**

## 6. Six real-world images — qualitative OOD reference only, v2 vs v3

`D:\solar_ai_training_scratch\Solar Real Images` — NOT used for
training, validation, or threshold tuning at any point. **No ground
truth exists; no accuracy is claimed or computable for either
checkpoint.**

| image | panel | v2 status/label/conf | v3 status/label/conf |
|---|---|---|---|
| …56 | P1 | predicted BIRD_DROP 0.729 | **uncertain** BIRD_DROP 0.406 |
| …57(1) | P1 | no_prediction | **predicted** CLEAN 0.471 |
| …57(1) | P2 | uncertain BIRD_DROP 0.390 | uncertain BIRD_DROP 0.410 |
| …57 | P1 | predicted CLEAN 0.616 | predicted CLEAN 0.887 |
| …57 | P2 | no_prediction | **predicted** CLEAN 0.843 |
| …58 | P1 | uncertain CLEAN 0.303 | **predicted** CLEAN 0.952 |
| …59(1) | P1 | predicted CLEAN 0.578 | predicted CLEAN 0.948 |
| …59(1) | P2 | uncertain CLEAN 0.382 | **predicted** CLEAN 0.930 |
| …59 | P1 | uncertain DUSTY 0.251 | uncertain CLEAN 0.302 (label changed) |
| …59 | P2 | predicted CLEAN 0.834 | predicted CLEAN 0.908 |

Totals: 10 panels (unchanged — panel detector untouched). v2: 4
predicted / 4 uncertain / 2 no_prediction. v3: **7 predicted / 3
uncertain / 0 no_prediction.**

**Honest reading, not a victory lap**: v3 is more confident and produces
fewer non-answers, and most of that shift is toward CLEAN with much
higher confidence on panels that were previously borderline. This is
*consistent with* (not proof of) the measured CLEAN/DUSTY/BIRD_DROP
recall gains in §4. But the single anecdote v2's report specifically
flagged as an encouraging (unverified) match — image …56, previously
described as "heavily marked with bird droppings," predicted BIRD_DROP
at 0.729 in v2 — **weakened under v3** (BIRD_DROP 0.406, now
`uncertain`, not `predicted`). Since there is no ground truth, this is
not evidence that v3 got *worse* at detecting bird droppings; it is
evidence that this specific single-image anecdote should never have been
treated as validation in the first place, and it is not repeated here as
one. `no_prediction` was never interpreted as CLEAN in either run.

## 7. Confusion / error analysis — v2 vs v3

Both normalized confusion matrices (test set, at each checkpoint's own
operating threshold 0.20):

**Zero confusion between the four active classes in both v2 and v3** — no
CLEAN↔DUSTY, no BIRD_DROP↔PHYSICAL_DAMAGE, etc., in either run. This
finding is unchanged and reinforced by the second run.

| | v2 recall (diagonal) | v3 recall (diagonal) | v2 background-miss-rate | v3 background-miss-rate |
|---|---|---|---|---|
| CLEAN | 0.52 | 0.57 | 0.35 | 0.26 |
| DUSTY | 0.36 | 0.43 | 0.32 | 0.35 |
| BIRD_DROP | 0.35 | 0.40 | 0.30 | 0.36 |
| PHYSICAL_DAMAGE | 0.61 | 0.70 | 0.03 | 0.03 |

(Figures from the plotted confusion matrix, IoU=0.5 single-threshold
matching at conf=0.20 — not directly the same methodology as the
multi-threshold mAP figures in §3/§4, so read this table as a separate,
consistent-within-itself comparison, not a re-derivation of those
numbers.) CLEAN's miss rate improved the most (0.35→0.26); DUSTY and
BIRD_DROP's background-miss rates did not improve on this particular
metric (both got very slightly worse, 0.32→0.35 and 0.30→0.36) even
though their AP50-95 improved in §4 — the two views are measuring
related but not identical things (one threshold-bound diagonal count vs.
an integral over the full precision-recall curve). All error remains
coverage/background-FP-driven, never cross-class confusion, in both
checkpoints.

## 8. Under-training — answered directly

**Was v2 under-trained? Yes, partially — and the ablation successfully
localizes exactly how much.** Extending training from 100→200 epochs
recovered a real, measurable ~+0.019 absolute mAP50-95 and ~+0.020
recall on the held-out test set, with DUSTY and BIRD_DROP (v2's weakest
classes) improving the most. But essentially all of that gain occurred
between epoch 100 and ~epoch 150-160; the remaining 40-50 epochs of the
200-epoch budget added no further measurable benefit and showed the
first signs of train/val divergence. **The correct framing is not "train
longer" but "v2's specific 100-epoch/patience-20 cutoff was too early by
roughly 50-60 epochs, and this recipe's ceiling — given this dataset and
architecture — is reached by around epoch 150-160, not epoch 100 and not
meaningfully improved by pushing to epoch 200."**

## 9. Decision framework verdict: **MODEST IMPROVEMENT**

- Not STRONG IMPROVEMENT: gains are small (single-digit percentage
  points), recall for 3 of 4 classes remains under 55%, and one class
  (PHYSICAL_DAMAGE) regressed.
- Not NO MATERIAL IMPROVEMENT: DUSTY and BIRD_DROP — the two classes v2
  itself flagged as weakest — improved by a real, consistent, non-noise
  margin (AP50-95 +19-29% relative) across precision, recall, AP50, and
  AP50-95 simultaneously.
- Not REGRESSION: aggregate test metrics improved; zero new cross-class
  confusion was introduced; only one of four classes (the smallest-
  sample one) got worse, and only on some metrics.

**Proceeding cautiously, per the framework**: v3 is a modest, honest
improvement over v2 — worth keeping as the better research checkpoint —
but it does not resolve v2's core weaknesses (DUSTY/BIRD_DROP recall
still below 40%, PHYSICAL_DAMAGE's small sample size still a variance
risk) and does not clear the bar for production consideration.
**Recommendation for next phase: stop spending compute on longer
training duration alone** — §2's plateau evidence shows this lever is
now exhausted for this recipe — and instead investigate the data/model
limitation directly (see §11).

## 10. Frozen-artifact verification

All artifacts declared frozen before this phase (now seven, with v2
newly added) were re-verified byte-identical both before and after this
work, via `tests/test_ground_pv_frozen_artifacts_integrity.py`:

| file | SHA-256 |
|---|---|
| `yolo_solar.pt` | `0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea` |
| `manifest.json` | `eabfec951724dff9975e57c9d3a290d272c61853a66b7c958fc3cfe58e907363` |
| `yolo_solar_ground_research_v1.pt` | `2b26334dc4023ba4c89a80e2350ddc9012358a48cdef713aee86c8e79a1bbb50` |
| `yolo_solar_ground_research_v2.pt` | `26e4d603de0718b9b20c6bf5c9761ef1b114b6954f280925ea6c08e469357aa3` |
| `yolo_solar_condition_research_v1.pt` (5-class) | `b077251a514424481d51b6a57ebcc30f4a607ea7559053cbb3ef7d0de88f9578` |
| `yolo_solar_condition_4class_research_v1.pt` (Phase 15A smoke) | `08994846e1087b6f485492e646f286ca9be8ea44ef7eb729c6e893fc31a8c7a3` |
| `yolo_solar_condition_4class_research_v2.pt` (Phase 15B, **newly frozen**) | `2dcf6122c18590f33f3b44ad88e054d409509274193977046ad2076d368de28e` |

## 11. Checkpoint

- Path: `weights/yolo_solar_condition_4class_research_v3.pt`
- SHA-256: `2e26b6afefffff79f43e75c80996f5fcd2aed435df56a33a1bdc5882d3a53c05`
- `last.pt` SHA-256: `7425afa3d3fbac5231092515e882768e29261ff1e732d78e12dc34c85f26e6f4`
- Verified via `scripts/verify_condition_4class_checkpoint.py` against the
  retrieved `training_manifest.json`: internally consistent.
- **Not production validated. Not promoted. Not wired into `models/`,
  `services/`, or the Streamlit app.** v2 remains untouched alongside it
  — both are kept as separate, independently reproducible research
  artifacts.

## 12. Recommendation for next phase

**Do not run another longer-duration training experiment on this exact
recipe** — the plateau in §2 shows that lever is exhausted (epoch
160→200 added nothing). If this research line continues, the
highest-leverage next steps are data/architecture-focused, not
duration-focused:

1. Investigate DUSTY/BIRD_DROP localization directly (still AP50-95
   ≈0.17-0.19) via false-positive/false-negative crop review — the same
   recommendation v2 made, now reinforced since more epochs alone did
   not close this gap to a materially different level.
2. Treat PHYSICAL_DAMAGE's regression as a variance flag, not a
   confirmed capability loss, given only 78 test instances — do not
   over-correct for it without more data.
3. Extend the threshold sweep below 0.20 for v3 specifically, now that
   §5 shows a genuine (if shallow) interior peak near 0.30 — a finer
   sweep near 0.20-0.35 could locate it more precisely.
4. Any future architecture or data change should be evaluated against
   **v3**, not v2, as the current-best research reference — but neither
   should be promoted or wired into the application without a
   genuinely independent, human-verified deployment-domain evaluation
   (the six-image set is not, and cannot become, that evaluation).
