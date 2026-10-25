# Augmentation ablation — mosaic+copy_paste (v4_aug) vs v2 baseline — Phase 15E-A

**Hypothesis under test**: following [[phase15d_a_resolution_ablation]]
(imgsz 640→960, STOPPED — 0/4 classes materially improved small-object
recall), does a targeted small-object-oriented **augmentation** change —
`mosaic` 0.5→1.0 and `copy_paste` 0.0→0.5, the two augmentations
Ultralytics' own documentation associates specifically with improved
small/occluded-object detection — recover the small-object recall
weakness Phase 15C documented?

**Result: no, not materially.** Using the same mechanical decision rule
introduced in Phase 15D-A (≥0.02 average tiny+small-bin recall
improvement, in ≥3 of 4 classes), **0 of 4 classes qualify**. A softer
read of the evidence (tiny bin only, see §4) shows a small, consistent
positive shift in all four classes — genuinely more encouraging than the
resolution experiment's result — but it does not survive averaging with
the small bin, which was flat-to-negative for three of four classes.
PHYSICAL_DAMAGE regressed on every official metric and showed the
sharpest small-object decline of the four classes.
**Recommendation: stop this augmentation direction; reassess the
dataset/labeling strategy or loss formulation** (§9), per the task's own
decision rule.

## 1. Config diff — v2 (baseline) vs v4_aug (candidate)

| | v2 (`condition_yolo11n_4class_baseline_v1`) | v4_aug (`condition_yolo11n_4class_aug_v1`) | changed? |
|---|---|---|---|
| mosaic | 0.5 | **1.0** | **yes — half of the ablated intervention** |
| copy_paste | 0.0 | **0.5** | **yes — half of the ablated intervention** |
| imgsz | 640 | 640 | no |
| epochs (budget) | 100 | 100 | no |
| patience | 20 | 20 | no |
| base model | yolo11n.pt (COCO), trained from scratch | yolo11n.pt (COCO), trained from scratch — **not** fine-tuned from v2/v3/v3_960 | no |
| dataset | `edithstark/solar-ai-ground-condition-4class-research-v1` | same (hash-asserted before training) | no |
| seed / optimizer / cos_lr | 42 / auto / true | 42 / auto / true | no |
| hsv_h/s/v, degrees, translate, scale, shear, perspective, flipud, fliplr, mixup | v2 values | **identical to v2** | no |
| Python / torch / torchvision / Ultralytics / CUDA / GPU | 3.12.13 / 2.7.1+cu126 / 0.22.1+cu126 / 8.4.138 / 12.6 / Tesla P100-PCIE-16GB | identical | no |
| loss / architecture / labels / taxonomy / confidence threshold / NMS | unchanged | unchanged | no |
| pseudo-labels used | no | no | no |

`environment_match: True` and `dataset_hashes_match: True`, both computed
programmatically by `scripts/compare_condition_resolution_experiment.py`
from the two retrieved manifests — **no unexpected environment-driven
changes occurred, and this is a genuinely single-lever ablation** (two
parameters under one conceptual "targeted small-object augmentation"
intervention, everything else byte-identical to v2).

## 2. Pre-training verification (per task requirement)

All five checks passed before the Kaggle launch was approved:

1. Split-manifest SHA-256: `600752137a4cbf8b24f506339b662ea0d0dbb67d1fc897f52fef83d6eb1d893e` — matches.
2. Row/split counts: 2,411 total = 1,686 train / 483 valid / 242 test — matches.
3. Family-split isolation: 1,170 families, **0 crossing a split boundary**.
4. v2 baseline checkpoint SHA-256: `2dcf6122c18590f33f3b44ad88e054d409509274193977046ad2076d368de28e` — matches.
5. All frozen artifacts (9 at the time, now 10 with v3_960 newly frozen — see §10) verified byte-identical via `tests/test_ground_pv_frozen_artifacts_integrity.py`.

## 3. Overall metrics — official Ultralytics (`test_default`, conf floor 0.001)

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| v2 (baseline aug) | 0.5921 | 0.4405 | 0.4438 | 0.2361 |
| v4_aug (mosaic=1.0, copy_paste=0.5) | 0.5592 | 0.4418 | 0.4427 | 0.2246 |
| **Δ (aug − v2)** | **−0.0329** | **+0.0013** | **−0.0011** | **−0.0115** |

Recall is essentially unchanged (+0.001), mAP50 essentially unchanged
(−0.001), while precision and mAP50-95 both declined. **A flat-to-mildly-negative
aggregate result** — not the outcome a successful small-object
intervention would be expected to produce (which should move recall
specifically, since the hypothesis is about missed detections, not false
positives).

## 4. Per-class metrics — official Ultralytics (`test_default`)

| class | metric | v2 | v4_aug | Δ (aug−v2) |
|---|---|---|---|---|
| CLEAN | Precision | 0.6273 | 0.5646 | −0.0626 |
| CLEAN | Recall | 0.5184 | 0.5276 | +0.0092 |
| CLEAN | AP50 | 0.5475 | 0.5316 | −0.0159 |
| CLEAN | AP50-95 | 0.3475 | 0.3349 | −0.0125 |
| DUSTY | Precision | 0.4670 | 0.4996 | **+0.0326** |
| DUSTY | Recall | 0.3244 | 0.3494 | **+0.0250** |
| DUSTY | AP50 | 0.3422 | 0.3654 | **+0.0231** |
| DUSTY | AP50-95 | 0.1503 | 0.1541 | +0.0038 |
| BIRD_DROP | Precision | 0.5952 | 0.5356 | −0.0595 |
| BIRD_DROP | Recall | 0.3421 | 0.3505 | +0.0083 |
| BIRD_DROP | AP50 | 0.3601 | 0.3630 | +0.0029 |
| BIRD_DROP | AP50-95 | 0.1478 | 0.1393 | −0.0085 |
| PHYSICAL_DAMAGE | Precision | 0.6789 | 0.6369 | **−0.0421** |
| PHYSICAL_DAMAGE | Recall | 0.5769 | 0.5397 | **−0.0372** |
| PHYSICAL_DAMAGE | AP50 | 0.5253 | 0.5106 | −0.0146 |
| PHYSICAL_DAMAGE | AP50-95 | 0.2988 | 0.2701 | **−0.0287** |

**DUSTY is the one class improving on all four official metrics** —
modest but uniform. **PHYSICAL_DAMAGE regresses on all four official
metrics** — the clearest, most consistent negative signal in this
table, and (per §6 below) the class this report flags as substantially
regressed, sample-size caveats notwithstanding. CLEAN and BIRD_DROP are
mixed (recall up slightly, everything else down).

## 5. Small/medium/large object recall — same size-bucket methodology as Phase 15C/15D

Size bins are Q1/Q2/Q3 quartiles of each class's own test-set GT box
area-fraction distribution (identical, data-driven method to Phase
15C/15D-A — never hand-picked), recomputed by running
`scripts/analyze_condition_4class_errors.py` against v4_aug (imgsz=640,
same resolution as v2 — no cross-resolution confound here):

| class | bin | v2 recall | v4_aug recall | Δ |
|---|---|---|---|---|
| CLEAN | tiny | 0.222 | 0.265 | **+0.043** |
| CLEAN | small | 0.568 | 0.540 | −0.028 |
| CLEAN | medium | 0.549 | 0.609 | **+0.060** |
| CLEAN | large | 0.807 | 0.838 | +0.031 |
| DUSTY | tiny | 0.101 | 0.115 | +0.014 |
| DUSTY | small | 0.292 | 0.303 | +0.011 |
| DUSTY | medium | 0.460 | 0.405 | −0.055 |
| DUSTY | large | 0.561 | 0.565 | +0.004 |
| BIRD_DROP | tiny | 0.117 | 0.150 | **+0.033** |
| BIRD_DROP | small | 0.360 | 0.308 | −0.052 |
| BIRD_DROP | medium | 0.523 | 0.545 | +0.022 |
| BIRD_DROP | large | 0.547 | 0.480 | −0.067 |
| PHYSICAL_DAMAGE | tiny | 0.111 | 0.150 | +0.039 (small N) |
| PHYSICAL_DAMAGE | small | 0.667 | 0.444 | **−0.223 (small N, ~9 instances)** |
| PHYSICAL_DAMAGE | medium | 0.750 | 0.750 | 0.000 |
| PHYSICAL_DAMAGE | large | 0.737 | 0.700 | −0.037 |

**A genuinely notable pattern**: the **tiny bin improved in ALL FOUR
classes** (+0.043, +0.014, +0.033, +0.039) — a more encouraging and more
consistent signal than Phase 15D-A's resolution experiment produced
(which improved only CLEAN's tiny bin). **But this does not survive
averaging with the small bin**, which declined for CLEAN, BIRD_DROP, and
(sharply, on a small sample) PHYSICAL_DAMAGE, and only barely rose for
DUSTY. Per the mechanical rule (§6), none of the four classes clears the
material-improvement bar once both bins are combined.

## 6. Mechanical small-object-recall verdict (reused, unmodified rule from Phase 15D-A)

`small_object_recall_improved()` — average of the tiny+small bin deltas
per class, ≥0.02 required to count as "improved", ≥3 of 4 classes
required for "material" (the SAME threshold, applied without
modification, as Phase 15D-A used for the resolution ablation — no
post-hoc threshold tuning):

| class | avg(tiny,small) Δ | improved? |
|---|---|---|
| CLEAN | +0.0070 | no |
| DUSTY | +0.0120 | no |
| BIRD_DROP | −0.0094 | no |
| PHYSICAL_DAMAGE | −0.0917 | no |

**0 of 4 classes show a material small-object recall improvement.**
Every class's avg(tiny,small) delta is small in magnitude; only
PHYSICAL_DAMAGE's is large, and it is negative.

**Cross-check note** (same expected methodological gap documented in
Phase 15C §3 and Phase 15D-A §4): this phase's own matching-based
overall recall (conf≥0.20/IoU≥0.5) is v2=0.4222, v4_aug=0.4182 — also
essentially flat, consistent with the official-metric flatness in §3.
Matching-based overall precision is slightly higher for aug (0.4954 vs
0.4809) — both figures reported, not just the more favorable one. The
confusion matrix for v4_aug (visually inspected) confirms **zero
cross-class confusion**, unchanged from every prior condition-model
phase; per-class diagonal recall proxies are CLEAN 0.54 (v2: 0.52,
slightly better), DUSTY 0.35 (v2: 0.36, flat), BIRD_DROP 0.34 (v2: 0.35,
flat), PHYSICAL_DAMAGE 0.61 (v2: 0.61, exactly flat) — all consistent
with the modest, mixed story in this report.

## 7. Did any class regress substantially?

**Yes — PHYSICAL_DAMAGE.** It is the only class to regress on **all
four** official metrics simultaneously (precision −0.042, recall
−0.037, AP50 −0.015, AP50-95 −0.029, §4) and it has the single largest
negative small-object delta of any class/bin combination in this report
(avg(tiny,small) = −0.092, §6, driven mainly by a small-bin drop from
0.667→0.444). **Caveat, reported honestly rather than omitted**:
PHYSICAL_DAMAGE has only 78 test instances, and its small bin has
roughly 9 instances — a single-digit-instance swing can move this ratio
substantially. This is flagged as the clearest regression signal in the
experiment, not dismissed, but also not treated as proof of a robust
capability loss given the sample size — the same caution this project
applied to PHYSICAL_DAMAGE's v2→v3 regression in
[[phase15b1_four_class_condition_duration_ablation]].

## 8. Training duration, best epoch

| | v2 (baseline aug) | v4_aug |
|---|---|---|
| train_seconds | 2,797.4 s (~46.6 min) | 2,832.1 s (~47.2 min) |
| ratio | 1.00× | 1.01× |
| best_epoch | 97 / 100 | 100 / 100 |
| patience (20) triggered? | no | no |

Duration is essentially unchanged (as expected — augmentation recipe
changes don't materially affect per-epoch compute the way resolution
does). `best_epoch=100/100` for v4_aug means the very last epoch was
still the best on validation — training had not visibly plateaued,
though `patience=20` never triggered either, matching v2's own behavior
(best_epoch 97/100).

## 9. Machine-readable summary

`scripts/compare_condition_resolution_experiment.py` (the same script
Phase 15D-A introduced, reused here unmodified — its schema is generic
across any two-checkpoint comparison) produced the full comparison
programmatically from the two training manifests and two error-analysis
summaries, committed at
`datasets/ground_pv_research/condition_4class_v1/reports/condition_yolo11n_4class_aug_v1/augmentation_comparison.json`.
The per-checkpoint error-analysis summary for v4_aug (size bins, FP/FN
counts, confidence distributions, family/data-quality flags) is
committed at
`datasets/ground_pv_research/condition_4class_v1/reports/condition_error_analysis_aug/error_analysis_summary.json`.
The bulky per-instance records and annotated gallery remain gitignored
under `research_runs/ground_pv/condition_error_analysis_aug/`
(regenerable via `scripts/analyze_condition_4class_errors.py --weights
weights/yolo_solar_condition_4class_research_v4_aug.pt --imgsz 640`).
`tests/test_phase15e_augmentation_experiment.py` locks in both the
report's schema and its actual reported finding (0/4 material
improvement, PHYSICAL_DAMAGE the largest regression) as a regression
guard against a future silent change to either the artifacts or the
decision-rule constants.

## 10. Checkpoint

- Path: `weights/yolo_solar_condition_4class_research_v4_aug.pt`
- SHA-256: `eb4d6151a53ded76049f7469946e3016384b48ba5bdbcbda2845efca7fdf618b`
- `last.pt` SHA-256: `08c633e276ab29e7be3ef32f534012c2fb3458187c8e09cde90e9d2c8d620940`
- Verified via `scripts/verify_condition_4class_checkpoint.py`: internally
  consistent with its retrieved `training_manifest.json`.
- **Not production validated. Not promoted. Not wired into `models/`,
  `services/`, or the Streamlit app.** v2, v3 (200-epoch/imgsz640), and
  v3_960 (imgsz960, now also frozen — see §11) remain untouched alongside
  it.

## 11. Frozen-artifact verification

All artifacts declared frozen before this phase (now nine, with v3_960
newly added) were re-verified byte-identical before and after this
work, via `tests/test_ground_pv_frozen_artifacts_integrity.py`:

| file | SHA-256 |
|---|---|
| `yolo_solar.pt` | `0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea` |
| `manifest.json` | `eabfec951724dff9975e57c9d3a290d272c61853a66b7c958fc3cfe58e907363` |
| `yolo_solar_ground_research_v1.pt` | `2b26334dc4023ba4c89a80e2350ddc9012358a48cdef713aee86c8e79a1bbb50` |
| `yolo_solar_ground_research_v2.pt` | `26e4d603de0718b9b20c6bf5c9761ef1b114b6954f280925ea6c08e469357aa3` |
| `yolo_solar_condition_research_v1.pt` (5-class) | `b077251a514424481d51b6a57ebcc30f4a607ea7559053cbb3ef7d0de88f9578` |
| `yolo_solar_condition_4class_research_v1.pt` (Phase 15A smoke) | `08994846e1087b6f485492e646f286ca9be8ea44ef7eb729c6e893fc31a8c7a3` |
| `yolo_solar_condition_4class_research_v2.pt` (Phase 15B, 640/100ep) | `2dcf6122c18590f33f3b44ad88e054d409509274193977046ad2076d368de28e` |
| `yolo_solar_condition_4class_research_v3.pt` (Phase 15B.1, 640/200ep) | `2e26b6afefffff79f43e75c80996f5fcd2aed435df56a33a1bdc5882d3a53c05` |
| `yolo_solar_condition_4class_research_v3_960.pt` (Phase 15D-A, 960/100ep, **newly frozen**) | `eca9332a320b41bb695e5f392f532a489ff8e2df7519cf613997ff5241514ace` |

Dataset content (`splits_4class_manifest.csv` SHA-256
`600752137a4cbf8b24f506339b662ea0d0dbb67d1fc897f52fef83d6eb1d893e`) and
family-grouped split isolation (1,170 families, 0 crossing a split
boundary, 2,411 rows = 1,686 train + 483 valid + 242 test) were
re-verified unchanged before training was launched.

## 12. Recommendation

**Stop this augmentation direction.** Per the task's explicit decision
rule ("if it does not materially improve, STOP this augmentation
direction and reassess the dataset/labeling strategy or loss
formulation"): §6 shows 0 of 4 classes with a material small-object
recall improvement under the exact same mechanical threshold Phase
15D-A used, and §7 identifies a real (if small-sample-caveated)
regression in PHYSICAL_DAMAGE across all four official metrics. The
tiny-bin-only signal in §5 (a small, consistent positive shift in all
four classes) is the most encouraging pattern seen across the two
intervention experiments run so far (resolution, then augmentation) —
worth remembering as a directional clue — but it is not large enough,
nor does it survive combination with the small bin, to justify
recommending this specific augmentation recipe for further investment.
**Both the resolution lever (Phase 15D-A) and this specific augmentation
lever have now been tried and stopped without a material fix to the
documented small-object weakness.** The next experiment should target
either the dataset/labeling strategy directly (e.g. Phase 15C's
flagged dense-tiny-annotation-cluster pattern, or first-party collection
of genuinely small, well-separated defect instances) or a loss-function
change (e.g. scale-aware loss reweighting), not a further tweak to
mosaic/copy_paste probabilities. Per this phase's own instructions, no
further training experiment was launched automatically after this one.
