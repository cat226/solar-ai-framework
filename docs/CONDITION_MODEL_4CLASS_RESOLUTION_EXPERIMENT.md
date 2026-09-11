# Resolution ablation — imgsz 640 (v2) vs imgsz 960 (v3_960) — Phase 15D-A

**Hypothesis under test**: does increasing training/inference resolution
from 640→960 recover the small-object recall weakness Phase 15C
documented (recall degrading monotonically with GT-box size in all four
active classes, `docs/CONDITION_MODEL_4CLASS_ERROR_ANALYSIS.md`)?

**Result: no.** Small-object recall did not materially improve for any
of the four active classes at the same 100-epoch budget. Aggregate
official metrics are mixed (recall/mAP50 up marginally, precision/mAP50-95
down). Training cost more than doubled (2.36×) for this outcome.
**Recommendation: stop the resolution intervention; reassess
augmentation/loss/data strategy** (see §9).

## 1. Config diff — v2 (baseline) vs v3_960 (candidate)

| | v2 (`condition_yolo11n_4class_baseline_v1`) | v3_960 (`condition_yolo11n_4class_960_v1`) | changed? |
|---|---|---|---|
| imgsz | 640 | 960 | **yes — the ablated variable** |
| epochs (budget) | 100 | 100 | no |
| patience | 20 | 20 | no |
| base model | yolo11n.pt (COCO), trained from scratch | yolo11n.pt (COCO), trained from scratch — **not** fine-tuned from v2/v3 | no |
| dataset | `edithstark/solar-ai-ground-condition-4class-research-v1` | same (hash-asserted before training) | no |
| seed / optimizer / cos_lr | 42 / auto / true | 42 / auto / true | no |
| augmentation (12 params) | identical values | identical values | no |
| Python / torch / torchvision / Ultralytics / CUDA / GPU | 3.12.13 / 2.7.1+cu126 / 0.22.1+cu126 / 8.4.138 / 12.6 / Tesla P100-PCIE-16GB | identical | no |

`environment_match: True` and `dataset_hashes_match: True`, both computed
programmatically by `scripts/compare_condition_resolution_experiment.py`
from the two retrieved manifests — **no unexpected environment-driven
changes occurred.**

## 2. Overall metrics — official Ultralytics (`test_default`, conf floor 0.001)

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| v2 (640, 100ep) | 0.5921 | 0.4405 | 0.4438 | 0.2361 |
| v3_960 (960, 100ep) | 0.5616 | 0.4516 | 0.4509 | 0.2159 |
| **Δ (960 − 640)** | **−0.0305** | **+0.0111** | **+0.0072** | **−0.0201** |

Mixed, small-magnitude result: recall and mAP50 rose marginally;
precision and mAP50-95 fell. **Not a clean win or a clean loss on
aggregate metrics alone** — this is exactly why the decision rule
requires looking at small-object recall specifically rather than
declaring success from overall mAP (§9).

## 3. Per-class metrics — official Ultralytics (`test_default`)

| class | metric | v2 (640) | v3_960 (960) | Δ (960−640) |
|---|---|---|---|---|
| CLEAN | Precision | 0.6273 | 0.5298 | **−0.0975** |
| CLEAN | Recall | 0.5184 | 0.5515 | +0.0331 |
| CLEAN | AP50 | 0.5475 | 0.5191 | −0.0285 |
| CLEAN | AP50-95 | 0.3475 | 0.3215 | −0.0259 |
| DUSTY | Precision | 0.4670 | 0.4576 | −0.0094 |
| DUSTY | Recall | 0.3244 | 0.2990 | −0.0255 |
| DUSTY | AP50 | 0.3422 | 0.3219 | −0.0204 |
| DUSTY | AP50-95 | 0.1503 | 0.1294 | −0.0209 |
| BIRD_DROP | Precision | 0.5952 | 0.5300 | −0.0651 |
| BIRD_DROP | Recall | 0.3421 | 0.3533 | +0.0111 |
| BIRD_DROP | AP50 | 0.3601 | 0.3582 | −0.0019 |
| BIRD_DROP | AP50-95 | 0.1478 | 0.1446 | −0.0032 |
| PHYSICAL_DAMAGE | Precision | 0.6789 | 0.7291 | **+0.0502** |
| PHYSICAL_DAMAGE | Recall | 0.5769 | 0.6026 | +0.0256 |
| PHYSICAL_DAMAGE | AP50 | 0.5253 | 0.6046 | **+0.0793** |
| PHYSICAL_DAMAGE | AP50-95 | 0.2988 | 0.2682 | −0.0306 |

**DUSTY is uniformly worse on all four metrics.** **PHYSICAL_DAMAGE is
the clearest beneficiary** (precision, recall, and AP50 all up
meaningfully; AP50-95 down). CLEAN and BIRD_DROP are mixed (recall up,
everything else down). No class shows a uniform, unambiguous
improvement across all four metrics.

## 4. Small/medium/large object recall — same size-bucket methodology as Phase 15C

Size bins are Q1/Q2/Q3 quartiles of each class's own test-set GT box
area-fraction distribution (data-driven, never hand-picked — identical
method and thresholds-per-class as `condition_error_analysis.py` used in
Phase 15C), recomputed independently for v2 and v3_960 by running
`scripts/analyze_condition_4class_errors.py` against EACH checkpoint at
its OWN training resolution (v2 at imgsz=640, v3_960 at imgsz=960 — a
mismatched-resolution evaluation would confound this comparison, so
this was never done):

| class | bin | v2 (640) recall | v3_960 (960) recall | Δ |
|---|---|---|---|---|
| CLEAN | tiny | 0.222 | 0.316 | **+0.094** |
| CLEAN | small | 0.568 | 0.514 | −0.054 |
| CLEAN | medium | 0.549 | 0.515 | −0.034 |
| CLEAN | large | 0.807 | 0.819 | +0.012 |
| DUSTY | tiny | 0.101 | 0.062 | **−0.039** |
| DUSTY | small | 0.292 | 0.225 | −0.067 |
| DUSTY | medium | 0.460 | 0.389 | −0.071 |
| DUSTY | large | 0.561 | 0.496 | −0.065 |
| BIRD_DROP | tiny | 0.117 | 0.114 | −0.003 |
| BIRD_DROP | small | 0.360 | 0.335 | −0.025 |
| BIRD_DROP | medium | 0.523 | 0.491 | −0.032 |
| BIRD_DROP | large | 0.547 | 0.486 | −0.061 |
| PHYSICAL_DAMAGE | tiny | 0.111 | 0.111 | 0.000 |
| PHYSICAL_DAMAGE | small | 0.667 | 0.579 | −0.088 |
| PHYSICAL_DAMAGE | medium | 0.750 | 0.636 | −0.114 |
| PHYSICAL_DAMAGE | large | 0.737 | 0.947 | +0.210 (small N, ≤19 instances) |

**Only CLEAN's tiny bin shows a real improvement (+0.094)** — and even
CLEAN's own small/medium bins got worse at the same time, so it is not a
clean win even within that one class. **DUSTY's tiny-bin recall got
worse, not better (−0.039)**, and DUSTY is worse across every single
bin. BIRD_DROP is flat-to-worse across every bin. PHYSICAL_DAMAGE's tiny
bin is unchanged (0.111 both); its large-bin jump is a small-sample
artifact (well under 20 instances).

**Mechanical verdict** (`small_object_recall_improved`, averaged over the
tiny+small bins per class, ≥0.02 absolute improvement required to count,
≥3 of 4 classes required for "material"):

| class | avg(tiny,small) Δ | improved? |
|---|---|---|
| CLEAN | +0.020 | no (just below the 0.02 threshold once rounded; driven entirely by tiny, offset by small) |
| DUSTY | −0.053 | no |
| BIRD_DROP | −0.014 | no |
| PHYSICAL_DAMAGE | −0.044 | no |

**0 of 4 classes show a material small-object recall improvement.**

**Cross-check note**: this phase's own matching-based overall recall
(conf≥0.20/IoU≥0.5, the same methodology as Phase 15C, computed
independently for both checkpoints) is v2=0.4222, v3_960=0.3879 —
**lower** for 960, even though the OFFICIAL Ultralytics `test_default`
recall in §2 is marginally HIGHER for 960 (0.4516 vs 0.4405). This is
the same expected, already-documented methodological gap as Phase 15C
§3 (official P/R report each class's best-F1 point on the full curve,
not a literal count at one fixed confidence) — both figures are reported
here rather than only the more favorable one. The confusion matrix for
v3_960 (visually inspected) confirms the same pattern at its own
diagonal: DUSTY 0.36→0.31, BIRD_DROP 0.35→0.32 (both worse), CLEAN
0.52→0.54 and PHYSICAL_DAMAGE 0.61→0.64 (both better) — consistent with
every other measurement in this report. **Zero cross-class confusion in
either checkpoint** (confirmed visually in both confusion matrices) —
this ablation does not change that finding.

## 5. Did the size-recall relationship improve? **No.**

The core, monotonic size-recall pattern Phase 15C documented (recall
rising from tiny→large within every class) is **still present in
v3_960**, essentially unchanged in shape, and for DUSTY specifically the
absolute recall at every single size bin is now lower than v2's. Higher
resolution did not flatten or reverse the pattern for the two
highest-volume classes (DUSTY, BIRD_DROP) that Phase 15C flagged as the
primary targets — it left the *shape* of the problem intact while
shifting the *level* down for those two classes.

## 6. Did overall performance improve or regress?

**Marginal, mixed — neither a clear improvement nor a clean regression
on official aggregate metrics** (recall +0.011, mAP50 +0.007, precision
−0.031, mAP50-95 −0.020, §2). Per this project's `docs/CONDITION_MODEL_4CLASS_TRAINING_COMPARISON.md`
decision framework, a change this small and mixed on its own would sit
near "NO MATERIAL IMPROVEMENT" territory even before considering the
primary hypothesis — and the primary hypothesis (§4/§5) is a clear
non-improvement.

## 7. Training duration, best epoch, final metrics

| | v2 (640) | v3_960 (960) |
|---|---|---|
| train_seconds | 2,797.4 s (~46.6 min) | 6,606.9 s (~110.1 min) |
| ratio | 1.00× | **2.36×** |
| best_epoch | 97 / 100 | 88 / 100 |
| patience (20) triggered? | no | no |

The 2.36× duration ratio is close to the theoretical (960/640)²≈2.25×
per-epoch compute scaling for a fixed epoch budget — internally
consistent, not an anomaly. (A live Kaggle-status polling log taken
during this run showed an apparent ~29-minute wall-clock window with one
anomalous ~10-minute gap between two status checks; that external
polling signal is treated as unreliable — likely a transient status-API
delay on one call — in favor of the in-kernel, code-measured
`train_seconds`, which is corroborated by the near-theoretical scaling
ratio above.)

## 8. Machine-readable summary

`scripts/compare_condition_resolution_experiment.py` produces the full
comparison programmatically (overall delta, per-class delta, size-bin
delta, mechanical small-object verdict) from the two training manifests
and two error-analysis summaries — committed at
`datasets/ground_pv_research/condition_4class_v1/reports/condition_yolo11n_4class_960_v1/resolution_comparison.json`.
Per-checkpoint error-analysis summaries (size bins, FP/FN counts,
confidence distributions, family/data-quality flags — same schema Phase
15C introduced) are committed at
`datasets/ground_pv_research/condition_4class_v1/reports/condition_error_analysis_v2/error_analysis_summary.json`
(newly generated for v2 in this phase, for a fair same-epoch-budget
comparison Phase 15C did not need) and
`.../condition_error_analysis_960/error_analysis_summary.json`. The
bulky per-instance records and annotated galleries for both remain
gitignored under `research_runs/ground_pv/` (regenerable via
`scripts/analyze_condition_4class_errors.py --weights ... --imgsz ...`).

## 9. Checkpoint

- Path: `weights/yolo_solar_condition_4class_research_v3_960.pt`
- SHA-256: `eca9332a320b41bb695e5f392f532a489ff8e2df7519cf613997ff5241514ace`
- `last.pt` SHA-256: `c059911dd9c6a89a54f317e75606604e2533ad91761a63731bf18354a98446be`
- Verified via `scripts/verify_condition_4class_checkpoint.py`: internally
  consistent with its retrieved `training_manifest.json`.
- **Not production validated. Not promoted. Not wired into `models/`,
  `services/`, or the Streamlit app.** v2 and v3 (200-epoch/imgsz640,
  now also frozen — see §10) remain untouched alongside it.

## 10. Frozen-artifact verification

All artifacts declared frozen before this phase (now eight, with v3
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
| `yolo_solar_condition_4class_research_v3.pt` (Phase 15B.1, 640/200ep, **newly frozen**) | `2e26b6afefffff79f43e75c80996f5fcd2aed435df56a33a1bdc5882d3a53c05` |

Dataset content (`splits_4class_manifest.csv` SHA-256
`600752137a4cbf8b24f506339b662ea0d0dbb67d1fc897f52fef83d6eb1d893e`) and
family-grouped split isolation (1,170 families, 0 crossing a split
boundary, 2,411 rows = 1,686 train + 483 valid + 242 test) were verified
unchanged before training was launched.

## 11. Recommendation

**Stop the resolution intervention.** Per the task's decision rule
("if 960 does not materially improve small-object recall, recommend
stopping the resolution intervention and reassessing
augmentation/loss/data strategy"): §4/§5 show 0 of 4 classes with a
material small-object recall improvement, and the two highest-volume
classes (DUSTY, BIRD_DROP) — Phase 15C's primary targets — are flat to
worse across every size bin, not just the smallest one. The 2.36×
training-cost increase bought, at best, a mixed and small aggregate
metric shift (§2/§6) and a real but narrow benefit to PHYSICAL_DAMAGE
(§3) that plausibly reflects that class's own small sample size (78 test
instances) rather than a genuine resolution effect. **Resolution alone
is not the lever that fixes this model's documented small-object
weakness** — the next experiment should target augmentation or loss
strategy (Phase 15C's original recommendation C), or a fresh
data-quality pass (the dense-annotation-cluster pattern Phase 15C
flagged), not a further resolution increase. Per this phase's own
instructions, no further training experiment was launched automatically
after this one.
