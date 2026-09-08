# Ground panel detector v2 — failure-targeted retraining (Phase 11)

**Recommendation: C — v2 improves some in-distribution test metrics but introduces
regressions. It does NOT improve any of the targeted real-world failure modes and
it collapses on the six owner photos. Keep v1 as the ground panel detector; do
not adopt v2.**

> RESEARCH ONLY. No production artifact was modified. `v1` and `weights/yolo_solar.pt`
> are unchanged. Real-world results are QUALITATIVE ONLY — there is no ground truth
> for the six owner photos, so no accuracy / precision / recall / F1 / mAP /
> confusion matrix is computed for them.

---

## 1. Objective

Phase 10 concluded that the bottleneck in the two-stage ground-PV pipeline is
first-stage panel localisation: v1 (`weights/yolo_solar_ground_research_v1.pt`)
missed the large foreground panel on image 4, emitted group boxes on the wide
array in image 6, and fired `cracked` false positives on roofing membrane
(images 1, 3). v2 retrains the **same** panel detector (classes `panel / cracked
/ dusty`) with a scale / domain-robustness augmentation intervention to target
those failures.

## 2. v1 baseline

`weights/yolo_solar_ground_research_v1.pt`, SHA
`2b26334dc4023ba4c89a80e2350ddc9012358a48cdef713aee86c8e79a1bbb50`. Test
(held-out leakage-clean, 649 images): **P 0.876 · R 0.801 · mAP50 0.865 ·
mAP50-95 0.677**. Per class: panel P .947 / R .925 / AP50 .963 / AP50-95 .843;
cracked P .809 / R .653 / AP50 .749 / AP50-95 .530; dusty P .871 / R .826 /
AP50 .882 / AP50-95 .660.

## 3. Training-data failure analysis

`scripts/analyze_ground_pv_dataset_scale.py` →
`ground_dataset_scale_analysis.json`. Dataset `datasets/ground_pv_research/yolo/`
(6,492 images, 6,319 panel boxes):

| finding | value |
|---|---|
| **image dimensions** | **100 % are 640×640** (Roboflow stretch-resized export) |
| owner-photo dimensions | 573×1280 (aspect 0.45) and 1280×573 (aspect 2.23) — **nothing like the training data** |
| panel box area | <25 % 52.3 % · 25–50 % 21.5 % · 50–75 % 15.7 % · **>75 % 10.6 %** (671 boxes) |
| boundary-touching panel boxes | 30.7 % |
| "image-4-like" (a panel > 55 % of frame, ≤ 2 panels) | 1,337 images (20.6 %) — but a large share are `panel = 0.5 0.5 1.0 1.0` (the box **is** the whole frame — a degenerate target), and **none in a portrait aspect** |
| "image-6-like" (≥ 4 panels, none > 40 %) | **114 images (1.8 %)** — and on inspection these are close-ups with many small (often noisy) `panel` boxes, **not** distant oblique array views |
| panels-per-image | 0 → 1,265 · 1 → 4,750 · 2 → 282 · ≥4 → 114 |

**The dataset does not contain examples resembling the owner photos.** There are
no portrait-aspect frames, no genuine distant multi-panel array shots, and the
"large foreground panel" cases are square, stock-photo, often full-frame boxes.
No annotations were fabricated, no owner photos were pseudo-labelled, and no
external images were added — this limitation is reported, not patched.

## 4. v2 configuration

Base `yolo11n.pt` (COCO), the **exact same** Kaggle dataset
`edithstark/solar-ai-ground-pv-research-v1` and leakage-clean split
(train 4,546 / valid 1,297 / test 649). Notebook `kaggle/ground_pv_v2_train.ipynb`
(regen `scripts/make_ground_v2_notebook.py`), kernel
`edithstark/ground-yolo11n-v2` v1 on a Tesla P100-16GB.

**Single intervention family — scale / domain robustness:**

| arg | v1 | v2 | rationale |
|---|---|---|---|
| `scale` | 0.3 | **0.9** | RandomResize object-scale jitter [0.7,1.3] → [0.1,1.9] |
| `mosaic` | 0.5 | **1.0** | always 4-image mosaic → shrinks objects + synthesises multi-panel scenes |
| `close_mosaic` | default | **10** | mosaic off for the last 10 epochs → clean real-scale convergence |
| `multi_scale` | off | **on** | per-batch imgsz jitter ±50 % |

**Unchanged from v1:** architecture (YOLO11n), COCO init, `optimizer=auto`,
`cos_lr`, `epochs=100`, `patience=20`, `seed=42`, `imgsz=640`, `batch=-1`,
dataset + split + test set, `hsv 0.015/0.4/0.4`, `degrees 5`, `translate 0.1`,
`fliplr 0.5`, no mixup/copy_paste/perspective/flipud/shear, pinned
`torch==2.7.1 / ultralytics==8.4.138`.

- **best epoch 86 / 100** (v1: 88). No overfitting — train *and* val losses were
  still decreasing at epoch 100 (v2 is still underfit, like v1).
- checkpoint SHA-256
  `26e4d603de0718b9b20c6bf5c9761ef1b114b6954f280925ea6c08e469357aa3` →
  `weights/yolo_solar_ground_research_v2.pt` (gitignored). last.pt `2d9bc0cf…`.
- Local re-evaluation reproduced the Kaggle test metrics exactly.
- **⚠️ training-cost regression: `multi_scale=True` on the P100 caused per-batch
  cuDNN re-benchmarking.** v2 took **16,661 s (~4 h 37 m) of pure training vs
  v1's ~48 min — a ~5.8× blow-up** for no measurable benefit. `multi_scale`
  should not be used here.

## 5. Test metrics (identical held-out split)

**Overall — v1 vs v2**

| metric | v1 | v2 | Δ |
|---|---|---|---|
| precision | 0.876 | 0.818 | **−0.058** |
| recall | 0.801 | 0.827 | **+0.026** |
| mAP50 | 0.865 | 0.876 | **+0.012** |
| mAP50-95 | 0.677 | 0.649 | **−0.028** |

**Per class (test)**

| class | metric | v1 | v2 | Δ |
|---|---|---|---|---|
| panel | precision | 0.947 | 0.900 | −0.047 |
| panel | recall | 0.925 | 0.945 | **+0.020** |
| panel | AP50 | 0.963 | 0.978 | **+0.015** |
| panel | AP50-95 | 0.843 | 0.824 | −0.019 |
| cracked | precision | 0.809 | 0.769 | −0.040 |
| cracked | recall | 0.653 | 0.697 | **+0.044** |
| cracked | AP50 | 0.749 | 0.746 | −0.003 |
| cracked | AP50-95 | 0.530 | 0.525 | −0.005 |
| dusty | precision | 0.871 | 0.784 | −0.087 |
| dusty | recall | 0.826 | 0.841 | **+0.015** |
| dusty | AP50 | 0.882 | 0.905 | **+0.023** |
| dusty | AP50-95 | 0.660 | 0.597 | −0.063 |

**Confusion matrix (v2 test, normalised by true class):** panel column strong —
0.90 detected, 0.05 missed. cracked 0.69 detected / 0.28 missed. dusty 0.58
detected / 0.32 missed. Background false positives: `cracked` 0.56 (up — v2
over-fires `cracked` on background), `panel` 0.28, `dusty` 0.16.

**Reading:** v2 trades precision for recall and marginally lifts mAP50 (panel
+0.015, dusty +0.023) while losing mAP50-95 (−0.028 overall) — its boxes are
looser. On the **in-distribution** 640×640 test set the two models are within
noise of each other. Per the brief, this is **not** grounds to declare
improvement.

Training/validation curves: `reports/v2/` + the run's `results.png`. PR / F1 /
confusion plots: the Kaggle output (not committed — bulky).

## 6. Six-photo real-world comparison (QUALITATIVE ONLY, full-frame, conf 0.45)

| # | image (dims) | v1 panels | v2 panels | v1 conf | v2 conf | winner |
|---|---|---:|---:|---|---|---|
| 1 | …56 (573×1280) | 1 | **0** | 0.87 | — | **v1** |
| 2 | …57 (1) (573×1280) | 2 | **0** | 0.51–0.87 | — | **v1** |
| 3 | …57 (573×1280) | 2 | **0** | 0.68–0.85 | — | **v1** |
| 4 | …58 (573×1280) | 1 | **0** | 0.58 | — | **v1** |
| 5 | …59 (1) (573×1280) | 2 | **0** | 0.59–0.63 | — | **v1** |
| 6 | …59 (1280×573) | 2 | **0** (1 via tiled fallback) | 0.67–0.75 | — | **v1** |

**v2 detects zero panels on all six owner photos at conf 0.45.** Dropping the
threshold to 0.05, v2's best `panel` confidence on these images is **0.16**, and
its single highest-confidence detection is often `cracked` (~0.35), not `panel`.
v1's panel confidences on the same images were 0.51–0.87.

**This is a severe real-world regression.** The aggressive augmentation made v2
fit the augmented **in-distribution** manifold well (panel test recall 0.945) but
generalise markedly *worse* to the owner photos, which differ in aspect ratio
(portrait/landscape vs square), capture device, and letterboxing. v1's lighter
augmentation transferred to the real photos; v2's does not.

Visual artefacts: `research_runs/ground_pv/v2_real_world/compare/` (gitignored),
`original | v1 | v2` per image — every v2 panel is blank.

## 7. Image 4 analysis

v1 produced a single 0.58 box over the background panels (the large foreground
panel was already missed). **v2 produces nothing** — no box at any class at conf
0.45, and only weak `panel`/`cracked` detections < 0.17 at conf 0.05. **v2 does
not recover image 4; it regresses it.**

## 8. Image 6 analysis

v1 produced two overlapping group boxes that together covered the array. **v2
produces nothing full-frame.** The opt-in tiled fallback (triggered because
full-frame v2 returned ≤ 1 panel) recovered **one** box over the left third of
the array — the same partial result seen in Phase 10, and worse than v1's two
boxes. **v2 does not separate the wide array; it regresses it.**

## 9. Crack false-positive investigation

On the in-distribution test set v2 actually **increased** `cracked`-on-background
false positives (confusion-matrix background→`cracked` 0.56 vs v1 ~0.28). On the
owner photos v2 emits no `cracked` boxes on roofing — but only because it emits
almost nothing at all. **v2 does not fix the roofing false-positive behaviour;
in-distribution it makes it worse.**

## 10. Condition-chain qualitative comparison

`v2 panel detector → panel crop → frozen condition model
(weights/yolo_solar_condition_research_v1.pt, SHA b077251a…)`:

| | v1 chain | v2 chain |
|---|---:|---:|
| panel crops (6 photos) | 10 | **0** |
| crops with a condition prediction | 8 | 0 |
| CLEAN / DUSTY / BIRD_DROP | 6 / 2 / 1 | 0 / 0 / 0 |

Because v2 detects no panels, the two-stage pipeline produces **no crops and no
condition output** on any owner photo. **Absence of a condition prediction does
not mean the panels are Clean** — it means the first stage failed. ELECTRICAL_DAMAGE
remains a HIGH-semantic-risk class (source boxes are cell cracks, not confirmed
electrical faults); hotspot remains excluded from RGB.

## 11. Optional tiled-fallback result

Fallback rule: run Phase 10 tiling only when full-frame v2 returns ≤ 1 panel.
It triggered on **all six** images (v2 full-frame returned 0). Result: 0 panels
on 5 images, 1 partial box on image 6. The fallback does not rescue v2 — it
inherits Phase 10's fragmentation/interior-tile failure on top of v2's already
weak per-tile confidence. **Not useful here.**

## 12. Limitations

- Six images, one rooftop, one device — not a representative real-world sample;
  all real-world judgements are visual, not scored.
- The single largest confound — **training data is 100 % square 640×640 while the
  owner photos are ~1:2.2** — is a data problem that augmentation cannot fully
  fix. Aspect-ratio augmentation is not exposed by Ultralytics; `shear`/
  `perspective` distort differently and were (correctly) left off.
- `multi_scale=True` was an over-reach: ~5.8× training cost, no benefit, and it
  is the most likely contributor to v2's OOD brittleness (the model learns to
  expect a specific set of scales).
- Only one v2 configuration was trained (compute-limited). A lighter variant
  (`scale` 0.5, `mosaic` 0.7, no `multi_scale`, no `close_mosaic` change) was
  not tested and might land differently.
- v2, like v1, is still underfit at 100 epochs.

## 13. Recommendation

**C — v2 improves some failure modes but introduces regressions.**

The only improvements are on the **in-distribution** test set: overall recall
+0.026, panel AP50 +0.015, dusty AP50 +0.023. These are marginal, partly offset
by test-set losses (precision −0.058, mAP50-95 −0.028, more `cracked` background
FPs), and they do **not** translate to the real world.

The regressions dominate:

1. **Real-world collapse** — v2 detects **0 panels on all 6 owner photos**
   (v1: a usable signal on 5/6). The two-stage pipeline is non-functional with v2.
2. **Image 4 not recovered** — regressed (v1: 1 box, v2: 0).
3. **Image 6 not separated** — regressed (v1: 2 group boxes, v2: 0 / 1-via-fallback).
4. **Roofing crack FP not fixed** — worse in-distribution.
5. **~5.8× training-cost blow-up** from `multi_scale` for no benefit.

**Actions:**

- **Keep `weights/yolo_solar_ground_research_v1.pt` as the ground panel
  detector.** Do not adopt v2. Do not merge, deploy, or replace any production
  model.
- The real fix is **data, not augmentation**: the training set needs
  genuinely representative examples — portrait/landscape rooftop photos, distant
  oblique arrays, phone-camera captures — sourced and licensed properly, not
  synthesised from square stock crops. Until then, v1 + the Phase 10
  off-panel `cracked`/`dusty` filter recommendation remains the best available
  research pipeline.
- If augmentation is revisited: drop `multi_scale`, use a moderate `scale`
  (~0.5) and `mosaic` (~0.7), and **validate on held-out real photos before
  trusting test metrics**.

## Integrity

| artifact | SHA-256 | state |
|---|---|---|
| `weights/yolo_solar.pt` | `0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea` | **unchanged** |
| `weights/yolo_solar_ground_research_v1.pt` | `2b26334dc4023ba4c89a80e2350ddc9012358a48cdef713aee86c8e79a1bbb50` | **unchanged** |
| `weights/yolo_solar_ground_research_v2.pt` | `26e4d603de0718b9b20c6bf5c9761ef1b114b6954f280925ea6c08e469357aa3` | new (gitignored research artifact) |
| `weights/yolo_solar_condition_research_v1.pt` | `b077251a514424481d51b6a57ebcc30f4a607ea7559053cbb3ef7d0de88f9578` | **unchanged** |
| `weights/mobilenet_solar_v1.pth` | `afccaccfcc309952f7a94d754aaafc22e7e3391416b9518c5f4a8635b1c2682b` | **unchanged** |
| `weights/manifest.json` | `eabfec951724dff9975e57c9d3a290d272c61853a66b7c958fc3cfe58e907363` | **unchanged** |

Production files modified: **none** (`models/`, `services/`, `app.py`, `pages/`,
`v1.0.0`). `git diff` shows no production changes.
