# Ground-PV CONDITION/DEFECT model — baseline report (`condition_yolo11n_baseline_v1`)

**MODEL ASSESSMENT: RESEARCH-ONLY**  ·  **PRODUCTION: KEEP SEPARATE**

> Experimental research model. A **second-stage** condition/defect detector that
> runs on **panel crops** from the separate ground panel detector — it is **not**
> a panel detector and is **not** production. Provenance grade **C**
> (`personal-experimental-unverified-provenance`); underlying images ultimately
> trace to Kaggle `pythonafroz/solar-panel-images` (licence **Unknown**). Do not
> redistribute / publish / use commercially / claim verified CC BY image rights.

---

## KAGGLE

| | |
|---|---|
| notebook | `edithstark/condition-yolo11n-baseline-v1` (`kaggle/ground_condition_train.ipynb`) |
| notebook version | 1 · status COMPLETE |
| dataset | `edithstark/solar-ai-ground-condition-research-v1` (PRIVATE) |
| GPU | Tesla P100-PCIE-16GB (17.1 GB), CUDA 12.6 |
| runtime | 2 917.7 s pure training (~48.6 min); longer wall-clock from Kaggle queue/setup |
| env | Python 3.12.13 · torch 2.7.1+cu126 · torchvision 0.22.1+cu126 · ultralytics 8.4.138 |

Dataset content SHA `dcebb5e7…` · data.yaml SHA `5fb557b8…` · split manifest SHA
`9eb47dd2…` · taxonomy SHA `4be303ba…`.

## TRAINING

| arg | value |
|---|---|
| base model | `yolo11n.pt` (COCO-pretrained) — never a Solar AI model |
| classes | 0=CLEAN 1=DUSTY 2=BIRD_DROP 3=PHYSICAL_DAMAGE 4=ELECTRICAL_DAMAGE (no PANEL) |
| imgsz / epochs / patience | 640 / 100 / 20 |
| batch | −1 (AutoBatch) |
| optimizer / cos_lr / pretrained | auto / True / True |
| seed | 42 |
| device | cuda:0 |
| augmentation | hsv (0.015/0.4/0.4), degrees 5, translate 0.1, scale 0.3, fliplr 0.5, mosaic 0.5; no shear/perspective/flipud/mixup/copy_paste — identical to the prior ground-detector run |

- **best epoch: 98 / 100.**  Early stopping (patience 20) never triggered.
- **checkpoint SHA-256:** `b077251a514424481d51b6a57ebcc30f4a607ea7559053cbb3ef7d0de88f9578`
  → saved locally as `weights/yolo_solar_condition_research_v1.pt` (gitignored).
  last.pt `da0e1b34…` → `weights/yolo_solar_condition_research_v1.last.pt`.
- Local re-evaluation (ultralytics 8.4.118, CPU) **reproduced the Kaggle metrics exactly.**

## TEST (held-out leakage-clean split, 250 images, 2 348 instances)

**Overall — precision 0.623 · recall 0.449 · mAP50 0.472 · mAP50-95 0.254**

| class | precision | recall | AP50 | AP50-95 | test imgs | test instances | unique families |
|---|---|---|---|---|---|---|---|
| CLEAN | 0.590 | 0.497 | 0.516 | 0.335 | 95 | 545 | 53 |
| DUSTY | 0.493 | 0.348 | 0.358 | 0.163 | 144 | 863 | 90 |
| BIRD_DROP | 0.565 | 0.320 | 0.374 | 0.149 | 100 | 726 | 56 |
| PHYSICAL_DAMAGE | 0.660 | 0.526 | 0.523 | 0.275 | 30 | 78 | 18 |
| **ELECTRICAL_DAMAGE** | **0.808** | **0.555** | **0.588** | **0.348** | 44 | 144 | 31 |

(validation, 502 images: precision 0.624 · recall 0.479 · mAP50 0.520 · mAP50-95 0.279.)

### Confusion-matrix findings

- **Class-vs-class confusion is minimal.** The model rarely calls one defect
  another. The dominant error is **missed detections** (false negative → background):
  CLEAN 0.50 missed · DUSTY 0.63 · BIRD_DROP 0.66 · PHYSICAL_DAMAGE 0.45 ·
  ELECTRICAL_DAMAGE 0.43.
- **False positives on background** come mostly from DUSTY (0.43) and CLEAN (0.28);
  PHYSICAL_DAMAGE and ELECTRICAL_DAMAGE almost never false-positive (0.03 each).
- Plots: `reports/plots/train/` (curves + train confusion matrix) and
  `reports/plots/test/` (held-out confusion matrix, Box PR / F1 curves).

## ⚠️ ELECTRICAL_DAMAGE — HIGH SEMANTIC RISK

**ELECTRICAL_DAMAGE: HIGH SEMANTIC RISK.**

Test metrics: **precision 0.808 · recall 0.555 · AP50 0.588 · AP50-95 0.348** — the
**highest precision of any class**, and (with PHYSICAL_DAMAGE) the lowest
false-positive rate.

**This is not evidence of electrical-fault diagnosis.** Every ELECTRICAL_DAMAGE
label is S1's `electrical-damage` class, and the provenance audit found those boxes
**visually correspond to cell cracks**, not confirmed electrical failure. The label
was preserved unchanged (not relabelled). The most likely explanation for the strong
score is simply that **cell cracks are crisp, sharply-localised visual features** and
therefore easy to detect and bound — exactly the pattern you would expect if the
class is really "visible cracks" wearing an "electrical-damage" name. Per-annotation
records: `manifests/electrical_damage_annotations.csv` (all flagged
`electrical_damage_semantic_risk = HIGH`).

## REAL WORLD — six owner photos (QUALITATIVE ONLY)

Pipeline: `photo → weights/yolo_solar_ground_research_v1.pt (panel detector) →
panel crop → weights/yolo_solar_condition_research_v1.pt (condition)`. The condition
model was **never** run on the full photographs. Full association chain preserved in
`research_runs/ground_pv/condition_real_world/` (per-image JSON, crops, annotated
images). **No ground truth — no accuracy / precision / recall / F1 / mAP.** A
prediction is a raw model output, **not a fact**.

| photo | panels (stage 1) | per-panel condition predictions (raw) |
|---|---|---|
| …15.25.56 | 2 | #1 DUSTY 0.33 · #2 — no prediction |
| …15.25.57 (1) | 2 | #1 BIRD_DROP 0.27 · #2 DUSTY 0.45 |
| …15.25.57 | 2 | #1 CLEAN 0.81 · #2 CLEAN 0.46 |
| …15.25.58 | 2 | #1 — no prediction · #2 — no prediction |
| …15.25.59 (1) | 2 | #1 CLEAN 0.32 · #2 CLEAN 0.55 |
| …15.25.59 | 2 | #1 CLEAN 0.60 / 0.28 · #2 — no prediction |

- 12 panels detected, **8 got a condition prediction, 4 got none** (absence is **not**
  "Clean").
- Predicted classes across all panels: CLEAN ×6, DUSTY ×2, BIRD_DROP ×1,
  **PHYSICAL_DAMAGE ×0, ELECTRICAL_DAMAGE ×0**.
- The stage-1 ground detector separately flagged `cracked` on two photos (conf 0.61,
  0.50), but the condition model predicted **no** PHYSICAL/ELECTRICAL on those crops —
  the two research models disagree.
- Confidences are low (mostly 0.27–0.55). No obvious gross false positives; the
  obvious "miss" is that several visibly dusty/soiled real panels drew no DUSTY box —
  consistent with the low DUSTY recall on the test set.

## MODEL ASSESSMENT — quality / failure analysis

1. **BIRD_DROP** — recall 0.32, AP50 0.37. Weak overall, but qualitatively the
   detections are **real**: on the test predictions, boxes land squarely on actual
   white droppings (several 0.9–1.0 hits on real splats). The model finds obvious
   droppings and misses subtle/small ones.
2. **DUSTY** — the weakest class (recall 0.35, AP50-95 0.16) despite the most data
   (863 test instances, 90 families). "Dust" is a diffuse, low-contrast, ill-bounded
   region; bounding-box detection is a poor fit and IoU-based AP punishes it hard.
3. **PHYSICAL_DAMAGE** — recall 0.53, AP50 0.52 on very little data (78 instances,
   18 families). High-variance; treat as indicative only.
4. **ELECTRICAL_DAMAGE** — strongest class. **See the HIGH-semantic-risk section —
   this most likely reflects the source labelling cell cracks as "electrical-damage",
   not electrical-fault detection.**
5. **CLEAN behaviour** — precision 0.59, recall 0.50. CLEAN is only trained where the
   source drew explicit clean-region boxes, so the model emits CLEAN boxes rather than
   inferring cleanliness from absence. **Absence of any detection is not "Clean"** and
   is reported as such.
6. **Class confusion** — negligible. Errors are misses, not mislabels.
7. **False positives** — concentrated in DUSTY and CLEAN on background; damage classes
   almost never false-positive.
8. **False negatives** — the primary failure mode across every class (43–66% of true
   instances missed at conf 0.25 / iou 0.5).
9. **Overfitting** — **none.** Train *and* validation losses were still decreasing at
   epoch 100 and val mAP was still rising. The model is **underfit**: it would likely
   benefit from more epochs, a larger backbone, and more (and better-provenance) data.
10. **Real defects vs dataset artefacts** — the model is learning **genuine visible
    features** (bird-drop localisation is convincing). **Caveat:** many source images
    carry visible **Shutterstock watermarks**; a watermark/stock-photo-style confound
    cannot be ruled out, and the unverified provenance compounds this.

**If ELECTRICAL_DAMAGE appears strong:** it does — and this most likely reflects the
source's cell-crack labelling rather than any electrical-fault capability. Do not
promote or describe this class as electrical-fault detection.

## HOTSPOT

Not added, not inferred. Hotspot requires a separate thermal/IR model and is never
derived from RGB.

## PRODUCTION SAFETY

| artifact | SHA-256 | state |
|---|---|---|
| `weights/yolo_solar.pt` | `0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea` | **unchanged** |
| `weights/yolo_solar_ground_research_v1.pt` | `2b26334dc4023ba4c89a80e2350ddc9012358a48cdef713aee86c8e79a1bbb50` | **unchanged** |
| `weights/mobilenet_solar_v1.pth` | `afccaccfcc309952f7a94d754aaafc22e7e3391416b9518c5f4a8635b1c2682b` | **unchanged** |

- Files modified in production: **none** (`models/`, `services/`, `app.py`, `pages/`,
  `weights/manifest.json`, `v1.0.0` all untouched).
- Dataset immutability: `datasets/ground_pv_research/condition/` images/labels
  unchanged by training; the existing `datasets/ground_pv_research/yolo/` dataset
  untouched.
- The condition model is **not** integrated into production inference.

## DISPOSITION

**RESEARCH-ONLY.** A weak but honest, reproducible, leakage-clean baseline on an
unverified-provenance dataset. **PRODUCTION: KEEP SEPARATE** — no merge, no deploy,
no replacement of any production model, no production-readiness claim.
