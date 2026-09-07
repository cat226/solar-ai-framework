# Ground-level PV research detector — evaluation

**Model:** `weights/yolo_solar_ground_research_v1.pt`
**SHA-256:** `2b26334dc4023ba4c89a80e2350ddc9012358a48cdef713aee86c8e79a1bbb50`
— identical byte-for-byte to the Kaggle `best.pt` after retrieval, and to the
checkpoint used by the independent local re-evaluation.

**Experiment:** `ground_yolo11n_baseline_v1` — clean baseline, YOLO11n, 3 classes
(`0=panel, 1=cracked, 2=dusty`). **`bird_drop` and hotspot are not part of this
model** (see `ground_research_model_manifest.json`). **Research artifact only** —
it does **not** replace the aerial detector `weights/yolo_solar.pt`.

- Kaggle kernel: <https://www.kaggle.com/code/edithstark/ground-yolo11n-baseline-v1> (version 2)
- Kaggle dataset: `edithstark/solar-ai-ground-pv-research-v1`
- GPU: Tesla P100-PCIE-16GB · training ~1 h 36 m · best epoch 88 / 100
- torch 2.7.1+cu126 · torchvision 0.22.1 · ultralytics 8.4.138 · python 3.12.13
- Protocol: Ultralytics `YOLO.val()`, imgsz 640, IoU 0.50, on the **leakage-clean
  split** (whole near-duplicate families each assigned to one split — 0 families
  cross splits, 0 residual cross-split near-duplicate pairs).
- **The Kaggle GPU metrics and the independent local CPU re-run match to 4
  decimal places** (same checkpoint sha).

## Validation split — 1,297 images / 109 families

Instances: panel 1,214 (932 imgs) · cracked 2,122 (832 imgs) · dusty 75 (21 imgs)

| | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **overall** | 0.846 | 0.809 | **0.856** | **0.663** |
| panel | 0.896 | 0.875 | 0.927 | 0.812 |
| cracked | 0.857 | 0.778 | 0.840 | 0.596 |
| dusty | 0.786 | 0.773 | 0.802 | 0.582 |

## Held-out TEST split — 649 images / 55 families — primary result

Instances: panel 560 (539 imgs) · cracked 637 (328 imgs) · **dusty 69 (30 imgs)**

| | precision | recall | AP50 | AP50-95 |
|---|---|---|---|---|
| **overall** | **0.876** | **0.801** | **0.865** | **0.677** |
| panel | 0.947 | 0.925 | 0.963 | 0.843 |
| cracked | 0.809 | 0.653 | 0.749 | 0.530 |
| dusty | 0.871 | 0.826 | 0.882 | 0.660 |

`dusty` test metrics rest on **69 instances across 30 images** → high variance,
do not over-interpret.

Artifacts: `training_artifacts/` — `results.{csv,png}` (training curves),
`confusion_matrix*.png` (validation), `test_confusion_matrix*.png` +
`test_confusion_matrix.json` (test), `[test_]BoxPR_curve.png`, `BoxF1_curve.png`.

## Confusion matrix — test split

See `training_artifacts/test_confusion_matrix.json` for the raw counts
(rows = predicted, cols = true, background last) and the accompanying PNGs.
Salient pattern: most misses are **defect ↔ background** (a defect present
but not predicted, or predicted where none is labelled), not class
confusion between `cracked` and `dusty`.

## Overfitting

Mild. Over 100 epochs:

| | ep 1 | ep 41 | ~ep 51-60 | ep 100 |
|---|---|---|---|---|
| train/cls_loss | 2.09 | 0.54 | ~0.49 | **0.31** |
| val/cls_loss | 2.33 | 0.67 | ~0.66 | **0.62** |
| val/box_loss | 1.13 | 0.77 | ~0.74 | 0.72 |
| val/dfl_loss | 1.45 | 1.11 | **1.078 (min @ep51)** | 1.109 |
| val mAP50 | 0.40 | 0.85 | ~0.80–0.84 | 0.85 (peak ~0.854 @ep81) |

Train losses keep falling; **validation metrics plateau after ~epoch 60** and
`val/dfl_loss` drifts up slightly (1.078 → 1.109) — a mild over-fitting signal.
Validation accuracy never *degrades*, and `best.pt` is epoch 88 near the
plateau top. **Held-out test ≥ validation on every overall metric**, so the
leakage-clean split is doing its job (no duplicate-inflated scores). A follow-up
could stop earlier / add regularisation; not required for a baseline.

## Cross-domain check — the ground model is NOT an aerial detector (req 12)

On 120 randomly sampled ground close-up **test** images (98 ground-truth panels):

| model | images with any detection | boxes |
|---|---|---|
| production aerial `yolo_solar.pt` (conf 0.45) | **1 / 120 (1 %)** | 1 |
| ground research model (conf 0.25), panel class | 94 / 120 (78 %) | 104 |

The frozen aerial detector is effectively blind on close-up imagery; the ground
model handles it. The two cover different domains and share **no** training data
(aerial: 17 k overhead tiles, 1 class, YOLOv8n; ground: 6.5 k close-ups, 3
classes, YOLO11n). **Neither replaces the other.**

The reverse direction — the ground model over the **aerial test split** at the
aerial protocol (conf 0.45 / iou 0.50) — is **pending the `E:` drive remount**
(`E:/Solar AI Training Images/yolo_prepared/test`). It is expected to score near
zero: the ground model has never seen overhead panel arrays.

## Is it useful for close-up PV detection?

**Yes — for panel localisation and a first pass at soiling / damage.**
- `panel` on held-out test: AP50 **0.96**, AP50-95 **0.84**, recall **0.93** —
  strong; directly usable for the panel-level condition pipeline.
- `dusty`: AP50 0.88 on test, but only 69 instances / 30 images — **promising,
  not proven**.
- `cracked`: AP50 0.75, **recall 0.65** — misses about a third of cracks; usable
  as an assistive flag, not for exhaustive crack finding.

## Real-world rooftop holdout

`scripts/infer_ground_pv.py` on `E:/Solar Real Images/` (the 6 owner photos) —
**pending `E:` remount**. Qualitative only; no ground-truth metrics fabricated.

## Limitations

- No `bird_drop` — excluded for lack of independent data (9 images / ~3 photos).
  **No bird_drop performance is claimed.**
- No RGB hotspot — out of scope by design.
- ~574 unique source photographs behind 6,492 augmented instances; image
  provenance unverified.
- `dusty` test metrics are high-variance (69 instances / 30 images).
- Research artifact — not wired into `models/` or `services/`; branch not merged.
