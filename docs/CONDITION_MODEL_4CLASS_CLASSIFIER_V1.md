# Phase 16-B — Panel-crop condition CLASSIFIER v1 (MobileNetV2)

Research checkpoint only. Not production. Not wired into `app.py`. Does not
modify, replace, or touch any existing checkpoint (box-detector condition
models v2/v3_960/v4_aug all remain frozen and unchanged).

## Motivation

`docs/CONDITION_MODEL_IMAGE_LEVEL_AGREEMENT.md` (Phase 16-A.1) found, with
zero new training, that the existing box-detection condition model's
weakness (test mAP50 0.444, box-level recall 0.441) is dominated by
**localization noise, not class recognition** — image-level agreement was
72.7%, vs. 44.1% box-level recall, on the same checkpoint. DUSTY showed the
single largest gap (+36pp), consistent with its already-documented diffuse,
boundary-less texture. That evidence justified building a crop-level
classifier (no localization requirement) instead of another box-detector
variant.

## What was built

1. **Derived classification dataset** (`scripts/build_condition_4class_classifier_dataset.py`):
   one dominant label per image, picked deterministically from the existing,
   already-audited box annotations (majority instance count → largest total
   area → alphabetical). No new photography, no new annotation. Every image
   keeps its EXISTING train/valid/test split assignment — leakage-clean by
   construction. 2,411 images kept (0 dropped); 910 had more than one
   distinct class in their original boxes, so picking one dominant label is
   a real, disclosed information loss versus the detection dataset.
   Per-split counts: train 1686 (BIRD_DROP 509 / CLEAN 367 / DUSTY 749 /
   PHYSICAL_DAMAGE 61), valid 483, test 242 (CLEAN 55 / DUSTY 107 /
   BIRD_DROP 71 / PHYSICAL_DAMAGE 9) — identical totals to the source
   detection dataset.
2. **Model**: MobileNetV2, ImageNet-pretrained, classifier head replaced —
   same architecture family as the project's existing production classifier
   (`models/classifier.py`), reused deliberately rather than inventing a new
   architecture for this experiment.
3. **Training**: Kaggle GPU (Tesla P100), class-weighted cross-entropy
   (inverse frequency — PHYSICAL_DAMAGE is ~10x rarer than DUSTY), AdamW
   lr=5e-4, batch 32, up to 30 epochs with early stopping (patience 6) on
   validation accuracy, seed 42. Best checkpoint (by val accuracy) kept, not
   the last epoch. Training environment, hyperparameters, and full
   per-epoch history are recorded in
   `docs/CONDITION_MODEL_4CLASS_CLASSIFIER_V1_TRAINING.json`.
4. Notebook: `kaggle/ground_condition_4class_classifier_train.ipynb`
   (regenerable via `scripts/make_condition_4class_classifier_notebook.py`).
   Dataset: `edithstark/solar-ai-ground-condition-4class-classifier-v1` (PRIVATE).

Training took **233 seconds** (best epoch 10, early-stopped ~epoch 16) on a
Tesla P100 — best validation accuracy 0.770.

## Result — held-out TEST set (242 images, evaluated once)

**Overall image-level accuracy: 0.752** (182/242), vs. **0.727** image-level
agreement for the box-detector on the same test images (Phase 16-A.1) — a
real, modest improvement on the apples-to-apples image-level comparison.

| class | n (test) | precision | recall | F1 | box-detector recall (16-A.1 baseline) |
|---|---:|---:|---:|---:|---:|
| BIRD_DROP | 71 | 0.727 | 0.676 | 0.701 | 0.388 |
| CLEAN | 55 | 0.750 | 0.764 | 0.757 | 0.545 |
| **DUSTY** | 107 | 0.802 | **0.794** | 0.798 | 0.359 |
| PHYSICAL_DAMAGE | 9 | 0.500 | 0.778 | 0.609 | 0.577 |

Confusion matrix (rows = true, cols = predicted, order BIRD_DROP / CLEAN /
DUSTY / PHYSICAL_DAMAGE):

```
[48,  6, 14,  3]
[ 5, 42,  7,  1]
[12,  7, 85,  3]
[ 1,  1,  0,  7]
```

## Interpretation

**DUSTY — the class Phase 16-A.1 flagged as most likely to benefit — shows
the largest real improvement**: recall 0.359 → 0.794, more than double.
This is consistent with the hypothesis that DUSTY's failures were
predominantly localization noise around a diffuse, boundary-less texture
that a classifier (no box to draw) doesn't have to get right. CLEAN and
BIRD_DROP improved more modestly. **PHYSICAL_DAMAGE precision is weak
(0.5)** — expected and disclosed, not hidden: only 9 test images, the
rarest class by a wide margin (61 train images vs. DUSTY's 749), so this
number carries very high variance and should not be over-read.

Note the comparison discipline: 0.752 (classifier) vs. 0.727 (box detector,
image-level) is the fair, apples-to-apples number — both are image-level,
no-localization-required metrics on the same test images. It is NOT fair to
compare 0.752 against the box detector's 0.444 mAP50, which is a different,
stricter, localization-requiring metric.

## What this does NOT show

- Not evidence about real-world (six-photo) performance — only the held-out
  test split was used here, which has ground truth; the six real reference
  photos still have none and remain qualitative-only.
- Not a claim that the classifier is production-ready — 0.752 image-level
  accuracy is a real improvement over the prior formulation, not a
  validated deployment bar. PHYSICAL_DAMAGE in particular needs more data
  before its precision figure means much.
- Not a replacement decision — the classifier drops localization entirely.
  For BIRD_DROP/PHYSICAL_DAMAGE, knowing *where* on the panel still has
  real product value that this reformulation gives up. A future hybrid
  (Option E from `docs/PHASE_16_ARCHITECTURE_REVIEW.md` §6) may combine
  both rather than fully replacing the detector.

## Status

Frozen research checkpoint: `weights/mobilenet_condition_4class_research_v1.pth`
(SHA-256 `1614b0a6a9072b64db4c883c158cfb2e0868a7255af2f89fc45c6979ca413a01`).
Not wired into `app.py` or any product page. Next decision: whether to
integrate this classifier into the Ground Inspection page (replacing or
running alongside the box-detector condition model), which is a product
decision to make deliberately, not a side effect of this experiment.
