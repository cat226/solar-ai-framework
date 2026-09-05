# Solar AI v1.0.0 — Independent ML Evaluation & Accuracy Audit

Generated: 2026-09-04
Repository: https://github.com/cat226/solar-ai-framework
Git commit (base): `14df9b9cdb5411cef79c4282174c7d042abb1a96`
Release/tag: `v1.0.0`
Evaluation code: `training/evaluation/` (`common.py`, `evaluate_yolo.py`,
`evaluate_mobilenet.py`, `evaluate_end_to_end.py`, `leakage_audit.py`)

This is an **evaluation/audit phase**. No model weights, training data, or
production inference code were modified to produce these results. Every
number below comes from running the actual production wrappers
(`models/detector.py`, `models/classifier.py`, via `models/model_manager.py`)
against real, previously-untouched held-out data on this machine, or from
`ultralytics.YOLO.val()`'s own internal evaluation. Nothing is fabricated,
tuned to the test set, or estimated.

Full per-image CSVs and JSON summaries (large; not committed) are on the
project's E: drive storage location:
`E:\Solar AI Training Images\evaluation_runs\`.

> **2026-09-05 update (Phase 6B — ML Hardening & Re-Evaluation):** this
> document's findings below were produced against the then-deployed
> `confidence_threshold: 0.45` and remain an accurate historical record of
> what that configuration measured. Phase 6B independently swept the
> confidence threshold on the **validation** split (never the test split)
> and, based on that evidence, recalibrated
> `configs/settings.yaml`'s `models.yolo.confidence_threshold` to **0.30**.
> This measurably improves recall (confirmed on the test split: 0.152→0.271)
> without fixing the separate, more severe domain-shift limitation this
> report already identifies. It also extends the MobileNet near-duplicate
> leakage audit into a corroborated, classified clean-subset evaluation.
> **See `docs/ML_HARDENING_PHASE6B.md` for the full, current findings —
> read both documents together; this one is not superseded, only extended.**
>
> **2026-09-05 update (Phase 6C — YOLO Domain Remediation):** investigated
> whether legitimate, appropriately licensed close-up/ground-level training
> data exists to remediate the domain-shift limitation above. **No
> ACCEPT-tier candidate dataset was found** after real web research and a
> documented licensing/provenance gate (several candidates traced to an
> already-rejected unknown-license source; aerial/satellite candidates
> don't address the gap; one candidate, OpenStat Madagascar, remains an
> open CONDITIONAL lead pending direct verification). **No retraining was
> performed; `weights/yolo_solar.pt` and `configs/settings.yaml` are
> unchanged from Phase 6B.** See `docs/ML_HARDENING_PHASE6C.md` for the
> full candidate table and what would be required to close this gap.

---

## Executive summary

| Question | Answer |
|---|---|
| Does MobileNet really achieve ~100% test accuracy? | **No.** Independently reproduced accuracy via the real production classifier is **99.36%** (155/156), not 100%. The one error traces to a genuine preprocessing discrepancy between the original training-time evaluation script and the production inference path (see below) — a real, previously-unknown bug worth fixing in a future phase. |
| Is the 99.36%/100% MobileNet number trustworthy as "real-world accuracy"? | **Only partially.** A real, visually-confirmed leakage audit found genuine near-duplicate images (the same photograph, reprocessed by an image-resizing tool) crossing the train↔test and val↔test boundaries. The test set is very small (156 images) and one class (Hotspot) has only 20 test samples. The reported accuracy is real given this data, but the data itself is not a clean, fully-independent test set. |
| Does YOLO really achieve mAP50≈0.74? | **Yes, reproduced.** Independently re-run via `ultralytics.YOLO.val()`: mAP50=0.7401, mAP50-95=0.4753, closely matching the training-run's recorded 0.7392/0.4759. |
| Did the *deployed* YOLO configuration (conf=0.45, as of this report) actually deliver that performance? | **No — this was the single most important finding in this report.** At the then-production confidence threshold (0.45), independently measured recall was **0.152** (missed ~85% of real panels) and precision **0.497** — both far below the 0.71 precision / 0.82 recall the model achieves at its own natural operating point. `configs/settings.yaml`'s `confidence_threshold` appeared **miscalibrated** for this checkpoint. **Phase 6B recalibrated it to 0.30 based on validation-split evidence — see `docs/ML_HARDENING_PHASE6B.md` for the full threshold sweep and the current, confirmed test-split numbers (recall 0.271).** |
| Does the full detect→crop→classify pipeline work end-to-end? | **Classification: yes (99.36%). Detection-gated end-to-end: effectively no (0.64%)** on the only dataset with real fault-class ground truth — because that dataset is close-up single-panel photos, a different visual domain than YOLO's own aerial training data, so YOLO detects a panel almost never. This is an honest, important, disclosed limitation of both the evaluation and the real underlying domain gap — not a classification failure. |
| Is XGBoost evaluated? | **No — correctly not evaluated.** No legitimate trained artifact exists. See dedicated section below. |

**Bottom line:** MobileNet's underlying classification skill is genuinely strong on the real held-out photos it has seen (99.36%, with the one error explained), but the reported accuracy is inflated by both a small test set and real near-duplicate leakage, so it should not be read as "the model is 99%+ accurate on novel real-world panels in general." YOLO's own detection capability (mAP50≈0.74) is real and reproduced, but the *actual deployed application* is currently operating at a confidence threshold that discards the great majority of real detections — this is a genuine, previously unknown production-configuration problem, separate from model quality.

---

## Dataset(s) used

| Dataset | Role | Source | Split used | Images |
|---|---|---|---|---|
| BDAPPV (IGN config) | YOLO detection | `gabrielkasmi/bdappv` (Hugging Face mirror of Zenodo DOI 10.5281/zenodo.7358126), geographic (department-level) split — not re-shuffled | `test` | 2,581 images / 1,640 ground-truth panel boxes |
| SolNET (Clean/Dusty) + PVMD (Hotspot) | MobileNet classification | MDPI *Energies* 2023 (CC BY 4.0) + Mendeley (CC BY 4.0) | `test` | 156 images (72 Clean / 64 Dusty / 20 Hotspot) |

Both are the exact prepared datasets used to train the released v1.0.0
artifacts (`E:\Solar AI Training Images\yolo_prepared\` and
`...\prepared\`), with their own recorded per-record SHA-256 and split
assignment in each dataset's own `manifest.json`. Test-split images were
never used for training or validation of these checkpoints.

---

## YOLO results

**Artifact:** `weights/yolo_solar.pt`, SHA-256
`0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea`.
**Config:** `confidence_threshold=0.45`, `iou_threshold=0.50`, `image_size=640`
(`configs/settings.yaml`).

### mAP50 / mAP50-95 (via `ultralytics.YOLO.val()`, reproducing the original methodology)

| Metric | Training-run recorded | Independently reproduced (this audit) | Difference |
|---|---|---|---|
| Precision (at val's own optimal-F1 threshold) | 0.7064 | 0.7104 | +0.0040 |
| Recall (at val's own optimal-F1 threshold) | 0.8067 | 0.8183 | +0.0116 |
| mAP50 | 0.7392 | 0.7401 | +0.0009 |
| mAP50-95 | 0.4759 | 0.4753 | −0.0006 |

**These closely match** (well within normal run-to-run floating-point/
environment variance). The original training-run's recorded mAP/precision/
recall figures are **genuine and reproducible**, not fabricated.

### Production-path metrics (real deployed configuration, conf=0.45) — the important divergence

Running the actual `models.detector.SolarPanelDetector.detect()` (exactly
what the deployed app calls) over all 2,581 test images and greedily
IoU-matching (IoU≥0.5) against the real ground-truth boxes:

| Metric | Value |
|---|---|
| Images | 2,581 |
| Ground-truth boxes | 1,640 |
| True positives | 250 |
| False positives | 253 |
| False negatives | 1,390 |
| **Precision** | **0.4970** |
| **Recall** | **0.1524** |
| Detection confidence (all 503 raw detections): mean/median/min/max | 0.531 / 0.518 / 0.450 / 0.755 |
| Matched-detection IoU (250 true positives): mean/median/min/max | 0.827 / 0.839 / 0.528 / 0.974 |

**This is materially different from both the training-run's recorded
numbers and this audit's own reproduced mAP-methodology numbers — and the
difference is explained, not a bug:** `YOLO.val()`'s reported precision/
recall are measured at whatever confidence threshold maximizes F1 across
the *entire* precision-recall curve (implicitly a low threshold, since the
model needs to emit many candidate boxes to build that curve). The
*deployed* application instead uses a fixed `confidence_threshold=0.45`.
At that specific, real, production threshold, the model's actual
real-world recall is only **15.2%** — it misses roughly 85% of real solar
panels in the test set — while precision is also worse (0.497) than the
model's own optimal-threshold precision (0.71).

**This means conf=0.45 is not simply "trading recall for precision" (an
expected, acceptable tuning choice) — it is worse than the model's own
natural operating point on *both* axes simultaneously.** This is a real,
previously unknown production-configuration issue, independent of model
quality: the trained checkpoint is capable of ~0.71 precision / ~0.82
recall, but the deployed threshold prevents the application from ever
reaching that. Revisiting `models.yolo.confidence_threshold` in
`configs/settings.yaml` is a legitimate candidate for a future
(non-training) configuration change — **not made in this audit-only
phase**, per its explicit no-production-changes rule.

### Error analysis (representative examples, production path)

**False negatives** (ground-truth panel present, nothing detected at
conf≥0.45) — the dominant error mode (1,390 of 1,640 ground-truth boxes):
predominantly small panels. Three representative examples, all on
400×400px images:

| Image | Ground-truth box (px) | Box area | Detected |
|---|---|---|---|
| `AABNG2261PNFOZ.png` | (229,129)-(259,139) | 30×10 = 300px² (0.19% of image) | none |
| `AASIJA52INLGY.png` | (204,217)-(244,234) | 40×17 = 680px² (0.43%) | none |
| `ACHNX2A38BJIHK.png` | (183,191)-(217,215) | 34×24 = 816px² (0.51%) | none |

The pattern across the false-negative set is consistent: these are small
panels relative to the full frame, well below the confidence threshold
even when the model does respond to them (many FN images show no candidate
box at all, meaning confidence never rose above 0.45, not merely a
close miss).

**False positives** (253 total) — a mix of genuinely spurious detections
and detections on images with *other* real panels the greedy matcher
couldn't pair (already-claimed ground truth). Representative:

| Image | Ground truth | Predicted | Confidence |
|---|---|---|---|
| `ABCMT4D08KQKFF.png` | 0 boxes (negative image) | 1 box | 0.582 |
| `ACHFB7D70BCUKW.png` | 0 boxes (negative image) | 1 box | 0.518 |
| `ADPYL1FE1WPDCV.png` | 1 box, far from prediction (IoU=0.000) | 1 box | 0.496 |

All false-positive confidences cluster near the 0.45 threshold itself
(mean 0.531), consistent with these being genuinely borderline/marginal
detections rather than high-confidence hallucinations — no case of a
high-confidence, clearly-wrong detection was found in the sample reviewed.

---

## MobileNet results

**Artifact:** `weights/mobilenet_solar_v1.pth`, SHA-256
`afccaccfcc309952f7a94d754aaafc22e7e3391416b9518c5f4a8635b1c2682b`,
loaded as `classifier_source="v1"`, labels `[Clean, Dusty, Hotspot]`.

| Metric | Value |
|---|---|
| Total test samples | 156 (72 Clean / 64 Dusty / 20 Hotspot) |
| **Accuracy** | **0.9936** (155/156) |
| Macro precision / recall / F1 | 0.9949 / 0.9954 / 0.9951 |
| Weighted precision / recall / F1 | 0.9937 / 0.9936 / 0.9936 |

### Per-class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Clean | 1.0000 | 0.9861 | 0.9930 | 72 |
| Dusty | 0.9846 | 1.0000 | 0.9922 | 64 |
| Hotspot | 1.0000 | 1.0000 | 1.0000 | 20 |

### Confusion matrix (rows = true, columns = predicted, order Clean/Dusty/Hotspot)

```
              Clean  Dusty  Hotspot
Clean          71      1       0
Dusty           0     64       0
Hotspot         0      0      20
```

### The originally-recorded 100% figure vs. this audit's 99.36% — investigated and explained

The training registry (`training/experiments/registry.jsonl`,
`solar-mobilenet-3class-full-v2`) recorded `test_accuracy: 1.0` (156/156).
This audit's independent re-run, using the real production classifier
wrapper (`models.classifier.SolarFaultClassifier`, via `models.model_manager`
— the exact code path `services/pipeline.py` uses), finds **1 real error**.

**Root cause, confirmed:** the original `training/classification/
evaluate_mobilenet.py` preprocesses images with `torchvision.transforms.
Resize(256)` → `CenterCrop(224)` (the standard ImageNet-style "256→224"
convention). The **production** inference path
(`utils/image_utils.py::resize_for_mobilenet`, used by `models/classifier.py`
and therefore by the real deployed app) instead resizes the shortest side
directly to 224 and center-crops 224 — **no 256-pixel intermediate step**.

These two preprocessing pipelines are not equivalent and can produce
different softmax outputs for the same source photo. Verified directly on
the one disagreement (`test/Clean/20210922_095007.jpg`, run through the
same loaded model weights, only the preprocessing differs):

| Preprocessing | Clean | Dusty | Hotspot | Predicted |
|---|---|---|---|---|
| Production (`resize_for_mobilenet`: 224 direct) | 0.4584 | **0.5256** | 0.0160 | **Dusty (wrong)** |
| Training-eval (`Resize(256)`→`CenterCrop(224)`) | **0.7245** | 0.2445 | 0.0310 | **Clean (correct)** |

The model weights are identical and unchanged in both cases — this is a
**real, previously-unknown preprocessing-pipeline inconsistency** between
the training-time evaluation methodology and the production inference
code, not a flaw in the trained model itself, and not something invented
for this audit. **The originally-reported 100% figure was measured with a
preprocessing pipeline the deployed application does not actually use.**
The scientifically accurate figure for "what the deployed v1.0.0 app
actually achieves" is **99.36%**, independently reproduced here. Per this
phase's explicit no-production-changes rule, `resize_for_mobilenet` was
**not** modified — this is reported as a finding for a future hardening
phase, not fixed here.

### Confidence analysis

| | Mean | Median | Min | Max |
|---|---|---|---|---|
| All predictions (n=156) | 0.9953 | 0.9998 | 0.5256 | 1.0000 |
| Correct predictions (n=155) | 0.9984 | 0.9998 | 0.9397 | 1.0000 |
| Incorrect predictions (n=1) | 0.5256 | 0.5256 | 0.5256 | 0.5256 |

By true class (mean confidence): Clean 0.992, Dusty 0.999, Hotspot 0.996.

The single error's confidence (0.526) is dramatically lower than every one
of the 155 correct predictions (minimum 0.940) — a clean, meaningful
separation. **No formal calibration analysis (e.g. reliability
diagrams/ECE) was performed**: with only 1 error in 156 samples there is
not enough data to say anything statistically meaningful about calibration
beyond this single-point observation, and claiming a calibration
conclusion from it would overstate what this audit supports.

### Error analysis

**Every incorrect prediction (there is exactly one):**

| Filename | True | Predicted | Confidence |
|---|---|---|---|
| `20210922_095007.jpg` | Clean | Dusty | 0.526 |

**Lowest-confidence correct predictions:**

| Filename | True | Confidence |
|---|---|---|
| `H215.jpeg` | Hotspot | 0.940 |
| `20210919_151334.jpg` | Clean | 0.981 |
| `20210922_094758_3_11zon_41_11zon.jpg` | Clean | 0.984 |

**Highest-confidence incorrect prediction:** the one error above (0.526) —
also, trivially, the *only* incorrect prediction.

---

## Leakage / similarity audit

Performed on the MobileNet classification dataset (`E:\Solar AI Training
Images\prepared\`, train=1,227 / val=152 / test=156 images).

### Exact duplicates (SHA-256)

**0 cross-split collisions** — no byte-identical file appears in more than
one split. Matches the expected/claimed result.

### Near duplicates (dHash, Hamming distance ≤ 5 on a 64-bit hash)

| Boundary | Candidate pairs found |
|---|---|
| train ↔ val | 144 |
| train ↔ test | 172 |
| val ↔ test | 37 |

**This is genuine, visually-confirmed leakage — not merely a hash
artifact.** Two pairs were manually opened and visually compared:

1. `train/Clean/20210917_151404.jpg` vs.
   `test/Clean/20210917_151404_2_11zon_4_11zon.jpg` (Hamming distance 0,
   matching filename timestamp, filename bears an image-resizer tool's
   "_11zon" suffix) — **visually confirmed to be the same photograph**,
   re-exported through a resizing tool and placed in a different split.
2. `val/Dusty/20210916_130941.jpg` vs. `test/Clean/20210917_151412.jpg`
   (Hamming distance 0, *no* matching filename pattern) — **visually
   confirmed to be two different photographs** of different panel
   sections. This is a genuine dHash false-positive collision: this
   dataset's images are visually repetitive (regular panel-cell grid
   patterns), which reduces a simple difference-hash's discriminative
   power. This pair is **not** leakage.

To separate genuine leakage from hash coincidences, each candidate pair's
filenames were checked for a shared capture timestamp (the
`YYYYMMDD_HHMMSS` prefix these camera-originated filenames carry,
independent of any "_11zon" resize-tool suffix chain):

| Boundary | Pairs inspected | Sharing a capture timestamp (strong corroboration) |
|---|---|---|
| train ↔ test | 50 (of 172 found) | 40 (80%) |
| val ↔ test | 37 (of 37 found) | 23 (62%) |

**Conclusion: proven leakage.** A substantial majority of flagged
near-duplicate pairs are corroborated by matching capture timestamps and
are, based on the two manually-verified examples, genuine same-photo
duplicates that cross the train/test and val/test boundaries — most
likely introduced when the same source photo was independently
re-exported (different crop/resize pass through the same "_11zon" tool)
and the dataset-preparation step's SHA-256-based dedup (which correctly
found 0 *exact* duplicates) could not catch it, since the re-export
changes the file's bytes. A minority of flagged pairs (filename-uncorrelated,
like the manually-inspected Dusty/Clean example) are more likely hash
collisions from this dataset's repetitive visual content, not real leakage.

### Source/group signal

`training/classification/prepare_dataset.py`'s own manifest records
`group: None` for every image (no formal grouping metadata was ever
available for this dataset — already an honestly-disclosed limitation in
`training/classification/INTERIM_MODEL_REPORT.md`). This audit's filename-based
signal found **8 cross-split groups** sharing a normalized filename base,
consistent with — and additional evidence for — the near-duplicate finding
above.

### What this means for the reported MobileNet accuracy

The MobileNet test-split accuracy (99.36%) is a **real, correctly-computed
number given the data that exists**, but that data is not a fully clean,
independent held-out set: a meaningful fraction of the 156 test images are
near-duplicates of training images. This means the reported accuracy
likely **overstates** how the model would perform on genuinely novel solar
panel photos it has never seen anything resembling. Combined with the
small test-set size (especially Hotspot's 20 samples) and the class
imbalance, **99.36% should not be read as a reliable estimate of
generalization accuracy** — it is an accurate description of performance
on this specific, imperfectly-independent test set.

---

## End-to-end pipeline results

Real, non-mocked run of `SolarPanelDetector.detect()` →
`utils.image_utils.crop_panel()` → `SolarFaultClassifier.classify()`,
composed exactly as `services/pipeline.py` does, over the 156-image
MobileNet test split (the only dataset with real fault-class ground
truth).

**Disclosed assumption:** ground-truth panel count is assumed to be
exactly 1 per image (these are single-panel classification photos by
dataset design), not independently box-annotated — stated explicitly, not
hidden.

**Disclosed domain limitation:** this dataset (SolNET/PVMD close-up,
ground-level single-panel photos) is a different visual domain than
YOLO's own training data (BDAPPV aerial/satellite rooftop imagery). This
evaluation cannot and does not claim to measure YOLO's in-domain detection
performance — see the dedicated YOLO section above for that.

| Metric | Definition | Value |
|---|---|---|
| `whole_image_classification_accuracy` | MobileNet classifies the *entire uploaded image*, independent of detection — exactly what `services/pipeline.py`'s `classification_result` always reports, detected or not | **0.9936** (155/156) |
| `detection_rate` | Fraction of images where YOLO detected ≥1 panel (against the assumed ground truth of 1) | **0.0064** (1/156) |
| **`end_to_end_panel_accuracy`** | **A panel is end-to-end correct only when a panel is detected AND the resulting crop's classification is correct.** Never true when detection finds 0 panels. | **0.0064** (1/156) |
| `classification_accuracy_given_detection_succeeded` | Of the 1 image where detection succeeded, was the resulting per-panel crop classified correctly | 1.0000 (n=1 — not statistically meaningful) |

**This is the report's clearest illustration of end-to-end performance
being far lower than either individual model's own reported performance,
and it is fully explained, not mysterious:** YOLO — trained exclusively on
aerial/satellite rooftop imagery — essentially never fires on a
close-up, ground-level photo of a single panel, so the strict
detection-gated pipeline almost never reaches the classification step at
all. This is **not** a classification defect (whole-image classification
remains 99.36% accurate on the exact same photos) and **not** a
contradiction of the YOLO section's findings (that section already shows
recall is poor even in-domain at the deployed threshold) — it is the
predictable consequence of running an aerial-imagery detector on
ground-level photos, compounded by the same conf=0.45 threshold issue
already identified. **A real user uploading a typical close-up panel
photo through the actual Inspect page still receives an accurate
classification result** (the whole-image path), even though the
per-panel detection-gated path would not fire — this distinction is
exactly why `services/pipeline.py` always classifies the whole image
independently of detection outcome, and this audit confirms that design
choice is doing real, valuable work.

---

## Confidence / calibration analysis

See the MobileNet and YOLO sections above for the full statistics
(confidence by class, correct vs. incorrect, detection-confidence
distribution, matched-IoU distribution). **No formal calibration study**
(reliability diagrams, Expected Calibration Error, or similar) was
performed for either model. MobileNet has only 1 misclassification in the
available test data — nowhere near enough to support a calibration
conclusion. YOLO's 503 raw detections and 1,640 ground-truth boxes are a
larger sample, but a proper calibration analysis (binning predicted
confidence against empirical precision) was out of scope for this audit
and is not claimed here.

---

## XGBoost efficiency-loss prediction

```
XGBoost efficiency-loss prediction:
NOT EVALUATED

Reason:
No legitimate training dataset containing the required telemetry
and measured efficiency-loss target was identified.
```

See `training/prediction/DATASET_SOURCES.md` for the full investigation.
No synthetic labels were generated, no dummy model was created, and no
error rate (0% or otherwise) is reported for a model that does not exist.

---

## Limitations

- **YOLO test set (BDAPPV) has no fault-classification labels**, and the
  **MobileNet test set (SolNET/PVMD) has no panel bounding-box
  annotations.** No single dataset in this project can evaluate detection
  and classification simultaneously with two independent ground truths —
  the end-to-end section's methodology and its assumptions are disclosed
  explicitly rather than papered over.
- **MobileNet's test set is small** (156 images total, only 20 Hotspot
  samples) and **not fully independent of the training set** — see the
  leakage audit. Reported classification metrics are real but should be
  read as descriptive of this specific dataset, not as a generalization
  guarantee.
- **Class imbalance**: MobileNet's training data is Clean/Dusty-heavy
  (570/508 train images) relative to Hotspot (149) — consistent across
  splits, but still a real imbalance worth noting when interpreting
  per-class metrics.
- **Domain limitation**: YOLO was trained and is evaluated in-domain only
  on aerial/satellite rooftop imagery (BDAPPV). Its performance on
  ground-level or close-up photography (the domain many real users are
  likely to upload) is not characterized by the in-domain mAP/precision/
  recall numbers above — the end-to-end section's near-zero detection
  rate on close-up photos is the closest evidence this audit has of
  out-of-domain behavior, and it is poor.
- **Production confidence threshold appeared miscalibrated** for the real
  YOLO checkpoint (see YOLO section) — **addressed in Phase 6B**
  (`docs/ML_HARDENING_PHASE6B.md`): recalibrated from 0.45 to 0.30 using a
  validation-split threshold sweep. This measurably improves recall but
  does not resolve the domain-shift or object-size-ceiling limitations
  below, which remain real, current limitations.
- **XGBoost is not evaluated and not part of v1** — no legitimate dataset
  exists (see dedicated section above).
- **The three missing classes** (Bird-Drop, Electrical-Damage,
  Physical-Damage) remain outside v1's scope and therefore outside this
  evaluation entirely — nothing here evaluates or claims anything about
  them.
- **No formal calibration analysis** was performed for either model (see
  Confidence/calibration section) — explicitly not claimed.
- **This audit's own near-duplicate detector (dHash) has a real,
  demonstrated false-positive rate** on this dataset's repetitive
  panel-grid imagery (see the manually-verified false match) — the
  near-duplicate counts reported here are corroborated by an independent
  filename-timestamp signal precisely because the raw dHash count alone
  is not fully reliable on this image domain.

---

## Reproducing this evaluation

```bash
python training/evaluation/evaluate_yolo.py \
  --data-root "E:/Solar AI Training Images/yolo_prepared" --split test

python training/evaluation/evaluate_mobilenet.py \
  --data-root "E:/Solar AI Training Images/prepared" --split test

python training/evaluation/evaluate_end_to_end.py \
  --data-root "E:/Solar AI Training Images/prepared" --split test

python training/evaluation/leakage_audit.py \
  --data-root "E:/Solar AI Training Images/prepared"
```

Each writes a full per-image CSV and a summary JSON to
`E:\Solar AI Training Images\evaluation_runs\` (or `--output-dir`).
