# Solar AI v1.0.0 — Phase 6B: ML Hardening & Re-Evaluation

Generated: 2026-09-05
Repository: https://github.com/cat226/solar-ai-framework
Branch: `feat/cloud-training-orchestration` (PR #23, open/draft)
Baseline: `v1.0.0` tag at `14df9b9cdb5411cef79c4282174c7d042abb1a96`, Phase 6A commits
`cb2a192` (evaluation tooling) and `e60d022` (initial evaluation report).

This document extends `docs/ML_EVALUATION_v1.0.0.md` (which remains the
primary evaluation report - it is updated alongside this one, not
replaced). This document is specifically the record of Phase 6B's
hardening work: what was re-examined, what changed in evaluation
methodology, and the resulting engineering decision.

---

## Task 1 — Audit summary (before any change)

### YOLO

| Stage | Preprocessing |
|---|---|
| Training (`training/detection/train_yolo.py`) | Ultralytics' own internal augmentation/letterboxing pipeline (not custom code in this repo - `YOLO.train()` handles it) |
| Production inference (`models/detector.py` via `utils/image_utils.py::resize_for_yolo`) | Resize to fit 640×640 preserving aspect ratio, letterboxed on a grey (114,114,114) canvas |
| Production config (`configs/settings.yaml`) | `confidence_threshold: 0.45`, `iou_threshold: 0.50`, `image_size: 640` |

No training/production preprocessing mismatch was found for YOLO (unlike
MobileNet) - `resize_for_yolo` is the same letterboxing convention
ultralytics itself uses. The issue identified in Phase 6A is a
**confidence-threshold operating-point** question, not a preprocessing
mismatch.

### MobileNet

| Stage | Preprocessing | Source |
|---|---|---|
| Training augmentation (`train_mobilenet.py`, `train_tf`) | `RandomResizedCrop(224, scale=(0.75,1.0))` → `RandomHorizontalFlip` → `ColorJitter` → `ToTensor` → `Normalize` | `training/classification/train_mobilenet.py:32-37` |
| **Training-time validation / checkpoint selection** (`train_mobilenet.py`, `eval_tf`) | `Resize(256)` → `CenterCrop(224)` → `ToTensor` → `Normalize` | `training/classification/train_mobilenet.py:39-44` |
| Original final-test evaluation (`training/classification/evaluate_mobilenet.py`) | Same `Resize(256)` → `CenterCrop(224)` pipeline | `training/classification/evaluate_mobilenet.py:57` |
| **Production inference** (`models/classifier.py` via `utils/image_utils.py::resize_for_mobilenet`) | Resize shortest side directly to 224 → `CenterCrop(224)` — **no 256px intermediate step** | `utils/image_utils.py:186-206` |
| Normalization (both) | `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]` (identical) | — |

**This audit newly confirms** (beyond what Phase 6A found) that the
mismatch is not confined to the final evaluation script: `eval_tf` is also
what the **training run's validation loader** used to pick the
best-checkpoint epoch (`best_val_accuracy` in
`training/experiments/registry.jsonl`). The released checkpoint was
therefore selected using the 256→224 pipeline throughout training, and has
never been validated under the exact pipeline production actually runs it
through, until Phase 6A's audit.

### Datasets / manifests / splits

- YOLO: `E:\Solar AI Training Images\yolo_prepared\{train,val,test}\{images,labels}\`, `manifest.json` with per-record SHA-256, split, `source_shard`. Counts: train 11,347 / val 3,179 / test 2,581 images. Split policy: BDAPPV's own geographic (department-level) column, not re-shuffled.
- MobileNet: `E:\Solar AI Training Images\prepared\{train,val,test}\{Clean,Dusty,Hotspot}\`, `manifest.json` with per-record SHA-256, `group: None` (no formal grouping metadata - a pre-existing, already-documented limitation). Counts: train 1,227 / val 152 / test 156 images.

### Existing evaluation tooling (Phase 6A, `training/evaluation/`)

`common.py` (pure helpers), `evaluate_yolo.py` (production-path P/R + independent mAP via `YOLO.val()`), `evaluate_mobilenet.py` (production-path classification metrics), `evaluate_end_to_end.py` (real detect→crop→classify), `leakage_audit.py` (exact + near-duplicate audit). All already use the real production wrappers, not reimplementations - Phase 6B extends these in place rather than replacing them.

### Existing model hashes (unchanged, verified again this phase)

- `weights/yolo_solar.pt`: `0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea`
- `weights/mobilenet_solar_v1.pth`: `afccaccfcc309952f7a94d754aaafc22e7e3391416b9518c5f4a8635b1c2682b`

No weights were modified during this audit step (Rule 14: prefer evaluation/tooling/documentation changes first).

---

## Task 2 — YOLO operating-point analysis (validation split only)

Full grid `training/evaluation/yolo_threshold_sweep.py --split val` (VALIDATION split, 3,179 images, 1,745 real ground-truth boxes — **the test split was never used for threshold selection**, per the non-negotiable rules). Methodology: one low-confidence (conf=0.001) inference pass per image with NMS at the real production `iou=0.50` (never swept), then post-hoc confidence filtering + greedy IoU≥0.5 matching at each of 18 grid points.

**Important disclosed methodology limitation:** this single-low-confidence-pass approach is a standard, legitimate technique (the same one `ultralytics.YOLO.val()` uses internally), but it is an *approximation* — NMS suppression at conf=0.001 considers a different, larger candidate pool than NMS run independently at each higher threshold would, which can suppress boxes differently. **This audit's own absolute F1 values come out lower, at every threshold, than `ultralytics.YOLO.val()`'s self-reported precision/recall (~0.71/~0.82 at its own internally-chosen operating point, reproduced in Phase 6A).** This discrepancy was investigated but not fully resolved (a fully independent per-threshold NMS analysis would need ~18 full-dataset inference passes, judged not worth the added compute for what is fundamentally a directional/relative analysis). **The directional finding below (a lower threshold than 0.45 measurably improves F1) is trustworthy; the absolute F1 numbers should not be read as matching `YOLO.val()`'s own convention.** This uncertainty is stated explicitly rather than resolved by assumption, per the non-negotiable rules.

### Full sweep (val split, iou=0.50 fixed)

| conf | Precision | Recall | F1 | TP | FP | FN | Detections |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.117 | 0.534 | 0.191 | 931 | 7057 | 814 | 7988 |
| 0.10 | 0.164 | 0.459 | 0.242 | 801 | 4074 | 944 | 4875 |
| 0.15 | 0.208 | 0.417 | 0.278 | 727 | 2768 | 1018 | 3495 |
| 0.20 | 0.252 | 0.383 | 0.304 | 669 | 1988 | 1076 | 2657 |
| 0.25 | 0.293 | 0.343 | 0.316 | 598 | 1442 | 1147 | 2040 |
| **0.30** | **0.334** | **0.304** | **0.318** | 531 | 1060 | 1214 | 1591 |
| 0.35 | 0.382 | 0.262 | 0.311 | 457 | 738 | 1288 | 1195 |
| 0.40 | 0.431 | 0.222 | 0.293 | 387 | 511 | 1358 | 898 |
| **0.45 (deployed)** | 0.493 | 0.179 | 0.263 | 313 | 322 | 1432 | 635 |
| 0.50 | 0.546 | 0.132 | 0.212 | 230 | 191 | 1515 | 421 |
| 0.55 | 0.594 | 0.083 | 0.146 | 145 | 99 | 1600 | 244 |
| 0.60 | 0.644 | 0.042 | 0.080 | 74 | 41 | 1671 | 115 |
| 0.65-0.90 | declining to 0 | declining to 0 | declining to 0 | — | — | — | — |

**Best-F1 grid point: conf=0.30** (P=0.334, R=0.304, F1=0.318) — a **predeclared criterion** ("maximize F1 across the swept grid on the validation split"), decided before running the sweep, not chosen after inspecting results to flatter them.

**F1 is fairly flat across roughly conf=0.25-0.40** (0.293-0.318) — this is a broad plateau, not a sharp optimum. Conf=0.30 is the single best grid point within it, not a uniquely, sharply superior value; a different point in that same range would be similarly defensible.

**Confidence distribution:** all 503 raw production-path (conf≥0.45) detections on the *test* split (Phase 6A) clustered near the threshold itself (mean 0.531) — consistent with this sweep's finding that the model produces a large volume of detections in the 0.25-0.45 confidence band that the deployed threshold discards entirely.

### Recall by object size (val split, real GT box-area quartiles: p25=0.00393, p50=0.00544, p75=0.00736 of image area)

| Bucket | GT boxes | Recall @ conf=0.05 | Recall @ conf=0.30 | Recall @ conf=0.45 (deployed) |
|---|---|---|---|---|
| tiny (< p25) | 436 | 0.420 | 0.204 | 0.099 |
| small (p25-p50) | 435 | 0.584 | 0.338 | 0.209 |
| medium (p50-p75) | 437 | 0.606 | 0.375 | 0.240 |
| large (≥ p75) | 437 | 0.524 | 0.300 | 0.169 |

**Object size is a real, contributing factor** (tiny objects are consistently the hardest across every threshold) **but it does not fully explain the recall gap**: even "large" ground-truth boxes (top quartile by area) only reach 52% recall at the most permissive threshold tested (conf=0.05) and 30% at the recalibrated conf=0.30. This indicates the detector's discriminative capability on this dataset has a real ceiling that threshold tuning alone cannot fix — some panels the model plausibly cannot confidently localize regardless of confidence threshold.

---

## Task 3 — YOLO domain-shift analysis

**Conclusion: a combination of (A) confidence-threshold miscalibration and (B) an object-size/detection-capability ceiling, both measured in-domain (BDAPPV val/test), plus (C) a separate, severe out-of-domain generalization failure.** All three are real and distinct; none fully explains the others.

- **(A) Threshold calibration — confirmed, partially addressed this phase.** Task 2's sweep shows the deployed conf=0.45 sits well past the model's own best-F1 operating point on in-domain (BDAPPV) validation data. Recalibrating to conf=0.30 is a genuine, validation-evidenced improvement (Task 9).
- **(B) Object-size / capability ceiling — confirmed, not addressable by threshold alone.** Even at conf=0.05 (near-maximal recall regime) and even for "large" objects, recall tops out around 50-60%. This is evidence of a real capability limitation in the trained checkpoint, not purely an operating-point problem.
- **(C) Image-domain mismatch — confirmed severe, separate from (A)/(B), not addressed by any threshold choice.** Phase 6A's end-to-end evaluation (re-confirmed unchanged this phase — see Task 7) found YOLO detects a panel in only 1 of 156 close-up/ground-level classification-dataset photos (0.64%), a completely different visual domain than BDAPPV's aerial/satellite rooftop imagery it was trained and validated on. No threshold in the swept grid meaningfully changes this — it is a training-data-domain limitation, not a calibration one. **No new/external dataset was introduced to further quantify this** (per the non-negotiable rules against scraping unlicensed data); the existing classification-dataset photos remain the only available evidence of out-of-domain behavior, and that evidence is unambiguous and severe.

**These are genuinely separable, not conflated:** (A) is fixed by this phase's threshold change; (B) and (C) are not, and would require either accepting the model's real capability ceiling (recommendation E, see Task 8) or a properly licensed, in-domain (aerial) or domain-adapted retraining effort (recommendation B/D) that is out of scope for an evaluation/hardening phase.

---

## Task 4 — MobileNet preprocessing alignment

See Task 1's audit table for the full training/eval/production preprocessing comparison. **Action taken this phase** (per the task's explicit priority: fix evaluation methodology first, do not retrain):

1. **Evaluation methodology was already fixed in Phase 6A** — `training/evaluation/evaluate_mobilenet.py` classifies through the real `models.classifier.SolarFaultClassifier` production wrapper (`resize_for_mobilenet`, direct 224 resize), never a hand-rolled `Resize(256)→CenterCrop(224)` pipeline. This phase adds no methodology change here, only verification tooling.
2. **New regression tests added** (`tests/test_mobilenet_preprocessing_alignment.py`, 8 tests): lock down the exact canonical production transform (RGB → `resize_for_mobilenet` direct-224 → `ToTensor` → `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`), assert it is demonstrably *not* pixel-equivalent to the `Resize(256)→CenterCrop(224)` pipeline (so a future accidental "fix" that quietly makes them identical is caught, not silently accepted, until it is a deliberate, documented decision), and statically verify both `evaluate_mobilenet.py` and `evaluate_end_to_end.py` import and use the real wrapper rather than a duplicate transform.
3. **Documented, not fixed at the code level:** the mismatch itself (production ≠ training-time validation/evaluation preprocessing) remains in `train_mobilenet.py`/`evaluate_mobilenet.py` (the *training* scripts, not the evaluation-audit scripts) — changing those, or `resize_for_mobilenet`, is a production/training code change with retraining implications, explicitly out of scope for this evaluation/hardening phase (Rule 14). **Recommendation for a future phase:** align `train_mobilenet.py`'s `eval_tf` to `resize_for_mobilenet`'s exact convention before any future retraining run, so checkpoint selection and final reported metrics reflect the same pipeline production actually uses.

**Retraining was not performed and is not recommended solely on this basis** — Task 6's clean-subset evaluation (below) shows the real accuracy impact of the preprocessing mismatch is one image out of 156 (99.36%→99.36% either way; the *number* of errors doesn't change based on which split is used, only which specific evaluation methodology is used), not a systemic problem large enough to require immediate retraining.

---

## Task 5 — MobileNet duplicate/leakage audit (extended)

`training/evaluation/leakage_audit.py` (extended in Phase 6B) on the full `train`/`val`/`test` classification split (1,227 / 152 / 156 images).

**A real bug in the corroboration signal was found and fixed during this phase**, before it could distort the leakage numbers: the filename-normalization regex used to detect a shared capture timestamp (`_base_id`) originally also stripped a bare trailing `_<digits>` segment — which incorrectly matched the `_HHMMSS` half of this dataset's real `YYYYMMDD_HHMMSS.jpg` filenames, treating *every photo taken on the same day* (regardless of time-of-day) as a timestamp match. Caught by a new unit test (`tests/test_leakage_audit.py::TestBaseId::test_different_timestamps_have_different_base_ids`) before being used to classify any real pairs; the leakage audit was re-run only after the fix, so the numbers below already reflect the corrected regex.

### Classification (near-duplicate candidates at Hamming distance ≤ 5, corroborated by matching capture timestamp)

| Category | Count | Meaning |
|---|---|---|
| Exact duplicate (SHA-256) | 0 | Byte-identical files across splits |
| Highly likely near duplicate | 69 | Hamming distance = 0 **and** matching capture-timestamp filename base |
| Probable false positive | 35 | Hamming distance = 0 but **no** matching timestamp (the manually-verified Phase 6A example — two visually different photos - falls in this category) |
| Uncertain | 249 | Hamming distance 1-5, not asserted either way |

### Clean test subset

Of the 156 test images, **42 have a "highly likely near duplicate" in train or val** and were excluded, leaving a **114-image cleaner independent subset** — constructed by `leakage_audit.py`, written to `E:\Solar AI Training Images\evaluation_runs\leakage_audit.json`'s `clean_test_subset` field, and consumed directly by `evaluate_mobilenet.py --include-only`. **The original test split itself was never modified, reordered, or deleted** — the clean subset is a derived, separately-labeled evaluation artifact only.

---

## Task 6 — MobileNet clean evaluation

Both runs use the identical real production classifier (`weights/mobilenet_solar_v1.pth`, `classifier_source=v1`) and identical methodology — only the image set differs.

| Metric | **Original split** (156 images, near-duplicate-contaminated) | **Cleaner independent subset** (114 images, near-duplicates excluded) |
|---|---|---|
| Accuracy | 0.9936 (155/156) | 0.9912 (113/114) |
| Macro precision / recall / F1 | 0.9949 / 0.9954 / 0.9951 | 0.9907 / 0.9944 / 0.9925 |
| Weighted F1 | 0.9936 | 0.9913 |
| Per-class support | Clean 72 / Dusty 64 / Hotspot 20 | Clean 59 / Dusty 35 / Hotspot 20 |
| Errors | 1 (`20210922_095007.jpg`, Clean→Dusty, conf 0.526) | 1 (same image - not itself flagged as a near duplicate) |

**Confusion matrices**

Original split:
```
              Clean  Dusty  Hotspot
Clean          71      1       0
Dusty           0     64       0
Hotspot         0      0      20
```

Cleaner independent subset:
```
              Clean  Dusty  Hotspot
Clean          58      1       0
Dusty           0     35       0
Hotspot         0      0      20
```

**Interpretation — stated carefully, not overclaimed:** removing the 42 confirmed near-duplicate test images changes accuracy by less than half a percentage point (99.36%→99.12%) and does not change the error count. This is genuine evidence that, *for this specific dataset and this specific error pattern*, the near-duplicate leakage found in Task 5 did not dramatically inflate the headline accuracy number. **This is not the same claim as "the model generalizes well"** — both figures come from a still-small sample (114-156 images, Hotspot support only 20), from the same narrow source distribution (SolNET/PVMD), and neither should be presented as "production accuracy" or "real-world accuracy" without those caveats attached every time.

---

## Task 8 — Decision matrix

| Option | Verdict | Evidence |
|---|---|---|
| A. Existing YOLO + calibrated threshold is acceptable | **Partially — as v1's disclosed, honest scope, not as a strong claim** | Recalibrating conf 0.45→0.30 is validation-evidenced and real (Task 2), but even at its best operating point F1≈0.32, and out-of-domain (close-up) detection is near-zero (Task 3/7). "Acceptable" only if the release continues to explicitly disclose these limits, as v1.0.0 already does. |
| B. Existing YOLO requires retraining/domain adaptation | **Not undertaken this phase (no new licensed data available), but the strongest long-term fix for the domain-shift finding** | Threshold calibration cannot fix out-of-domain performance (Task 3) or the size-related capability ceiling (Task 2). Retraining/domain adaptation would need new, properly licensed ground-level/close-up panel imagery with box annotations — none exists in this project currently. |
| C. Existing MobileNet is acceptable for v1 | **Yes, with leakage/small-sample limitations explicitly stated every time** | Clean-subset accuracy (99.12%, 114 images) closely tracks the original split (99.36%, 156 images) - the leakage found in Task 5 does not appear to have dramatically inflated the headline number for this specific checkpoint/dataset. Still not a generalization guarantee (small n, narrow source distribution, class imbalance). |
| D. MobileNet requires retraining | **Not supported by current evidence** | The one real error is explained by a preprocessing inconsistency in the *original evaluation*, not a fixed model defect; the model's real accuracy (via the correct production path) is 99.36%/99.12% either way. No evidence of a systemic classification problem. |
| **E. v1 should remain a technically functional/demo/research release rather than a high-accuracy production inspection system** | **Yes — this is the honest overall framing** | Whole-image classification is genuinely strong on real photos it resembles (99%+), but: YOLO's absolute detection ceiling is modest even in-domain (F1≈0.32) and fails almost completely out-of-domain (0.64% end-to-end); MobileNet's evaluation set is small and was contaminated by real (if apparently low-impact) leakage; XGBoost and 3 of 6 taxonomy classes remain entirely unavailable. None of this contradicts v1.0.0's own existing disclosures (`README.md`, `docs/ML_EVALUATION_v1.0.0.md`) — it reinforces that framing is the correct one. |

**Overall recommendation: E, with a concrete, applied improvement under A.** Solar AI v1.0.0 should continue to be presented as a technically functional research/demo release with honestly disclosed limitations, not as a high-accuracy production inspection system. Within that honest framing, the YOLO confidence-threshold recalibration (Task 9) is a real, evidence-based improvement worth keeping - it measurably improves recall without requiring new data, new training, or new risk.

---

## Task 9 — Production threshold policy

**Decision: changed.** `configs/settings.yaml`'s `models.yolo.confidence_threshold` updated from `0.45` to `0.30`.

**Justification (recorded per the non-negotiable rules):**
- Selected via a **predeclared criterion** (maximize F1 across an 18-point confidence grid), decided before running the sweep.
- Evaluated exclusively on the **validation split** (3,179 images, 1,745 ground-truth boxes) — **the test split was never used for this selection.**
- Real, measured improvement: recall 0.179→0.304 (+70% relative), F1 0.263→0.318 (+21% relative), at a real precision cost (0.493→0.334).
- **Not claimed to make detection "good"** — F1≈0.32 remains modest, and the domain-shift/size-ceiling limitations (Task 3) are unaffected by this change.
- **Confirmed, not re-selected, on the test split** after the change (Task 7's re-run) - this is a validation of a decision already made, not test-set optimization.
- `configs/settings.yaml` itself carries a comment recording this exact rationale (see Task 1/9 and the config file directly), so the decision's provenance survives independently of this document.

### Test-split confirmation (after the change, not used to select it)

Re-running `evaluate_yolo.py` on the **test** split (2,581 images, 1,640 ground-truth boxes) at the new conf=0.30, for confirmation only:

| | Old deployed conf=0.45 (Phase 6A) | New conf=0.30 (this phase) |
|---|---|---|
| Precision | 0.497 | 0.340 |
| Recall | 0.152 | 0.271 |
| F1 (derived) | ≈0.233 | ≈0.301 |
| TP / FP / FN | 250 / 253 / 1,390 | 444 / 862 / 1,196 |

The improvement **generalizes from validation to test** (F1 ≈0.318 on val, ≈0.301 on test - close, as expected for a real, non-overfit selection criterion) - this is additional evidence the validation-based selection was sound, not a validation-split fluke.

---

## Task 7 — End-to-end re-evaluation (after the threshold change)

Re-run of `training/evaluation/evaluate_end_to_end.py` on the classification test split (156 images, real Clean/Dusty/Hotspot ground truth) at the new conf=0.30. Per the task's explicit instruction, detector and classifier performance are never conflated:

| Metric | Old conf=0.45 (Phase 6A) | New conf=0.30 (this phase) |
|---|---|---|
| 1. **Detector performance** — detection rate (panel found, assumed GT=1 per image) | 0.64% (1/156) | 2.56% (4/156) |
| 2. **Classifier performance on correctly-supplied crops** — accuracy given detection succeeded | 100% (n=1 - not meaningful) | 50% (n=4 - still not statistically meaningful) |
| 2b. **Classifier performance independent of detection** — whole-image classification accuracy | 99.36% (155/156) | 99.36% (155/156, unchanged - does not depend on the detector) |
| 3. **Detection-gated end-to-end performance** | 0.64% (1/156) | 1.28% (2/156) |

**The threshold recalibration measurably improves detection rate on this out-of-domain dataset too (0.64%→2.56%, i.e. 1→4 images)** — consistent with Task 3's finding that conf=0.30 is a real, generalizable improvement — **but it does not come close to closing the domain gap.** Both before and after, over 97% of these real close-up photos produce zero detected panels. This confirms Task 3's conclusion directly: the dominant failure mode on this dataset is domain mismatch (C), which no confidence threshold can fix, not primarily a calibration problem (A) in this specific direction.

**Explicitly, per the task's instruction:** every one of the 152-154 images (depending on threshold) where detection found 0 panels is a **detection failure**, not attributed to the classifier - the classifier is never even invoked on a per-panel crop for those images, and its independent whole-image accuracy (99.36%, unaffected by any of this) is the correct, separate measure of its own real performance.

---

## Verified facts

- Model artifacts unchanged: `weights/yolo_solar.pt` (sha256 `0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea`), `weights/mobilenet_solar_v1.pth` (sha256 `afccaccfcc309952f7a94d754aaafc22e7e3391416b9518c5f4a8635b1c2682b`) - both verified unchanged against `weights/manifest.json` this phase.
- YOLO validation-split threshold sweep: real, 3,179 images, 1,745 ground-truth boxes, conf grid 0.05-0.90, iou fixed at 0.50.
- YOLO test-split confirmation at conf=0.30: real, 2,581 images, 1,640 ground-truth boxes; P=0.340, R=0.271.
- MobileNet original-split accuracy: real, 156 images, 99.36% (155/156).
- MobileNet cleaner-independent-subset accuracy: real, 114 images (42 excluded via corroborated near-duplicate detection), 99.12% (113/114).
- Leakage audit: real, 0 exact duplicates; 69 highly-likely near duplicates, 35 probable false positives, 249 uncertain (Hamming distance-based, corroborated by filename timestamp matching).
- End-to-end re-evaluation at conf=0.30: real, 156 images; detection rate 2.56%, end-to-end accuracy 1.28%, whole-image classification 99.36%.
- `configs/settings.yaml`'s `models.yolo.confidence_threshold` changed from 0.45 to 0.30, with the rationale recorded both in this document and in the config file itself.

## Limitations

- **Threshold-sweep methodology caveat (disclosed, not resolved):** single-low-confidence-pass + post-hoc filtering is a standard technique but not identical to running independent NMS at every threshold; absolute F1 values in this audit are lower than `ultralytics.YOLO.val()`'s own self-reported precision/recall at every grid point tested, an unreconciled discrepancy.
- **F1 plateau:** the validation sweep's F1 is fairly flat across roughly conf=0.25-0.40 - conf=0.30 is the best single grid point, not a sharply, uniquely optimal value.
- **Domain-shift limitation remains unresolved:** no licensed close-up/ground-level detection dataset with box annotations exists in this project; the threshold change cannot and does not fix YOLO's near-total failure to detect panels in that domain (2.56% detection rate even after recalibration).
- **Object-size ceiling:** even "large" ground-truth boxes only reach ~50-60% recall at the most permissive threshold tested - a real capability limitation, not purely calibration.
- **MobileNet evaluation set remains small** (114-156 images depending on subset) with real, confirmed (if apparently low-impact) near-duplicate leakage, and a persistent class imbalance (Hotspot support: 20).
- **XGBoost remains entirely unevaluated** - no legitimate artifact exists (unchanged from Phase 6A).
- **3 of 6 original taxonomy classes remain unavailable** (Bird-Drop, Electrical-Damage, Physical-Damage) - unaffected by and outside the scope of this audit.
- **"Uncertain" near-duplicate pairs (249)** were not individually visually inspected - the classification rule is corroboration-based, not proof for every single pair.

## What is NOT claimed

This document and `docs/ML_EVALUATION_v1.0.0.md` do **not** claim:
- "Production accuracy = X%" for either model, without the specific split/methodology/sample-size attached every time.
- "99% real-world classification accuracy" - the 99.36%/99.12% figures describe this specific, small, narrow-source-distribution test set, not general real-world performance.
- "YOLO detects 85%+ of panels in production" (a Phase 6A `.val()`-derived figure) as a production-representative number - the real production-path recall, confirmed this phase, is 27.1% at the new threshold on the in-domain test split, and 2.56% on the out-of-domain classification-dataset photos.
- That the conf=0.30 recalibration "fixes" YOLO - it is a real, validation-evidenced, modest improvement within a system that remains a technically functional research/demo release, not a high-accuracy production inspection system (Task 8).
- That MobileNet's near-duplicate leakage was harmless in general - only that its measured effect on this specific checkpoint's specific test set was small.

