# Phase 14B — evaluation protocol for a future individual-panel detector

Defined BEFORE any Phase 14C training happens, per Phase 14B req 20-21. This
is a specification only — no detector has been trained or evaluated under
it yet. The reusable metric helper referenced below
(`training.detection.ground_pv.representative_dataset.compute_count_error_stats`)
exists and is unit-tested, but has not been run against any real model
output because no held-out ground truth exists yet (see
`dataset_report.md`).

## 1. What must be reported

### 1a. Standard detection metrics (aggregate)
- Precision, Recall, mAP50, mAP50-95 — computed the same way the existing
  ground research detector evaluation does (`scripts/evaluate_ground_pv.py`),
  against the `test` split ONLY, never `train`/`valid`.

### 1b. Panel-COUNT accuracy (the actual product requirement)
The deployed product's requirement is "find every individual visible
panel", i.e. `panel_count = N`, not just "does an average box overlap a
ground-truth box". Aggregate mAP can look fine while systematically
missing whole panels in a subset of layouts — exactly the two failure
modes recorded in [[phase14a_dynamic_panel_detection]] (image 4, image 6).
So, per image:

```json
{"image_id": "...", "ground_truth_panel_count": 8, "predicted_panel_count": 7, "count_error": -1}
```

computed by `compute_count_error_stats(predictions, ground_truth)` — pure
python, image_id -> panel_count dicts in, per-image + aggregate stats out.
Only image_ids present in BOTH dicts are scored; a prediction with no
matching ground truth (or vice versa) is reported as a count, never
silently ignored or assumed to be a match.

Aggregate count metrics required:
- **exact count accuracy** — fraction of images where `predicted == ground_truth`.
- **mean absolute count error** — `mean(|predicted - ground_truth|)`.
- **mean (signed) count error** — reveals a systematic over/under-counting bias.
- **overcount rate** / **undercount rate** — fraction of images each direction.

### 1c. Category breakdown (never rely on aggregate mAP alone)
Report every metric above SEPARATELY for each of the following
non-exclusive categories (an image can be in more than one):

- 4-panel scenes, 6-panel scenes, 8-panel scenes (and any other observed count)
- large-foreground-panel scenes (tagged per `failure_cases.csv` /
  manifest `notes`, or a future dedicated manifest column)
- touching/grouped-panel scenes (same)
- high-perspective scenes
- portrait images, landscape images

Recall is the priority metric to watch per-category: the product
requirement is "find every panel", so a high-precision/low-recall detector
that reliably finds *some* panels but silently drops others in, say, all
8-panel scenes, is a category-specific failure aggregate mAP alone would
mask.

## 2. What ground truth is required before this protocol can run

- A `test`-split holdout with human-verified (`annotation_status ==
  "annotated_human_verified"`) individual-panel boxes — see the annotation
  policy in `training/detection/ground_pv/representative_dataset.py`'s
  module docstring (one physical panel = one box; ambiguous fragments not
  annotated).
- That holdout's scenes must never have appeared in `train`/`valid` — enforced
  by `verify_no_scene_crosses_splits` / the dataset audit's
  `cross_split_exact_duplicate_groups` / `cross_split_near_duplicate_pairs`
  checks (both must be 0).
- `ground_truth_panel_count` per test image derived directly from its
  verified label file's box count (`len(boxes)`), never from a model
  prediction, never estimated.

**None of this exists yet** — see `dataset_report.md`'s verdict. This
document defines the protocol so it is ready to apply the moment a genuine
holdout exists; it does not claim one does.

## 3. What this protocol explicitly forbids

- Never score against `train`/`valid` images and call it "evaluation".
- Never treat the CURRENT detector's own predictions as ground truth for
  scoring a FUTURE detector (that would just measure agreement between two
  models, not correctness).
- Never compute mAP as if it were sufficient on its own — panel-count
  accuracy must always be reported alongside it, per §1b.
- Never claim accuracy/precision/recall/mAP from the six-image qualitative
  reference set (`docs/DYNAMIC_PANEL_DETECTION.md` sec.7) — it has no
  verified ground truth and is explicitly qualitative-only.

## 4. Reusable code

`training/detection/ground_pv/representative_dataset.compute_count_error_stats`
— unit-tested (`tests/test_ground_pv_representative_dataset.py::TestCountErrorEvaluation`,
5 tests: exact match, over/undercount, signed-bias detection, unmatched
predictions/ground-truth handled without fabrication, empty-input edge
case). Ready to call once real predictions + real ground truth both exist.
