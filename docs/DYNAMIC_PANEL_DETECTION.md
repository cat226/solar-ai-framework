# Dynamic multi-panel detection (Phase 14A)

Research-only architecture prototype. Branch `feat/ground-pv-research-detector`,
not merged, not deployed. Trains nothing and modifies no checkpoint — pure
inference orchestration over the existing
`weights/yolo_solar_ground_research_v1.pt` (unchanged, verified below).

## 1. Why this phase exists

The deployment concept changed: a camera mounted near a solar array is no
longer assumed to see exactly 4 panels. A single capture may show 4, 6, 8,
or some other count, in a single row, single column, or grid, at whatever
angle the camera happens to be mounted at.

**The system does not assume a fixed number of panels.** `panel_count` is
always derived from however many `panel`-class detections the checkpoint
returns above the confidence threshold for *this* image — nothing in
`training/detection/ground_pv/dynamic_panel_detection.py` branches on a
specific count, and nothing constructs a fixed `P1..P4` schema. The same
code path handles 0, 1, 2, 8, or 20 detections identically.

## 2. Architecture

```
FULL CAMERA IMAGE
      -> PANEL DETECTOR (yolo_solar_ground_research_v1.pt, unmodified)
      -> N raw "panel"-class detections (other classes explicitly filtered out)
      -> geometry-based spatial ordering -> P1 .. PN
      -> geometry / overlap sanity diagnostics (valid vs suspicious, per panel)
      -> JSON report + optional annotated image
      -> (future phase) condition model per panel
```

Module: `training/detection/ground_pv/dynamic_panel_detection.py`.
CLI: `scripts/detect_dynamic_panels.py`.
Unit tests: `tests/test_ground_pv_dynamic_panel_detection.py` (37 tests, no
GPU — geometry/ordering is pure python; the model call is a thin,
duck-typed wrapper exercised with a fake detector, matching the existing
convention in `tiled_inference.py` / `test_ground_pv_tiled_inference.py`).

This phase is **panel localization and panel identity/order only**. No
`CLEAN` / `DUSTY` / `BIRD_DROP` / `PHYSICAL_DAMAGE` / `ELECTRICAL_DAMAGE`
condition label is produced anywhere in this module. That is a separate
concern (`training/detection/ground_pv/panel_condition.py`) that operates
*after* panels exist, on a separate (and much weaker — see
[[ground_pv_condition_model_baseline]]) research checkpoint, and is
deliberately not wired into this pipeline in this phase.

## 3. Spatial ordering algorithm

Panel IDs are assigned deterministically from geometry, never from
whatever order the detector happened to emit boxes in:

1. Compute each panel's center `(center_x, center_y)` in pixel space.
2. Normalize centers relative to image width/height.
3. **Sort all detections by normalized `center_y` ascending, first.** This
   single sort is what makes every following step (and thus the final
   `P1..PN` assignment) independent of the detector's original output
   order — a hard requirement, proven by
   `TestOrderingIndependentOfInputOrder` (shuffles the same 8 detections
   20 times and asserts an identical result every time; also checks the
   reversed-list case and the full report end-to-end).
4. Greedily cluster into rows: a detection joins the current row if its
   normalized `center_y` is within `row_grouping_tolerance` (default 0.10,
   i.e. 10% of image height) of that row's running mean `center_y`;
   otherwise it starts a new row. This tolerates perspective skew — it does
   **not** require identical `y` across a row.
5. Sort panels within each row left-to-right by normalized `center_x`.
6. Assign `P1..PN` sequentially, row-major.

Nothing hard-codes a 2x2 / 2x3 / 4x2 layout. A single row (all one
cluster), a single column (N clusters of size 1), or an irregular grid all
fall out of the same clustering step — verified for 2x2, 2x3, 2x4, 4x2,
single-row, single-column, and perspective-shifted-row synthetic layouts in
`TestLayouts`.

**Ambiguous geometry is flagged, not hidden.** If two adjacent row
clusters' vertical pixel extents actually overlap (a "row" wasn't a clean
concept for this layout), `geometry.ordering_status` is set to
`"ambiguous"` instead of `"stable"`, and every panel in that image picks up
a `suspicious_row_geometry` warning — the report never silently presents
uncertain ordering as certain.

## 4. Diagnostics — valid vs suspicious, never silently deleted

A post-NMS sanity layer runs *after* the model's own NMS (YOLO IoU=0.50 by
default) and *after* spatial ordering, so warnings can reference final
`Pn` ids. It never removes a detection — it only annotates:

- **Overlap / duplicate suspects** — `high_overlap_with_P{n}` when two
  panels' IoU ≥ `overlap_warning_iou` (default 0.30) **or** one box is
  ≥ `containment_warning_fraction` (default 0.80) contained in the other.
  Threshold is deliberately lower than the model's own NMS IoU (0.50) so it
  catches boxes that *survived* NMS but still overlap substantially.
- **Extremely tiny / large panel** — relative to image area
  (`min_relative_area` 0.001 / `max_relative_area` 0.60, both configurable).
- **Extreme aspect ratio** — `< min_aspect_ratio` (0.20) or
  `> max_aspect_ratio` (5.0).
- **Size inconsistency** — a panel's area more than
  `size_inconsistency_ratio` (default 3x) away from the *median* panel area
  in that same image (needs ≥ 2 panels).
- **Isolated detection** — a panel whose nearest-neighbor normalized-center
  distance exceeds `isolation_distance_multiplier` (default 3x) times the
  array's median nearest-neighbor distance (needs ≥ 2 panels).
- **Suspicious row geometry** — see §3.

Every threshold above is a field on `DynamicPanelConfig`
(`training/detection/ground_pv/dynamic_panel_detection.py`) and a CLI flag
on `scripts/detect_dynamic_panels.py` — nothing is a scattered magic number.

Per-panel `status` is `"valid"` (no warnings) or `"suspicious"` (≥ 1
warning). Image-level `analysis_status` is `"ok"` (all valid),
`"warning"` (≥ 1 suspicious panel), or `"failed"` (zero accepted
detections — see §5).

## 5. Zero-detection and low-confidence handling

Zero accepted panel detections is reported as an explicit failure to
detect:

```json
{"panel_count": 0, "analysis_status": "failed", "failure_reason": "no_panel_detected"}
```

This is **never** reinterpreted as "no solar panels exist in the image."
Confidence is exposed on every accepted panel; a detection below
`confidence_threshold` never enters the accepted list (Ultralytics' own
`conf=` filtering at inference time), and nothing here silently raises a
low-confidence box's apparent status.

## 6. Limitations

- **Panel IDs are spatial identities for a single capture only.** `P1..PN`
  is reassigned fresh from geometry every time this pipeline runs on an
  image. It is **not yet guaranteed that "P3" in one capture is the same
  physical panel as "P3" in a later capture** — the camera could shift, a
  panel could leave the frame, lighting/occlusion could change which
  detections clear the confidence threshold. **Persistent physical panel
  identity across time is a separate, unsolved problem for the next
  phase**, not something this module claims to provide.
- **This does not fix the underlying detector's accuracy.** The geometry
  layer can only order and diagnose what
  `yolo_solar_ground_research_v1.pt` actually detects. If the detector
  misses a panel outright (see image 4 in §7 below), no amount of
  post-processing recovers it.
- **Moderate-overlap "group boxes" are not reliably caught by the overlap
  diagnostic.** When the detector emits two large boxes that each span
  several physical panels (the known image-6 failure mode — see
  [[ground_pv_tiled_inference_experiment]]), their IoU/containment can
  still fall *below* the overlap-warning thresholds if the group boxes are
  large and only partially overlapping. The diagnostic is tuned to catch
  clearly-duplicate boxes on the *same* panel, not to detect that a box
  is semantically wrong (covers multiple physical panels) — that is a
  detector-quality problem, not something geometry alone can fix.
- **Row-clustering assumes roughly planar, roughly axis-aligned rows.** A
  camera looking at a strongly non-planar or heavily-foreshortened array
  may produce a `row_grouping_tolerance` that is too tight (rows falsely
  split) or too loose (physically distinct rows falsely merged, and/or
  flagged `ambiguous`). The tolerance is configurable per-deployment but
  is not auto-tuned.
- **No condition classification.** This module answers "where are the
  panels and what order are they in", not "what condition are they in".

## 7. Six real-world images — qualitative smoke test only

**This is NOT a ground-truth evaluation.** No manually-verified panel
bounding boxes exist for these six owner-provided photographs, so **no
accuracy / precision / recall / F1 / mAP is computed or claimed** — a
"panel_count" that looks plausible is not scored as "correct". This is
purely a qualitative check that the new architecture runs correctly and
behaves sensibly on real, out-of-distribution images (the same domain-gap
context recorded in [[ground_pv_research_detector]] real-world holdout and
[[ground_pv_detector_v2_experiment]]).

Command:

```
python scripts/detect_dynamic_panels.py \
    --source "D:\solar_ai_training_scratch\Solar Real Images" \
    --weights weights/yolo_solar_ground_research_v1.pt \
    --output-dir research_runs/ground_pv/dynamic_detection_real_world \
    --save-annotated
```

(default `--conf 0.45`, matching the documented real-world-holdout
convention — not lowered to inflate the panel count.)

| # | filename (…suffix) | panel_count | confidences | analysis_status | geometry |
|---|---|---|---|---|---|
| 1 | …56 | 1 | 0.874 | **warning** | 1 row, stable |
| 2 | …57 (1) | 2 | 0.505, 0.866 | **warning** | 2 rows, **ambiguous** |
| 3 | …57 | 2 | 0.848, 0.676 | ok | 1 row, stable |
| 4 | …58 | 1 | 0.580 | ok | 1 row, stable |
| 5 | …59 (1) | 2 | 0.635, 0.589 | **warning** | 2 rows, **ambiguous** |
| 6 | …59 (landscape) | 2 | 0.665, 0.750 | ok | 1 row, stable |

Totals: 10 accepted `panel`-class detections across 6 images, 0 images with
zero detections (`analysis_status="failed"` never triggered on this set) —
consistent with the earlier full-frame holdout in
[[ground_pv_research_detector]] (same checkpoint, same conf=0.45, same 10
panel detections; this run additionally orders and diagnoses them).

**Known difficult cases:**

- **Image 4 (large foreground panel):** `panel_count=1`, and that one
  detection is the thin *background* strip (bbox y≈0–154 of a 1280px-tall
  frame) — the large panel filling most of the frame is **still not
  detected**, exactly the pre-existing failure mode recorded in
  [[ground_pv_research_detector]] / [[ground_pv_tiled_inference_experiment]].
  The dynamic pipeline correctly reports 1 valid, ordered panel; it cannot
  invent a detection the underlying model never produced. **The detector,
  not the ordering/diagnostics layer, is the bottleneck here.**
- **Image 6 (overlapping/group-like boxes):** `panel_count=2`, both marked
  `valid`, single stable row, **no overlap warning fired**. The two boxes
  do overlap (IoU ≈ 0.11, one ≈17%/27% contained in the other) but stay
  below the 0.30 IoU / 0.80 containment warning thresholds — each box still
  plausibly spans a *group* of adjacent physical panels rather than one
  panel each, matching the pre-existing "group box" failure mode. This is
  the clearest example in this limitations section: the geometry layer
  ordered and reported what was given to it correctly, but it cannot
  detect that a semantically-wrong box (one box = several physical panels)
  is wrong — see §6.
- **Images 1, 2, 5 (portrait crops with a large background box):** flagged
  `warning` — `extremely_large_area` and/or `size_inconsistent_with_array`
  fired correctly on a box spanning most of the frame's height (a
  background row of panels boxed as a single large detection) alongside a
  much smaller foreground box, and two of the three additionally tripped
  `suspicious_row_geometry` (`ordering_status="ambiguous"`) because that
  large box's vertical extent overlaps the smaller box's row. This is the
  diagnostics layer working as intended: it surfaces a plausibly-wrong
  detection instead of silently presenting it as `valid`.
- **Images 1 and 3's previously-noted `cracked` false positives are
  out of scope for this phase** — this pipeline only accepts the `panel`
  class; `cracked`/`dusty` detections are explicitly filtered out before
  any of the above analysis runs (see req A).

Per-image JSON + annotated images (gitignored, not committed):
`research_runs/ground_pv/dynamic_detection_real_world/`.

## 8. Difference between panel detection and condition analysis

| | This module (Phase 14A) | `panel_condition.py` (existing, separate) |
|---|---|---|
| Question answered | "Where are the panels, and in what order?" | "What defects/condition does each panel show?" |
| Model | `yolo_solar_ground_research_v1.pt` (3-class: panel/cracked/dusty) | Same detector's non-panel classes, or the separate `yolo_solar_condition_research_v1.pt` (5-class, [[ground_pv_condition_model_baseline]], underfit / research-only) |
| Output | `P1..PN` with bbox, geometry, valid/suspicious status | `Clean` / `Dust` / `BirdDrop` / `Damage` per panel |
| Assumes panels already exist? | No — this IS the panel-existence step | Yes — takes panel detections as input |
| Wired together in this phase? | **No, deliberately not** | N/A |

## 9. Production-model safety

Verified before and after this work:

- `weights/yolo_solar.pt` — production aerial detector — **untouched**.
- `weights/yolo_solar_ground_research_v1.pt` — the checkpoint this module
  calls — **untouched, read-only**. No training, no fine-tuning, no
  retraining occurs anywhere in this phase.
- `weights/manifest.json` — **untouched**.
- No dataset downloaded, modified, or retrained against. No Kaggle job
  launched.

## 10. What this phase does NOT claim

This phase establishes the **software architecture** for handling an
arbitrary, dynamically-detected panel count with deterministic spatial
ordering. It does **not** claim to have solved real-world panel detection:
the underlying checkpoint still misses large foreground panels (image 4)
and still emits group-like boxes on wide array shots (image 6) exactly as
before. The next material improvement is collecting representative
real-deployment-domain images and training a better individual-panel
detector (see §11), not further post-processing of the current model's
output.

## 11. Recommended Phase 14B data requirements

Based on the six-image qualitative smoke test and the accumulated findings
in [[ground_pv_research_detector]] / [[ground_pv_detector_v2_experiment]] /
[[ground_pv_phase12_data_acquisition_audit]]:

1. **Non-square, camera-realistic aspect ratios.** The training set is
   100% square 640×640 (Roboflow stretch-resized —
   [[ground_pv_detector_v2_experiment]]); all six real photos are portrait
   ~1:2.2 or landscape ~2.2:1. A representative dataset needs native
   aspect-ratio images from a camera mount similar to the deployment case.
2. **Large-foreground-panel examples with a genuine visible border**, not
   the degenerate whole-frame boxes already in the current data (noted in
   [[ground_pv_detector_v2_experiment]]'s scale-failure analysis) — to fix
   the image-4-style miss.
3. **Distant, multi-panel array shots with individually-separable panels**
   (not close-up crops mislabeled as "wide array"), so the detector learns
   panel *boundaries* within a row rather than grouping adjacent panels
   into one box — to fix the image-6-style group-box problem.
4. **A held-out set with manually-verified panel bounding boxes** for the
   deployment camera's actual mounting geometry, so a *real* accuracy /
   precision / recall / mAP evaluation becomes possible — everything in
   this document and in the prior real-world holdout report remains
   qualitative-only until that exists.
5. Per [[ground_pv_phase12_data_acquisition_audit]], no suitable
   Tier A/B public dataset was found for this specific domain gap; (1)-(4)
   most plausibly require **permissioned first-party collection** at or
   near the actual deployment site(s), not another public-dataset search.
