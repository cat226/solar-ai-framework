# Four-class panel condition pipeline (Phase 15A)

Research-only architecture and data pipeline. Branch
`feat/ground-pv-research-detector`, not merged, not deployed. No
production checkpoint modified.

## 1. Architecture

```
FULL CAMERA IMAGE
      -> dynamic panel detector (Phase 14A, weights/yolo_solar_ground_research_v1.pt, unmodified)
      -> P1 .. PN                     (N never assumed — see docs/DYNAMIC_PANEL_DETECTION.md)
      -> crop each accepted panel detection (padded, clipped to the image)
      -> [diagnostic] crop-quality checks — warnings only, never a rejection
      -> 4-class condition model on the crop
      -> per-panel condition result: status + label + confidence + full detection list
      -> assembled per-image JSON report
```

Module: `training/detection/ground_pv/panel_condition_pipeline.py`.
CLI: `scripts/analyze_panel_conditions.py`.
Tests: `tests/test_ground_pv_panel_condition_pipeline.py` (duck-typed fake
models, no GPU needed — same pattern as Phase 14A's tests).

This pipeline NEVER re-answers "is this a panel" — that is entirely the
dynamic panel detector's job. It only ever runs condition inference on
crops the panel detector already produced, and the number of condition
inferences always equals the accepted panel count for that image.

## 2. Four-class taxonomy

Defined once, canonically, in
`training/detection/ground_pv/condition_taxonomy.py`:

```
0 = CLEAN
1 = DUSTY
2 = BIRD_DROP
3 = PHYSICAL_DAMAGE
```

**ELECTRICAL_DAMAGE is not part of the active RGB condition taxonomy.**
It existed in the prior 5-class research dataset and checkpoint
(`weights/yolo_solar_condition_research_v1.pt`, kept frozen — see
[[ground_pv_condition_model_baseline]]), but the source audit found those
annotations visually correspond to **cell cracks**, not confirmed
electrical failure. Removing it is not a data-cleanup convenience; it is
because the label itself was never trustworthy. It is never silently
relabelled as PHYSICAL_DAMAGE — see §3.

Also explicitly and permanently excluded: SNOW (out of scope), generic
"DEFECTIVE" (too ambiguous — never present in the source data anyway),
and PANEL (that's a localization concept, not a condition — belongs to
Phase 14A, not here).

**HOTSPOT requires thermal/IR data and is not inferred from RGB imagery.**
A future separate thermal/IR modality, not this one.

`assert_active_taxonomy()` fails closed: any model or dataset whose class
list isn't EXACTLY `[CLEAN, DUSTY, BIRD_DROP, PHYSICAL_DAMAGE]` in that
order is rejected with a clear error, never silently accepted as if it
were the active taxonomy. `scripts/analyze_panel_conditions.py` calls this
before running — it refuses to run against the 5-class checkpoint.

## 3. Dataset: `condition_4class_v1`

Derived from the existing 5-class research dataset
(`datasets/ground_pv_research/condition/`, itself built from provenance-
grade-C Kaggle-sourced S1/S3, see
[[ground_pv_condition_dataset_build]]) — **read-only, never mutated**
(verified via a before/after tree digest; the build aborts if it ever
changed).

Because the 5-class taxonomy already numbers CLEAN=0, DUSTY=1,
BIRD_DROP=2, PHYSICAL_DAMAGE=3 identically to the active 4-class taxonomy,
converting is pure filtering: drop every ELECTRICAL_DAMAGE (class 4)
annotation LINE, keep every other line verbatim (never renumbered, never
relabelled), and drop any image whose annotations were entirely
ELECTRICAL_DAMAGE (same "empty after class removal" policy the 5-class
builder already applies to snow-only images). Because this is a strict
class-filtered SUBSET of an already leakage-clean, family-grouped split —
no image ever changes split, no family/scene grouping is recomputed — the
result is leakage-clean by construction, re-verified defensively anyway.

Module: `training/detection/ground_pv/build_condition_4class_dataset.py`.
CLI: `scripts/build_condition_4class_dataset.py`.

**Real build results** (see §4 for the full table):

| | 5-class source | 4-class derived |
|---|---|---|
| images | 2,509 | 2,411 |
| ELECTRICAL_DAMAGE instances | 1,119 | 0 (removed, not relabelled) |
| images dropped (all-electrical-damage) | — | 98 |
| images with partial removal (kept) | — | 381 |

All quality gates passed: no ELECTRICAL_DAMAGE/SNOW/DEFECTIVE/PANEL class
present, no family crosses a split, no exact duplicate crosses a split,
`electrical_damage_never_relabelled: True` by construction.

## 4. Dataset statistics

From the real conversion (`datasets/ground_pv_research/condition_4class_v1/reports/condition_4class_build_report.md`):

| class | train img | valid img | test img | total img | total instances |
|---|---|---|---|---|---|
| CLEAN | 639 | 184 | 95 | 918 | 6,000 |
| DUSTY | 959 | 261 | 144 | 1,364 | 8,550 |
| BIRD_DROP | 737 | 194 | 100 | 1,031 | 7,742 |
| PHYSICAL_DAMAGE | 205 | 79 | 30 | 314 | 725 |

Split: train 1,686 / valid 483 / test 242 (preserved exactly from the
5-class leakage-clean split). PHYSICAL_DAMAGE remains the rarest class by
a wide margin (314 images / 725 instances) — a real class-imbalance
limitation carried forward from the 5-class dataset, not introduced by
this conversion.

## 5. Condition inference architecture

The condition model is a **detector** (same architecture family as the
existing 5-class research checkpoint), run on a **panel crop** — not the
full image, and not a whole-image classifier. This was already the
established pattern (`scripts/infer_condition_on_panels.py`); Phase 15A
keeps it, because a detector naturally supports **multiple condition
detections on one crop** (e.g. a panel that is both dusty and has a
bird-drop mark) without forcing an artificial single-label diagnosis.

Per panel, the output carries both:
- `condition` — the single HIGHEST-confidence detection, for a stable,
  simple schema (`status`/`label`/`confidence`).
- `condition_detections` — the FULL list of raw detections on that crop,
  for when multiple simultaneous conditions matter.

## 6. Dynamic panel handling

`panel_count` and the per-panel results list are always derived from
however many panels `dynamic_panel_detection.detect_dynamic_panels`
(Phase 14A) returns for THIS image — 4, 6, 8, or any other N. Nothing in
`panel_condition_pipeline.py` special-cases a count. Verified by tests
covering 2x2/2x3/2x4 grids and arbitrary N in (1,3,5,7,9,12): the number
of condition-model calls always exactly equals the accepted panel count.

## 7. Panel crop generation

`compute_padded_crop_box` pads each panel's detected bbox by a
configurable fraction (`crop_pad`, default 0.04 — matches the existing
`infer_condition_on_panels.py --crop-pad` convention) and clips to the
image bounds. `check_crop_quality` then runs diagnostics (never used to
reject a crop, per req 9 — warnings only):

- `panel_crop_too_small` — either side below `min_crop_side_px` (20px default)
- `panel_crop_insufficient_area` — crop area vs full image area below `min_crop_relative_area`
- `panel_crop_extreme_aspect_ratio` — outside `[min_aspect_ratio, max_aspect_ratio]`
- `panel_crop_truncated` — the actual crop area (after edge-clipping) is
  less than `min_surviving_crop_fraction` of the INTENDED padded box —
  catches a panel detected right at the image edge
- `panel_crop_possibly_blurry` — a simple, dependency-free (no cv2/scipy)
  discrete-Laplacian-variance blur metric below `blur_variance_threshold`

## 8. Confidence semantics

`ConditionInferenceConfig`:

- `detection_floor` (default **0.25**) — reuses the EXISTING documented
  convention from `scripts/infer_condition_on_panels.py`'s
  `--condition-conf` default. This is the YOLO inference-time conf cutoff:
  a box below this never appears at all.
- `predicted_threshold` (default **0.45**) — reuses the project's OTHER
  existing documented "trustworthy confidence" convention (the ground
  panel detector's own real-world-holdout bar, docs/DYNAMIC_PANEL_DETECTION.md).
  Neither value is invented from nothing — both come from conventions
  already established elsewhere in this codebase, for consistency.

A detection with confidence in `[detection_floor, predicted_threshold)` is
reported `"uncertain"`; at or above `predicted_threshold`, `"predicted"`.

## 9. No-prediction semantics

Three explicit states, never conflated:

- **`"predicted"`** — top detection confidence >= `predicted_threshold`.
- **`"uncertain"`** — the model DID detect something, but below the
  trustworthy bar.
- **`"no_prediction"`** — the model returned ZERO detections on this crop.

**A `"no_prediction"` or `"uncertain"` result is NEVER interpreted as
`"CLEAN"`.** Absence of a detection is exactly that — absence — not a
diagnosis. Verified by tests
(`test_no_prediction_never_becomes_clean`, and the CLI's own disclaimer).

## 10. Limitations

- **PHYSICAL_DAMAGE is rare** (314 images / 725 instances vs. DUSTY's
  1,364 images / 8,550 instances) — a real class-imbalance limitation
  inherited from the 5-class dataset, unaddressed by this phase.
- **Provenance grade C** — the underlying images ultimately trace to a
  Kaggle dataset with an Unknown license; personal, non-commercial
  research use only (unchanged from the 5-class dataset's own caveat).
- **Blur/truncation thresholds are newly-defined, not empirically tuned**
  against this specific deployment's cameras — documented, configurable
  defaults, not validated production thresholds.
- **The dynamic panel detector's own known failure modes still apply**
  (a large foreground panel can be missed entirely; a wide array shot can
  produce group-like boxes instead of per-panel boxes — see
  docs/DYNAMIC_PANEL_DETECTION.md §7) — this pipeline can only run
  condition inference on whatever the panel detector actually finds.
- **Any trained checkpoint from this phase is a SMOKE-SCALE artifact at
  best** (CPU-only, a small fraction of the training set, a handful of
  epochs — see `scripts/train_condition_4class_smoke.py`'s own docstring
  and the Phase 15A final report for whether a run actually completed on
  this machine). It exists only to prove the architecture end-to-end if
  it succeeds; it is never a claim of production accuracy.

## 11. Research-only status

Nothing in this phase is wired into `models/`, `services/`, or the
production Streamlit app. `weights/yolo_solar.pt`,
`weights/manifest.json`, and `weights/yolo_solar_condition_research_v1.pt`
are all verified unchanged before and after this work. Any new checkpoint
is saved under a clearly distinct filename
(`weights/yolo_solar_condition_4class_research_v1.pt`) and is explicitly
research-only.

## 12. Future: temporal analysis

This phase's panel identity (`P1..PN`) is spatial, per-capture only —
[[phase14a_dynamic_panel_detection]] already documents that it is "not
yet guaranteed to represent persistent physical panel identities across
time." Condition results inherit that same limitation: a condition read
on "P3 today" carries no guarantee of correspondence to "P3 tomorrow" if
the camera moves or panels enter/leave frame. Tracking condition CHANGE
over time for the same physical panel — the natural next step toward a
real monitoring product — requires solving persistent panel identity
first, which remains out of scope for both Phase 14 and Phase 15A.
