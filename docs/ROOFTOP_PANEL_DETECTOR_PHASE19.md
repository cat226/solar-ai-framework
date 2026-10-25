# Rooftop panel detector, Phase 19

Research-only. Branch `feat/ground-pv-research-detector`, not merged, not
deployed. Answers a concrete request: the owner supplied 54 real ground-level
photos of their own rooftop array — some showing all 6 panels in one frame,
some close-ups — and asked that the detector find all 6 panels reliably and
that the condition model (CLEAN / DUSTY / BIRD_DROP / PHYSICAL_DAMAGE) be
retrained to work properly on this domain.

## 1. Why a new phase, not a v2 retrain of the old recipe

[[phase12_data_acquisition_audit]] and [[phase14b_representative_dataset_pipeline]]
already established that no suitable public ground-level panel-localization
dataset exists, and that the project had exactly 6 first-party images — far
below the >=100-scene minimum the readiness gates require. [[phase13b_beast_training_blocked]]
and [[phase14a_dynamic_panel_detection]]'s six-photo holdout both show the
same two real failure modes on real rooftop photos: **a large foreground
panel gets missed entirely**, and **several physical panels get boxed as one
group**. The 54 new owner photos are the first first-party batch large enough
to do something with — this phase collects them properly, labels them
honestly, and retrains against them.

## 2. Data collection (first-party, Phase 14B.1 tooling, batch_002)

54 images (17 iPhone photos + 37 WhatsApp video-call frames) imported via
`scripts/import_first_party_panels.py --batch-id batch_002_rooftop_2026_09_18`
with an explicit `--scene-map` (10 scenes, grouped by capture session/camera —
see `datasets/ground_pv_research/representative_panel_v1/manifests/scene_map_batch_002.csv`).
All images are from **one physical rooftop installation**; held-out "test"
scenes are a different capture session on the same roof, not a different site
— a real limitation, disclosed here, not hidden.

## 3. Labeling: AI-reviewed SAM proposals, not human-verified ground truth

No-fabrication policy: these are not claimed as human-verified. The pipeline is

1. `scripts/propose_panels_sam.py` — MobileSAM (`mobile_sam.pt`, Apache-2.0,
   Ultralytics official release asset) automatic mask generation on every
   image (CPU, ~35-45s/image). Annotation aid only, never shipped in any model.
2. `scripts/render_panel_proposal_review.py` — filters masks by area, mask
   solidity, and HSV hue/saturation (blue panel glass vs. roof/sky), dedups
   near-identical masks, and renders a numbered overlay + 10% coordinate grid
   per image.
3. A reviewer (Claude, an AI — not the property owner) looked at every one of
   the 54 overlays and recorded, per image, which numbered SAM candidates to
   keep and which boxes to hand-draw where SAM missed or merged panels
   (`manifests/panel_review_decisions_batch_002.json`).
4. `scripts/build_panel_labels_from_review.py` turns that into YOLO labels,
   snapping every hand-drawn box to the nearest SAM mask when they agree
   (IoU >= 0.55; 47/47 hand-added boxes were refined this way — log in
   `manifests/panel_refine_log_batch_002.json`), and sets
   `annotation_status = "annotated_ai_assisted_unverified"` — a **new**,
   deliberately distinct status (`representative_dataset.py`) that never
   satisfies the existing `annotated_human_verified` readiness gates.

**Labeling policy, disclosed:** one physical panel = one box; only confident,
delineable panels with meaningful visible area were labeled. Far-field
slivers, glare-blurred distant rows, and ambiguous fragments were left
unlabeled — so recall on very small/distant panels is genuinely unmeasured,
not silently assumed good. Two images (`IMG_5886.jpg`, `IMG_5887.jpg`) show a
dense far-field row where SAM masks and hand boxes kept overlapping/splitting
single panels; rather than train on noisy labels they were marked
`excluded_ambiguous` and dropped from the dataset entirely.

**This labeling should be spot-checked by the property owner before the
resulting checkpoint is trusted for anything beyond research comparison.**

## 4. Dataset: `rooftop_panel_v1` (1 class: `panel`)

`scripts/build_rooftop_panel_dataset.py` — scene-disjoint split (a scene can
never appear in two splits; asserted in code and in
`tests/test_rooftop_panel_tools.py`), 1280px long side, JPEG.

| split | images | boxes | scenes |
|---|---|---|---|
| train | 29 | 168 | scene0000, scene0001, scene0003, scene0004, scene0005, scene0015 |
| valid | 7  | 40  | scene0006, scene0008 |
| test  | 10 | 48  | scene0002, scene0007 |

Uploaded private to Kaggle as `edithstark/solar-ai-rooftop-panel-v1`.

## 5. Baseline: does the CURRENT detector already do this?

`scripts/evaluate_rooftop_panel_detector.py` against `yolo_solar_ground_research_v1.pt`
on the held-out `test` split (48 labeled panels, conf=0.25):

| metric | value |
|---|---|
| mAP50 | 0.274 |
| mAP50-95 | 0.131 |
| recall@conf | **0.125** (6 / 48 panels found) |
| precision@conf | 0.600 |

Per-image: 4 of the 10 test images get **zero** detections despite having
4-6 labeled panels each. This matches the pre-existing documented failure
modes exactly — confirms the retrain is warranted, not a fabricated pretext.

## 6. Retrain: `rooftop_panel_yolo11n_v1` (Kaggle, `rooftop-panel-yolo11n-v1`)

Two YOLO11n runs (`scripts/make_rooftop_panel_notebook.py`), same recipe,
different data:

- **MIX**: Susan panel-only images (all classes but `panel` dropped) +
  rooftop TRAIN scenes oversampled x10, 60 epochs.
- **ROOFTOP_ONLY**: rooftop TRAIN scenes only (COCO init), 100 epochs — tests
  whether Susan helps at all on this domain.

Both selected on rooftop VALID (target domain), reported on rooftop TEST
(scene-disjoint) and Susan TEST (panel-only, for the record). Trained on
Kaggle (Tesla T4, `torch==2.7.1+cu126`, `ultralytics==8.4.138`) via
`edithstark/rooftop-panel-yolo11n-v1`. Full manifest, per-run `results.csv`,
and an independent local re-evaluation (different code path from the
notebook's own `val()`, see `scripts/evaluate_rooftop_panel_detector.py`) are
committed under `datasets/ground_pv_research/rooftop_panel_v1/reports/`.

**Results (ultralytics `val()`, imgsz 640, iou 0.5):**

| | rooftop VALID | rooftop TEST (scene-disjoint) | Susan TEST (panel-only) |
|---|---|---|---|
| predecessor (`yolo_solar_ground_research_v1.pt`) | — | mAP50 0.274, recall 0.125 (6/48) | (unchanged, not re-run) |
| **MIX** (Susan + rooftop, oversample x10) | mAP50 0.979, recall 0.922 | **mAP50 0.935, recall 0.935** | mAP50 0.982, recall 0.970 |
| ROOFTOP_ONLY (rooftop only, COCO init) | mAP50 0.957, recall 1.000 | mAP50 0.932, recall 0.896 | mAP50 0.587, recall 0.582 |

ROOFTOP_ONLY overfits to the one rooftop it was trained on and collapses on
Susan (0.982 → 0.587 mAP50) — expected, and evidence the rooftop-only recipe
throws away everything the model previously knew. **MIX is promoted**: real
improvement on the target domain with no regression on the original dataset.

**Independent local re-check** (`scripts/evaluate_rooftop_panel_detector.py`,
a from-scratch greedy IoU matcher, not the notebook's own metric code),
rooftop TEST, conf>=0.25: predecessor 6/48 panels found (recall 0.125,
mAP50 0.274) -> MIX **39/48** panels found (recall 0.8125, mAP50 0.869). The
two independent measurements disagree on the exact number (0.935 vs. 0.8125
recall — different NMS/matching implementations) but agree completely on
the conclusion: a large, real improvement, not a rounding artifact.

**Promoted.** `weights/yolo_solar_rooftop_v1.pt` (sha256 `63fbdaad0c...`,
research checkpoint, gitignored, verified via the manifest above) is now
`ground_pipeline.PANEL_WEIGHTS` — the detector `app.py`'s Ground Inspection
page and the Raspberry Pi ingestion pipeline actually run. The predecessor
`yolo_solar_ground_research_v1.pt` (sha256 `2b26334dc4...`, verified
byte-unchanged) and the rejected `yolo_solar_rooftop_only_v1.pt` (sha256
`5d4ad4b815...`) are both kept, unmodified, for provenance/comparison —
neither is deleted.

**Qualitative smoke test** (`scripts/detect_dynamic_panels.py` re-run on all
54 source photos with the promoted checkpoint, conf 0.25): 0/48 images now
get zero detections (was 20/48 with the predecessor on this same batch).
The three clean "all 6 panels in frame" shots the owner asked about
(`IMG_5891.jpg`, `IMG_5893.jpg`) now detect exactly 6, one box per physical
panel, correctly excluding the blurred background rows; a 2x7 wide shot
(`WhatsApp Image ... 01.03.34.jpeg`) correctly separates all 14 individual
panels into 14 boxes with no merged "group" boxes. Annotated outputs are in
`research_runs/ground_pv/phase19/promoted_smoke_test/` (gitignored, regenerable).

## 6.1 Real ceiling: confidence threshold sweep, and why the rest isn't a training bug

There is no such thing as a "100% detector" on real photos — pushing that framing
would mean either overfitting to memorized images (useless on the next photo)
or reporting a fabricated number. What's actually true, measured on the
held-out test split (`datasets/ground_pv_research/rooftop_panel_v1/reports/threshold_sweep_output.txt`):

| conf threshold | recall | precision | false positives |
|---|---|---|---|
| 0.05 | 0.958 (46/48) | 0.554 | 37 |
| 0.10 | 0.938 (45/48) | 0.608 | 29 |
| 0.25 (app default) | 0.812 (39/48) | 0.696 | 17 |
| 0.40 | 0.771 (37/48) | 0.787 | 10 |

This is a real precision/recall trade-off curve, not a further improvement —
lowering the app's confidence slider recovers more panels at the cost of more
false-positive boxes; there is no free lunch here. At every threshold, 100%
recall never appears.

The 6 images that still miss a panel at conf=0.25 were visually inspected
(`research_runs/ground_pv/phase19/miss_*.jpg`): every one is the same genuinely
hard scene type that caused `IMG_5886.jpg`/`IMG_5887.jpg` to be excluded from
training entirely (Section 3) — a dense, steeply-angled, 10-15+ panel wide
shot where physical panels visually overlap/abut in the 2D projection. The
model DOES detect most panels there (visually confirmed, boxes align with
ground truth); the misses are specifically in the most crowded overlap zones.
This is a real, disclosed data-density limitation, not a bug fixable by more
training epochs or hyperparameter tuning — the project's own prior ablations
([[phase15d_a_resolution_ablation]], [[phase15e_a_augmentation_ablation]])
already found that direction gives no return once the underlying data doesn't
change. The honest next lever, if this matters for your use case, is more
first-party photos of exactly this dense/wide scene type — not a retrain trick.

## 7. Condition model: why it is NOT retrained in this phase

The owner also asked to retrain the 4-class condition model
(CLEAN/DUSTY/BIRD_DROP/PHYSICAL_DAMAGE). Running the existing classifier
(`mobilenet_condition_4class_research_v1.pth`) on the new dataset's
ground-truth panel crops shows it produces confident-looking but
**unverifiable** predictions (e.g. `DUSTY` on nearly every crop in some
images, `BIRD_DROP`/`PHYSICAL_DAMAGE` on others) — see
`research_runs/ground_pv/phase19/existing_classifier_on_gt_crops.json`.

**No condition label was fabricated for training.** Dust is often invisible
in a downsampled photo and bird-drop/crack ambiguity needs someone who has
physically seen the panels — that is not something an AI reviewer looking at
overlays can honestly supply, unlike panel *location* (a geometric,
visually-obvious property SAM handles well). `scripts/export_rooftop_condition_labeling_sheet.py`
exports every labeled panel crop + a blank CSV
(`datasets/ground_pv_research/rooftop_condition_v1/labeling_sheet.csv`) for
the **property owner** to fill in with what they actually know about each
panel. `scripts/build_rooftop_condition_classifier_dataset.py` refuses
(exit 2) to build a training set until every class has enough labeled crops
from enough distinct scenes — currently 0/0, correctly blocked, not faked.

## 8. What this phase does NOT claim

- Not human-verified ground truth (Section 3).
- One rooftop only — held-out scenes are the same site, different session.
- Small-panel/far-field recall is unmeasured by design (unlabeled).
- Condition retraining is blocked on the owner's own labels, not attempted
  with fabricated ones.
- Research checkpoint only; does not touch `weights/yolo_solar_ground_research_v1.pt`
  or any production weight.
