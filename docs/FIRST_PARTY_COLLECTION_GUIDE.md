# First-party panel data collection guide (Phase 14B.1)

Short and practical. For the tooling and current dataset status, see
`training/detection/ground_pv/representative_dataset.py` and
`datasets/ground_pv_research/representative_panel_v1/reports/dataset_report.md`.

## A. Capture requirements

- Use the intended deployment camera/mount where possible — a phone photo
  from an arbitrary angle is a worse proxy than the real setup.
- Preserve native resolution. Preserve native aspect ratio. Do not crop,
  rotate, or resize before import — `scripts/import_first_party_panels.py`
  never converts a source image, and cropping/resizing throws away exactly
  the framing information the dataset needs.
- Photograph the COMPLETE visible array where practical, not a partial
  crop of it.

## B. Deliberately vary what you capture

The two confirmed real-world failure modes ([[phase14a_dynamic_panel_detection]])
are: a large foreground panel getting missed entirely, and several
physical panels getting boxed as one group. Collect FOR these cases, not
just easy ones:

- panel counts: 4, 6, 8 are the priority targets, but don't artificially
  restrict to only those — any N is useful.
- distance: close-up (one panel fills most of the frame) through
  full-array-at-a-distance.
- angle: straight-on, oblique, low camera, high camera, left/right offset.
- a single panel large enough to fill most of the frame, with its border
  actually visible (not a whole-frame degenerate crop).
- panels touching or with minimal gap between them (no isolating
  background strip) — the case that produces "group boxes".
- lighting: sunny, cloudy, morning, midday, late afternoon, strong
  shadows, backlighting where practical.
- realistic occlusion: vegetation, posts, cables, dust, a person passing
  — do not stage anything unsafe or unrealistic.

## C. Scene identity

- One physical installation/capture session = one scene. If you take 5
  photos of the same array from different angles in one visit, they
  share one `scene_id`.
- Tell the importer what you know — don't rely on automatic grouping to
  guess it (it only catches near-identical BURST frames, never "same
  array, different angle"):
  ```
  python scripts/import_first_party_panels.py --source ... --batch-id batch_003 --scene-id scene_012
  ```
  or, for a batch spanning multiple scenes, `--scene-map scene_map.csv`
  (columns `filename,scene_id`).
- If you genuinely don't know / a batch is a grab-bag of unrelated shots,
  leave both flags off. Every image gets `scene_status = "needs_review"` —
  correct and honest, not silently resolved. Resolve it later by re-running
  the import with an explicit mapping, or by editing the manifest directly.

## D. Annotation

- **ONE PHYSICAL PANEL = ONE BOUNDING BOX.** A box around a group of
  panels is a mis-annotation, not a shortcut.
- Class `panel`, `class_id` 0. No condition classes — this dataset is
  panel localization only.
- A partially-visible panel IS annotated when you can confidently tell
  it's a panel (enough border/cells visible), even if cropped by the
  frame edge or another object.
- An ambiguous fragment you CANNOT confidently identify as a panel is NOT
  annotated. Note why in the manifest if practical.
- Ground truth is human-verified only. `scripts/generate_annotation_proposals.py`
  can draw the CURRENT detector's raw predictions as a visual aid — every
  such image is watermarked **"PROPOSAL — NOT GROUND TRUTH"** and saved
  under a `.proposal.json`/`.proposal.jpg` extension, never as a `.txt`
  label file. A proposal becomes ground truth only when you personally
  draw/confirm each box in the real label file.
- Once you've verified an image's boxes, set that manifest entry's
  `annotation_status = "annotated_human_verified"`, then run
  `sync_panel_count_gt_from_annotations` (or the audit CLI, which reports
  disagreements) so `panel_count_gt` is derived from the actual boxes —
  never type that number in separately.

## E. Safety

- Do not enter restricted, unsafe, or elevated areas merely to get a
  photo. A missing difficult-case example is not worth an injury.
- Collect only with permission for the property/installation being
  photographed.

## Quick reference

```
# 1. Import a batch (originals preserved, deduplicated, metadata extracted)
python scripts/import_first_party_panels.py --source "D:\new_photos" --batch-id batch_004 --scene-id scene_020

# 2. (optional) generate DRAFT visual proposals from the current detector
python scripts/generate_annotation_proposals.py \
    --images-dir datasets/ground_pv_research/representative_panel_v1/raw/batch_004 \
    --weights weights/yolo_solar_ground_research_v1.pt \
    --output-dir datasets/ground_pv_research/representative_panel_v1/proposals/batch_004

# 3. Annotate for real (one panel = one box) in annotations/, verify, then
#    mark annotation_status="annotated_human_verified" in the manifest.

# 4. Check progress against the readiness gates
python scripts/audit_representative_panel_dataset.py
```
