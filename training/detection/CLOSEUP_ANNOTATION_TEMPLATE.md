# Close-up solar-panel detection — annotation template

This is the exact input format `training/detection/prepare_closeup_dataset.py`
expects. It exists so that the moment a legitimate, appropriately-licensed
close-up/ground-level solar-panel image source is acquired, annotation can
begin immediately against a known, already-tested target format, rather
than inventing one ad hoc.

**No annotation has been produced from this template yet.** Manual
bounding-box annotation is a human judgment task (deciding what counts as
a "visible enough" partial panel, drawing a tight box) that this project
does not perform automatically or synthetically — doing so would violate
this project's own no-fabrication policy (fabricated/AI-generated boxes
are explicitly disallowed as training ground truth). The remaining step is
manual: a person with rights to use candidate images opens each one in a
real annotation tool (e.g. LabelImg, CVAT, makesense.ai) and exports YOLO
format, then fills in `provenance.json` by hand for every image.

## Directory layout

```
SOURCE_DIR/
    images/
        panel_0001.jpg
        panel_0002.jpg
        ...
    labels/
        panel_0001.txt      # YOLO format - see below
        panel_0002.txt
        ...
    provenance.json         # one entry per image id (filename stem) - see below
```

## Label format (`labels/<id>.txt`)

Standard YOLO txt: one line per visible panel,
`<class_id> <x_center> <y_center> <width> <height>`, all four coordinates
normalized to `[0, 1]` relative to image width/height.

- **`class_id` is always `0`** ("solar_panel" - the only detection class).
  Do not introduce Clean/Dusty/Hotspot at this stage; those are MobileNet
  classification labels applied downstream to a *cropped* panel, never
  YOLO detection classes.
- A genuinely panel-free image gets an **empty** `labels/<id>.txt` file
  (present, zero bytes) - not a missing file. This is how a real negative
  sample is represented; a missing label file is treated as an error, not
  an implicit negative.
- Include partial/occluded panels if a person could still reasonably
  identify it as a panel from the visible portion; box only the visible
  extent, do not guess at an occluded boundary.

Example (`panel_0001.txt`, one full panel roughly centered, one partial
panel at the right edge):

```
0 0.482 0.510 0.360 0.420
0 0.930 0.610 0.140 0.310
```

## Provenance format (`provenance.json`)

```json
{
  "panel_0001": {
    "source_url": "https://example.org/where-this-photo-came-from",
    "license": "CC-BY-4.0",
    "rights_holder": "Jane Doe / Example Institution",
    "group_key": "session-2026-09-05-rooftop-A",
    "notes": "Optional free text - e.g. camera/angle/lighting context."
  }
}
```

- **`license`** must be one of the values `prepare_closeup_dataset.py`
  accepts (see its `_ALLOWED_LICENSES`): `CC0`, `CC-BY-*`, `CC-BY-SA-*`,
  `MIT`, or `explicit-written-grant` (a direct written grant from the
  rights holder, outside a standard license template - describe the grant
  in `notes` if used). Anything else, including "free to view/download"
  with no explicit license, is rejected by the pipeline, not silently
  accepted - this project was burned once already by trusting a re-
  uploader's self-declared license over unclear original provenance (see
  `training/classification/DATASET_SOURCES.md`'s Bird-Drop correction) and
  will not repeat that mistake here.
- At least one of `source_url` or `rights_holder` is required.
- **`group_key`** (optional but recommended) should identify images from
  the same capture session/scene (e.g. several photos of the same rooftop
  taken moments apart). `prepare_closeup_dataset.py` guarantees every
  image sharing a `group_key` lands in the same train/val/test split, so
  near-duplicate/same-scene content never leaks across splits.

## What happens next, once real annotated data exists

```bash
python training/detection/prepare_closeup_dataset.py \
  --source-dir path/to/SOURCE_DIR \
  --output path/to/output_root \
  --seed 20260905
```

This validates every image against the rules above (rejecting anything
missing provenance, an unapproved license, a malformed label, or an
exact/near-duplicate already accepted), then writes a standard
`train/val/test/{images,labels}` YOLO dataset plus a `manifest.json`
recording per-image provenance, the dataset's content hash, and the exact
split-assignment method - ready to hand to `training/detection/train_yolo.py`
(Experiment B/C in `docs/ML_DOMAIN_REMEDIATION.md`) once enough images
exist to make a training run meaningful.
