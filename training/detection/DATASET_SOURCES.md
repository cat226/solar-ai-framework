# YOLO Detection — Dataset Provenance Report

Generated: 2026-09-03
Repository: https://github.com/cat226/solar-ai-framework
Branch: feat/mobilenet-training-pipeline

## Production requirement

`models/detector.py` / `models/model_manager.py` expect an Ultralytics YOLO **detection**
model (reads `pred.boxes.xyxy`, `pred.boxes.conf`, `pred.boxes.cls` — the standard detection
task, not segmentation). Config (`configs/settings.yaml`): `confidence_threshold: 0.45`,
`iou_threshold: 0.50`, `image_size: 640`. No fixed multi-class label list is enforced in code
(unlike MobileNet's 6-class contract) — the detector's job is panel localization, and a
single class ("solar panel") is sufficient for the current pipeline (YOLO → crop → MobileNet
classifies the condition).

## Status Definitions

Same as `training/classification/DATASET_SOURCES.md`: VERIFIED + USABLE, VERIFIED + LICENSE
REVIEW REQUIRED, REQUEST REQUIRED, RESTRICTED, REJECTED, UNKNOWN.

## Candidates investigated

| Dataset | Source | License | Access | Format | Status |
|---|---|---|---|---|---|
| BDAPPV — crowdsourced aerial PV arrays | Kasmi et al., *Scientific Data* 2023 / Zenodo 7358126 | CC BY 4.0 (Google/IGN base imagery under their own separate terms) | OPEN, populated, downloadable (8.16GB + 17.7MB) | Segmentation masks (polygon-derived) | VERIFIED + USABLE, needs mask→bbox conversion |
| Multi-resolution PV segmentation (PV01/PV03/PV08) | Hou, Ling, Yujun — IGSNRR CAS / Zenodo 5171712 | CC BY 4.0 | OPEN, populated, downloadable (~6.8GB, 3 resolution tiers) | Segmentation masks | VERIFIED + USABLE, needs mask→bbox conversion |

### 1. BDAPPV — "A crowdsourced dataset of aerial images with annotated solar photovoltaic arrays and installation metadata"

- **Publication**: Kasmi, G., Saint-Drenan, Y-M., Trebosc, D., Jolivet, R., Leloux, J., Sarr, B., Dubus, L. — *Scientific Data* (Nature), 2023. Peer-reviewed, data-availability-mandated venue.
- **DOI / Zenodo**: https://zenodo.org/record/7358126
- **License**: CC BY 4.0 for the dataset itself. Note: underlying base imagery is from Google (subject to Google's own imagery terms) and IGN France (Open License 2.0) — attribution to both the dataset authors and imagery providers required.
- **Access**: OPEN, no request needed. `bdappv.zip` (8.16 GB, 28,807 Google + 17,325 IGN images) and `data.zip` (17.7 MB, metadata).
- **Annotation format**: Segmentation masks (13,303 Google + 7,686 IGN masks) generated from crowdsourced polygon annotations — **not bounding boxes directly**. Converting to YOLO-format bounding boxes requires computing the axis-aligned bounding box of each mask/polygon instance. This is a standard, well-defined, disclosed transformation (not a subjective relabeling judgment) — the resulting boxes would be documented as "derived from BDAPPV segmentation masks," not represented as originally hand-drawn bounding boxes.
- **Mask granularity — CONFIRMED 2026-09-03** by downloading and inspecting `data.zip`
  (17.7MB, not the full 8GB archive): the underlying annotations are **vector polygons**,
  not raster masks. `data/replication/campaign-google/polygon-analysis.json` (13,303
  entries) is a JSON list of `{id, polygons: [{points: [{x,y}, ...], score, area}, ...]}` —
  one entry per image, one polygon per individual panel installation, pixel-coordinate
  points. This is per-instance by construction (no connected-component splitting needed):
  a YOLO bounding box is simply `(min(x), min(y), max(x), max(y))` over each polygon's
  points, normalized by the source image's width/height. The IGN campaign has the
  equivalent `campaign-ign/polygon-analysis.json` (7,686 entries). Still need to confirm,
  before the full download: each image's actual pixel dimensions (to normalize
  coordinates) and how `id` maps to the corresponding image filename inside the 8GB
  `bdappv.zip` (not yet downloaded).
- **Status**: VERIFIED + USABLE (license/access), conversion path confirmed straightforward.
  Full 8GB image archive not yet downloaded (only the 17.7MB polygon metadata was).

### 2. Multi-resolution PV panel segmentation dataset (PV01/PV03/PV08)

- **Publication**: Hou, J., Ling, Y., Yujun, L. — State Key Laboratory of Resources and Environmental Information System (IGSNRR, CAS) / Provincial Geomatics Center of Jiangsu.
- **DOI / Zenodo**: https://zenodo.org/record/5171712
- **License**: CC BY 4.0.
- **Access**: OPEN, no request needed. ~6.8 GB across three spatial-resolution tiers (0.1m/0.3m/0.8m per pixel) — 71,448 recorded downloads, an established/widely-used dataset in this research area.
- **Annotation format**: Segmentation masks, same mask→bbox conversion consideration as BDAPPV. PV03 samples are additionally categorized by background land-use type; PV01 rooftop samples by roof type — these sub-categories are not relevant to our single-class "solar panel" detection need but don't hurt.
- **Status**: VERIFIED + USABLE (license/access), CONVERSION REQUIRED.

## Assessment vs. the classification pipeline's blockers

Unlike Bird-Drop/Electrical-Damage/Physical-Damage for MobileNet, **YOLO detection data is
not currently blocked** — two independently-licensed (CC BY 4.0), peer-reviewed-or-widely-cited,
publicly downloadable, no-request-needed sources exist. The remaining work is engineering
(mask→bbox conversion, YOLO-format dataset preparation, training pipeline), not
provenance/access research.

## Recommended next steps (not yet performed)

1. Download a small sample from one dataset (not the full 8GB/6.8GB) to inspect actual mask
   file structure and confirm per-instance vs. coarse-mask granularity before committing to a
   full download.
2. Decide which dataset (or both, combined) to use based on that inspection.
3. Build `training/detection/prepare_dataset.py` following the same conventions as
   `training/classification/prepare_dataset.py` (local-source-only, SHA-256 duplicate
   detection, deterministic leakage-safe splitting, provenance manifest) — but converting
   masks to YOLO-format bounding-box label files instead of copying class-labeled images.
4. Build `training/detection/train_yolo.py` using Ultralytics' training API against the
   converted YOLO-format dataset.
5. Evaluate on an untouched test set (mAP, precision/recall at the configured
   conf=0.45/iou=0.50 thresholds).

**No download, conversion, or training has been performed yet** — this document records
provenance research only, per this project's policy of not fabricating progress.
