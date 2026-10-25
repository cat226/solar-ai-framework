# Ground-level PV research dataset — remediation report

- raw dir (immutable): `D:\Documents\Projects\solar-ai-framework\datasets\ground_pv_research\raw\panel_solar_bw945_v1`
- derived dir: `D:\Documents\Projects\solar-ai-framework\datasets\ground_pv_research\yolo`
- seed: 42

**Remediation verdict: APPROVED WITH CAVEATS**

## Counts

- raw images: 6493
- derived images: 6492
- images removed: 1 (exact byte-duplicates: 1)
- filename collisions renamed: 0

## Near-duplicate families

- grouping: exact SHA-256 U source basename U 16x16 dHash Hamming <= 8 (different-basename only)
- families over raw: 537   over kept: 537
- largest family: 273 images   singletons: 0
- 16x16 dHash near-duplicate pairs merged: 333

## Final split (leakage-clean)

| split | images | unique families | target |
|---|---|---|---|
| train | 4546 | 373 | ~70% |
| valid | 1297 | 109 | ~20% |
| test | 649 | 55 | ~10% |

- **no family crosses splits: True**
- residual cross-split near-duplicate pairs: 0

Official Roboflow split (reference only): {'test': 648, 'train': 4546, 'valid': 1299}

## Class counts (derived, per split)

| class | train | valid | test | total |
|---|---|---|---|---|
| panel | 4545 | 1214 | 560 | 6319 |
| cracked | 6277 | 2122 | 637 | 9036 |
| dusty | 207 | 75 | 69 | 351 |

## Conversions & exclusions

- polygon → bounding box conversions: 614
- unparseable label lines dropped: 0
- **bird_drop excluded**: 81 annotations across 9 images
  - reason: only 9 images / ~3 unique source photos in the audited source dataset - insufficient independent data to train or evaluate; not a product limitation
  - the affected images are kept (their `panel` boxes remain); only bird_drop boxes are dropped
- derived taxonomy: {0: 'panel', 1: 'cracked', 2: 'dusty'}

## Traceability

- `<derived>/TRACEABILITY.csv` and `splits/leakage_clean_split/manifest.csv`: every derived image → source_path, source_official_split, family_id, sha256, classes.
- `splits/official_split/manifest.csv`: every raw image → its official split (provenance; not overwritten, not used for training).

