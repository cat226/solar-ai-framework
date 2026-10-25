# Unified condition/defect research dataset — build report

**FINAL STATUS: APPROVED WITH CAVEATS**

> PERSONAL, NON-COMMERCIAL RESEARCH ONLY. Provenance grade **C**, tagged `personal-experimental-unverified-provenance`. Underlying images ultimately trace to Kaggle `pythonafroz/solar-panel-images` (licence: **Unknown**). Do NOT redistribute / publish / use commercially / claim verified CC BY image rights.

## Sources

| src | dataset | ver | task | annotation |
|---|---|---|---|---|
| S1 | defects-kgig6/detection-for-defects-in-solar-panels | 2 | detection | bounding boxes |
| S3 | solar-panel-a5cyn/dusty-and-bird-droppings | 1 | detection | bounding boxes |

## Counts

- total source images (S1 + S3): 2673
- images dropped: {'empty': 28, 'exact_duplicate_within_pool': 6, 'snow_only': 115, 'all_malformed': 15}
  - snow-only: 115 · empty: 28 · all-malformed: 15 · exact-dup-within-pool: 6
- malformed annotation lines dropped: 16
- snow boxes dropped: 6144
- **derived images (unique after dedup): 2509**
- near-duplicate families (over all source images): 1244

## Leakage-clean split

| split | images | unique families |
|---|---|---|
| train | 1757 | 835 |
| valid | 502 | 240 |
| test | 250 | 120 |

- **no family crosses splits: True**
- crossing families (should be empty): {}

## Class distribution (unified taxonomy)

| class | train img | valid img | test img | total img | total instances | unique families |
|---|---|---|---|---|---|---|
| CLEAN | 639 | 184 | 95 | 918 | 6000 | 508 |
| DUSTY | 959 | 261 | 144 | 1364 | 8550 | 850 |
| BIRD_DROP | 737 | 194 | 100 | 1031 | 7742 | 585 |
| PHYSICAL_DAMAGE | 205 | 79 | 30 | 314 | 725 | 212 |
| ELECTRICAL_DAMAGE | 340 | 95 | 44 | 479 | 1119 | 317 |

## ELECTRICAL_DAMAGE — semantic-risk warning

- images: 479   annotations: 1119
- **`electrical_damage_semantic_risk = HIGH`** on every one — the audit found S1's `electrical-damage` boxes visually correspond to **cell cracks**, not confirmed electrical failure. The original source class is preserved; nothing was relabelled. Per-annotation records: `manifests/electrical_damage_annotations.csv`.

## Overlap with the existing 6,492-image ground dataset

{'existing_images': 6492, 'candidate_images': 2509, 'exact_sha_matches': 0, 'perceptual_near_dup_images': 33, 'overlap_fraction': 0.0132}

## Quality gates

- 10_derived_images_with_zero_boxes: 0
- 13_raw_S1_S3_unchanged: True
- 14_existing_ground_dataset_unchanged: True
- 1_no_exact_dup_across_splits: True
- 2_no_family_crosses_splits: True
- 3_no_invalid_images: True
- 4_no_malformed_annotations_in_output: True
- 5_no_defective_class: True
- 6_no_snow_class: True
- 7_no_fabricated_boxes: True
- 8_every_derived_image_has_provenance: True
- 9_electrical_damage_semantic_risk_preserved: True
- **ALL PASS: True**
