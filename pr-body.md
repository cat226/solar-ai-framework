## Summary

Fixes MobileNetV2 training pipeline class-ordering issues and adds dataset preparation utility.

## Changes

### 1. Fixed evaluate_mobilenet.py class ordering

**Root cause:** evaluate_mobilenet.py used 	orchvision.datasets.ImageFolder directly, which sorts class directories alphabetically. The production model expects a fixed class order:

1. Clean
2. Dusty
3. Bird-Drop
4. Electrical-Damage
5. Physical-Damage
6. Hotspot

ImageFolder alphabetical order differs from this production order, causing misaligned targets, incorrect confusion matrices, and wrong classification reports.

**Fix:** Added _map_dataset_to_production() which:
- Validates all six required classes exist
- Rejects unknown classes (including Snow-Covered)
- Explicitly maps ImageFolder alphabetical indices to production indices
- Ensures targets, confusion matrix, and classification_report all use production order

### 2. Verified 	rain_mobilenet.py class ordering

	rain_mobilenet.py already contains the correct explicit mapping in _dataset():
`python
mapping = {ds.class_to_idx[name]: CLASSES.index(name) for name in CLASSES}
ds.targets = [mapping[target] for target in ds.targets]
`

No changes needed.

### 3. Added prepare_dataset.py

New dataset preparation utility that:
- Operates only on local source directories (no automatic downloads)
- Validates exact six-class production schema
- Rejects unknown/missing classes
- Detects duplicates via SHA-256
- Prevents leakage between train/val/test splits
- Supports deterministic splitting with configurable seed
- Supports grouping by source identifier to prevent related images crossing splits
- Writes manifest with provenance metadata

CLI:
`ash
python training/classification/prepare_dataset.py \
  --source /path/to/raw \
  --output /path/to/output \
  --seed 42
`

### 4. Added tests

	ests/test_mobilenet_training.py — 13 tests covering:
- ImageFolder alphabetical ordering cannot silently alter labels
- Explicit production class mapping
- Missing class rejection
- Unknown class rejection
- Snow-Covered rejection (not mapped to Hotspot)
- Deterministic splitting
- Duplicate detection
- No duplicates across splits
- Expected directory structure
- Manifest provenance

## Validation

- Full pytest suite: **753 passed, 4 skipped**
- pip check: No broken requirements
- All CLI help commands work

## Notes

- No model weights fabricated
- No synthetic data created
- No training executed (requires real dataset)
- Production class order preserved
