# MobileNet solar-condition training

This directory contains reproducible preparation, training, and evaluation entry points for the production MobileNetV2 classifier.

## Production contract

The application expects six output classes in this exact order:

1. `Clean`
2. `Dusty`
3. `Bird-Drop`
4. `Electrical-Damage`
5. `Physical-Damage`
6. `Hotspot`

The trained artifact is `weights/mobilenet_solar.pth` and must be a PyTorch state dict compatible with the architecture constructed by `models/model_manager.py`.

## Data policy

Training data is intentionally not stored in Git. Each raw source must be downloaded separately and recorded in a local manifest with source URL, version/date, license, and attribution requirements.

Do not relabel incompatible classes. In particular, `Snow-Covered` is not `Hotspot` and must not be used as a substitute.

Prepared layout:

```text
training/data/classification/
  Clean/
  Dusty/
  Bird-Drop/
  Electrical-Damage/
  Physical-Damage/
  Hotspot/
```

## Split policy

Split original images before augmentation. Augmented copies must remain confined to the training split.

**Class labels are NOT grouping identifiers.** The dataset preparation utility only treats records as belonging to the same group when real panel/module/source identity metadata exists (e.g., from filename patterns or directory structure). When no grouping metadata is available, images are split deterministically *within each class* to preserve class stratification, targeting approximately 80/10/10 train/val/test proportions per class.

This means:
- Each class is independently represented in train, val, and test whenever there are enough samples.
- No single class is confined to a single split unless it has fewer than ~10 images.
- The same SHA-256 never appears in more than one split.

Keep the test set untouched until final evaluation.

## Commands

From the repository root:

```bash
python training/classification/prepare_dataset.py --source /path/to/raw --output training/data/classification
python training/classification/train_mobilenet.py --data-root training/data/classification --output weights/mobilenet_solar.pth
python training/classification/evaluate_mobilenet.py --data-root training/data/classification --checkpoint weights/mobilenet_solar.pth
```

The scripts must fail closed on missing classes, malformed images, duplicate inputs, invalid splits, or incompatible checkpoints. They must never download or fabricate training data.

## Interim (non-production) runs on a class subset

When one or more production classes are legitimately blocked on data access/licensing (see
`DATASET_SOURCES.md` for current status), all three scripts accept an opt-in `--classes` flag
naming a subset of the six production classes:

```bash
python training/classification/prepare_dataset.py --source /path/to/raw --output training/data/classification_interim --classes Clean Dusty Hotspot
python training/classification/train_mobilenet.py --data-root training/data/classification_interim --output weights/mobilenet_interim.pth --classes Clean Dusty Hotspot
python training/classification/evaluate_mobilenet.py --data-root training/data/classification_interim --checkpoint weights/mobilenet_interim.pth --classes Clean Dusty Hotspot
```

Omitting `--classes` preserves the exact original behavior: all six production classes required,
fail-closed if any is missing. A checkpoint trained on a subset is **not** compatible with the
production `ModelManager` and must never be saved to `weights/mobilenet_solar.pth` or otherwise
presented as the production model. Manifests and evaluation output from a subset run record
`is_production_class_set: false` so this is never ambiguous.

## Evaluation gate

Report per-class precision, recall, F1, support, overall accuracy, macro-F1, weighted-F1, and a confusion matrix. Training accuracy alone is not a production gate. The final checkpoint must load through the repository MobileNet model manager and pass output-shape/probability validation.
