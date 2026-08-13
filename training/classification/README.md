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

Split original images before augmentation. Augmented copies must remain confined to the training split. When source metadata permits, group by panel/module/source before splitting to reduce leakage. Keep the test set untouched until final evaluation.

## Commands

From the repository root:

```bash
python training/classification/prepare_dataset.py --source-root /path/to/raw --output-root training/data/classification --manifest training/data/classification/manifest.json
python training/classification/train_mobilenet.py --data-root training/data/classification --output weights/mobilenet_solar.pth
python training/classification/evaluate_mobilenet.py --data-root training/data/classification --checkpoint weights/mobilenet_solar.pth
```

The scripts must fail closed on missing classes, malformed images, duplicate inputs, invalid splits, or incompatible checkpoints. They must never download or fabricate training data.

## Evaluation gate

Report per-class precision, recall, F1, support, overall accuracy, macro-F1, weighted-F1, and a confusion matrix. Training accuracy alone is not a production gate. The final checkpoint must load through the repository MobileNet model manager and pass output-shape/probability validation.
