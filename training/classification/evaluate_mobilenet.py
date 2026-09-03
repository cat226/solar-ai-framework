from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

try:
    from training.classification._dataset_remap import RemappedImageFolder, remap_to_class_order
except ImportError:
    from _dataset_remap import RemappedImageFolder, remap_to_class_order

PRODUCTION_CLASSES = ["Clean", "Dusty", "Bird-Drop", "Electrical-Damage", "Physical-Damage", "Hotspot"]
CLASSES = PRODUCTION_CLASSES


def _map_dataset_to_production(ds: datasets.ImageFolder, classes: list[str] | None = None) -> RemappedImageFolder:
    """Remap ImageFolder targets from alphabetical indices to the given class order.

    ImageFolder sorts directory names alphabetically, but the model expects a
    fixed class order. This function validates the required classes and remaps
    every target to that order's index.
    """
    if classes is None:
        classes = PRODUCTION_CLASSES
    return remap_to_class_order(ds, classes)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        choices=PRODUCTION_CLASSES,
        help=(
            "Subset of the six production classes the checkpoint was trained on, for "
            "evaluating an explicitly non-production interim model. Defaults to all six "
            "production classes."
        ),
    )
    a = p.parse_args()
    classes = a.classes if a.classes else PRODUCTION_CLASSES
    if len(set(classes)) != len(classes):
        raise SystemExit(f"--classes contains duplicate entries: {classes}")
    if not a.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {a.checkpoint}")
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    ds = _map_dataset_to_production(datasets.ImageFolder(a.data_root / "test", transform=tf), classes)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            probs = torch.softmax(out, dim=1)
            if not torch.isfinite(probs).all() or not torch.allclose(probs.sum(1), torch.ones(len(probs)), atol=1e-4):
                raise RuntimeError("invalid probability output")
            y_true.extend(y.tolist())
            y_pred.extend(out.argmax(1).tolist())
    # classification_report's dict output already contains accuracy plus macro/weighted
    # precision, recall, and f1 - derive from it once instead of five redundant sklearn passes.
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    result = {
        "classes": classes,
        "is_production_class_set": classes == PRODUCTION_CLASSES,
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
