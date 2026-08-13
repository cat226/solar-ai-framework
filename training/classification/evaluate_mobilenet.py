from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

CLASSES = ["Clean", "Dusty", "Bird-Drop", "Electrical-Damage", "Physical-Damage", "Hotspot"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    if not a.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {a.checkpoint}")
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    ds = datasets.ImageFolder(a.data_root / "test", transform=tf)
    if ds.classes != CLASSES:
        raise SystemExit(f"test classes must exactly equal {CLASSES}; got {ds.classes}")
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 6)
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
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "report": classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
