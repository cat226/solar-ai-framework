#!/usr/bin/env python3
"""scripts/train_condition_4class_classifier.py — Phase 16-B: train a
panel-crop CONDITION CLASSIFIER (MobileNetV2, same architecture family as
the project's existing production classifier, models/classifier.py) on the
derived classification dataset from
scripts/build_condition_4class_classifier_dataset.py.

Baseline for comparison: weights/yolo_solar_condition_4class_research_v2.pt
(the box-detector), test mAP50 0.444 / recall 0.441 (box-level),
image-level agreement 0.727 (docs/CONDITION_MODEL_IMAGE_LEVEL_AGREEMENT.md).
This script trains the classifier reformulation the 16-A.1 analysis
justified building. Evaluated on the SAME held-out test split (by content —
same images, now single-labeled) for an apples-to-apples image-level
comparison.

No production artifact touched. Writes a NEW checkpoint
(weights/mobilenet_condition_4class_research_v1.pth) — the existing
production v1 3-class classifier (weights/mobilenet_solar_v1.pth) and the
existing box-detector checkpoints are never modified.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_INPUT_SIZE = 224  # matches configs/settings.yaml models.mobilenet.input_size


def _build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(_INPUT_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(_INPUT_SIZE),
        transforms.CenterCrop(_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _run_epoch(model, loader, device, criterion, optimizer=None) -> tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train_mode:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            total_correct += int((logits.argmax(1) == labels).sum().item())
            total_n += images.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def _evaluate_confusion(model, loader, device, num_classes: int) -> list[list[int]]:
    model.eval()
    cm = [[0] * num_classes for _ in range(num_classes)]
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(1).cpu()
        for t, p in zip(labels.tolist(), preds.tolist()):
            cm[t][p] += 1
    return cm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/ground_pv_research/condition_4class_classifier_v1"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=4, help="early stop if val accuracy doesn't improve for N epochs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint-out", type=Path, default=Path("weights/mobilenet_condition_4class_research_v1.pth"))
    ap.add_argument("--report-out", type=Path, default=Path("docs/CONDITION_MODEL_4CLASS_CLASSIFIER_V1_TRAINING.json"))
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = datasets.ImageFolder(args.dataset / "train", transform=_build_transforms(train=True))
    valid_ds = datasets.ImageFolder(args.dataset / "valid", transform=_build_transforms(train=False))
    test_ds = datasets.ImageFolder(args.dataset / "test", transform=_build_transforms(train=False))

    # ImageFolder assigns class indices alphabetically — verify all three
    # splits agree with each other before trusting a shared label list.
    assert train_ds.classes == valid_ds.classes == test_ds.classes, (
        f"class order mismatch across splits: {train_ds.classes} vs {valid_ds.classes} vs {test_ds.classes}"
    )
    classes = train_ds.classes
    num_classes = len(classes)
    print(f"device={device}  classes={classes}  train={len(train_ds)}  valid={len(valid_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.to(device)

    # Class-weighted loss — train split is imbalanced (PHYSICAL_DAMAGE is
    # ~10x rarer than DUSTY); weighting by inverse frequency so training
    # doesn't just learn to always predict the majority class.
    counts = [0] * num_classes
    for _, y in train_ds.samples:
        counts[y] += 1
    weights = torch.tensor([len(train_ds) / (num_classes * max(1, c)) for c in counts], dtype=torch.float32, device=device)
    print(f"train class counts: {dict(zip(classes, counts))}  loss weights: {weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: list[dict] = []
    best_val_acc = -1.0
    best_epoch = -1
    epochs_since_improve = 0
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        train_loss, train_acc = _run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, valid_loader, device, criterion, optimizer=None)
        ep_dt = time.time() - ep_t0
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc, "seconds": ep_dt})
        print(f"epoch {epoch:2d}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  ({ep_dt:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch, epochs_since_improve = val_acc, epoch, 0
            torch.save(model.state_dict(), args.checkpoint_out)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"early stopping — no val improvement for {args.patience} epochs (best epoch {best_epoch}, val_acc={best_val_acc:.3f})")
                break

    total_dt = time.time() - t0

    # Reload the BEST checkpoint (not necessarily the last epoch) for test evaluation.
    model.load_state_dict(torch.load(args.checkpoint_out, map_location=device))
    test_loss, test_acc = _run_epoch(model, test_loader, device, criterion, optimizer=None)
    cm = _evaluate_confusion(model, test_loader, device, num_classes)

    per_class_metrics = {}
    for i, cls in enumerate(classes):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        fp = sum(cm[r][i] for r in range(num_classes)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) > 0 else None
        per_class_metrics[cls] = {
            "n_test_images": sum(cm[i]), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    import hashlib
    ckpt_sha = hashlib.sha256(args.checkpoint_out.read_bytes()).hexdigest()

    report = {
        "schema_version": "1.0",
        "phase": "16-B — panel-crop classifier (MobileNetV2)",
        "baseline_for_comparison": {
            "checkpoint": "weights/yolo_solar_condition_4class_research_v2.pt",
            "box_level_test_map50": 0.444,
            "box_level_test_recall": 0.441,
            "image_level_agreement": 0.727,
            "source": "docs/CONDITION_MODEL_4CLASS_V1.md, docs/CONDITION_MODEL_IMAGE_LEVEL_AGREEMENT.md",
        },
        "architecture": "torchvision MobileNetV2, ImageNet-pretrained, classifier head replaced (num_classes=4)",
        "classes_index_order": classes,
        "dataset": str(args.dataset),
        "train_class_counts": dict(zip(classes, counts)),
        "class_loss_weights": dict(zip(classes, weights.tolist())),
        "hyperparameters": {
            "epochs_requested": args.epochs, "epochs_run": len(history), "batch_size": args.batch_size,
            "lr": args.lr, "optimizer": "AdamW", "patience": args.patience, "seed": args.seed,
            "input_size": _INPUT_SIZE, "device": str(device),
        },
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "training_seconds": total_dt,
        "history": history,
        "test_set": {
            "n_images": len(test_ds),
            "overall_accuracy": test_acc,
            "overall_loss": test_loss,
            "confusion_matrix": cm,
            "confusion_matrix_class_order": classes,
            "per_class": per_class_metrics,
        },
        "checkpoint_path": str(args.checkpoint_out),
        "checkpoint_sha256": ckpt_sha,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nTEST accuracy: {test_acc:.3f}  (best epoch {best_epoch}, val_acc={best_val_acc:.3f})")
    print(f"confusion matrix (rows=true, cols=pred, order={classes}):")
    for row in cm:
        print("  ", row)
    for cls, m in per_class_metrics.items():
        print(f"  {cls:<16} n={m['n_test_images']:4d}  precision={m['precision']}  recall={m['recall']}  f1={m['f1']}")
    print(f"\ncheckpoint: {args.checkpoint_out}  sha256={ckpt_sha[:16]}...")
    print(f"report: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
