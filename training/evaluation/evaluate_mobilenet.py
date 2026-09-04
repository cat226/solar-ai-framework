#!/usr/bin/env python3
"""training/evaluation/evaluate_mobilenet.py — Independent evaluation of the
real, released v1.0.0 MobileNet classifier (weights/mobilenet_solar_v1.pth)
against its untouched held-out classification test split.

Uses the actual production classifier wrapper (models.classifier.
SolarFaultClassifier), loaded via models.model_manager exactly as
services/pipeline.py does. Never mocks the model, never modifies weights,
config, or ground truth. This deliberately does not reuse
training/classification/evaluate_mobilenet.py - a fresh, independent
implementation, so a latent bug in the original script (if any) cannot
silently reproduce itself here.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image

from training.evaluation.common import (
    accuracy,
    confusion_matrix,
    default_output_root,
    macro_and_weighted_prf1,
    per_class_prf1,
    sha256_file,
)


def _confidence_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    sorted_v = sorted(values)
    n = len(sorted_v)
    median = sorted_v[n // 2] if n % 2 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    return {"count": n, "mean": sum(values) / n, "median": median, "min": min(values), "max": max(values)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="prepared/-style root (train/val/test/<Class>/)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit-per-class", type=int, default=None, help="Smoke-test only.")
    args = parser.parse_args()

    output_dir = args.output_dir or default_output_root()
    output_dir.mkdir(parents=True, exist_ok=True)

    from models.model_manager import model_manager
    from models.classifier import SolarFaultClassifier

    labels = model_manager.classifier_labels  # loads the classifier; real v1 labels
    source = model_manager.classifier_source
    weights_path = Path(model_manager.mobilenet_status["v1_path"]) if source == "v1" else Path(model_manager.mobilenet_status["six_class_path"])
    print(f"Evaluating MobileNet source={source} labels={labels} weights={weights_path} "
          f"(sha256={sha256_file(weights_path)[:16]}...)")

    classifier = SolarFaultClassifier()
    classifier.set_model(model_manager.get_classifier(), labels=labels)

    split_dir = args.data_root / args.split
    records: list[dict] = []
    t0 = time.time()
    for true_label in labels:
        class_dir = split_dir / true_label
        if not class_dir.is_dir():
            print(f"  WARNING: no directory for class {true_label!r} at {class_dir}")
            continue
        image_paths = sorted(p for p in class_dir.iterdir() if p.is_file())
        if args.limit_per_class is not None:
            image_paths = image_paths[: args.limit_per_class]
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            result = classifier.classify(image)
            records.append({
                "filename": img_path.name,
                "true_label": true_label,
                "predicted_label": result.label,
                "confidence": result.confidence,
                "correct": result.label == true_label,
                "probabilities": result.probabilities,
            })
    elapsed = time.time() - t0
    print(f"  classified {len(records)} images in {elapsed:.1f}s")

    y_true = [r["true_label"] for r in records]
    y_pred = [r["predicted_label"] for r in records]

    acc = accuracy(y_true, y_pred)
    per_class = per_class_prf1(y_true, y_pred, labels)
    agg = macro_and_weighted_prf1(per_class)
    cm = confusion_matrix(y_true, y_pred, labels)

    all_conf = [r["confidence"] for r in records]
    correct_conf = [r["confidence"] for r in records if r["correct"]]
    incorrect_conf = [r["confidence"] for r in records if not r["correct"]]
    conf_by_class = {
        label: _confidence_stats([r["confidence"] for r in records if r["true_label"] == label])
        for label in labels
    }

    errors = [r for r in records if not r["correct"]]
    sorted_correct = sorted((r for r in records if r["correct"]), key=lambda r: r["confidence"])
    sorted_incorrect = sorted((r for r in records if not r["correct"]), key=lambda r: r["confidence"], reverse=True)

    print(f"  accuracy={acc:.4f} macro_f1={agg['macro_f1']:.4f} weighted_f1={agg['weighted_f1']:.4f}")
    print(f"  errors: {len(errors)} / {len(records)}")
    print(f"  confusion matrix ({labels}): {cm}")

    summary = {
        "weights_path": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "classifier_source": source,
        "labels": labels,
        "dataset_split": args.split,
        "total_samples": len(records),
        "per_class_support": {label: per_class[label]["support"] for label in labels},
        "accuracy": acc,
        "per_class": per_class,
        "macro_weighted": agg,
        "confusion_matrix": cm,
        "confidence_stats": {
            "overall": _confidence_stats(all_conf),
            "correct_predictions": _confidence_stats(correct_conf),
            "incorrect_predictions": _confidence_stats(incorrect_conf),
            "by_true_class": conf_by_class,
        },
        "errors": errors,
        "lowest_confidence_correct": sorted_correct[:10],
        "highest_confidence_incorrect": sorted_incorrect[:10],
        "wall_seconds": elapsed,
    }

    summary_path = output_dir / f"mobilenet_{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    per_image_csv = output_dir / f"mobilenet_{args.split}_per_image.csv"
    with per_image_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "true_label", "predicted_label", "confidence", "correct"])
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print(f"\nWrote per-image CSV: {per_image_csv}")
    print(f"Wrote summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
