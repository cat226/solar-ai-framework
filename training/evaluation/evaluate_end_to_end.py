#!/usr/bin/env python3
"""training/evaluation/evaluate_end_to_end.py — Independent evaluation of the
real, released v1.0.0 production pipeline (detection -> panel crop ->
classification), using the actual SolarPanelDetector and
SolarFaultClassifier wrappers together, exactly as services/pipeline.py
composes them (crop via utils.image_utils.crop_panel).

Important, disclosed limitation of this specific evaluation (there is no
dataset with ground truth for both tasks simultaneously - see
docs/ML_EVALUATION_v1.0.0.md's "Leakage/similarity audit" and "End-to-end"
sections for the full explanation):

  - The classification test split (SolNET/PVMD) has real Clean/Dusty/Hotspot
    ground truth, but is a *different visual domain* than the YOLO detector's
    own training data (BDAPPV: aerial/satellite rooftop imagery). These are
    single-panel, ground-level/close-up photos. Ground-truth panel count is
    *assumed* to be exactly 1 per image (these datasets are collected and
    labeled as single-panel classification photos), not independently
    box-annotated - this assumption is stated explicitly, never hidden.
  - The YOLO test split (BDAPPV) has real panel-count/box ground truth, but
    no fault-class labels at all (BDAPPV is a detection-only dataset).

This script therefore evaluates the classification-split direction: real
end-to-end detection+classification correctness on real held-out photos
with real fault-class ground truth. It reports BOTH:
  - whole_image_classification: what a real user actually sees as the
    pipeline's classification_result, independent of whether YOLO detects a
    panel box in this out-of-domain photo (matches services/pipeline.py's
    behavior - whole-image classification always runs).
  - end_to_end_panel: the stricter, detection-dependent metric - a panel is
    "end_to_end_panel_correct" only when >=1 panel is detected AND the
    resulting crop's classification is correct - and is never true when
    detection finds 0 panels, however plausible the true label.
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

from training.evaluation.common import accuracy, confusion_matrix, default_output_root, macro_and_weighted_prf1, per_class_prf1, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Classification prepared/-style root (train/val/test/<Class>/)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit-per-class", type=int, default=None, help="Smoke-test only.")
    args = parser.parse_args()

    output_dir = args.output_dir or default_output_root()
    output_dir.mkdir(parents=True, exist_ok=True)

    from models.model_manager import model_manager
    from models.detector import SolarPanelDetector
    from models.classifier import SolarFaultClassifier
    from utils.image_utils import crop_panel
    from utils.config import CFG

    labels = model_manager.classifier_labels
    detector = SolarPanelDetector()
    detector.set_model(model_manager.get_detector())
    classifier = SolarFaultClassifier()
    classifier.set_model(model_manager.get_classifier(), labels=labels)

    yolo_sha = sha256_file(Path(CFG["models"]["yolo"]["weights"]))
    mn_status = model_manager.mobilenet_status
    mn_sha = sha256_file(Path(mn_status["v1_path"]))
    print(f"Detector sha256={yolo_sha[:16]}...  Classifier sha256={mn_sha[:16]}...  labels={labels}")

    split_dir = args.data_root / args.split
    records: list[dict] = []
    t0 = time.time()
    for true_label in labels:
        class_dir = split_dir / true_label
        if not class_dir.is_dir():
            continue
        image_paths = sorted(p for p in class_dir.iterdir() if p.is_file())
        if args.limit_per_class is not None:
            image_paths = image_paths[: args.limit_per_class]
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")

            det = detector.detect(image)
            whole_result = classifier.classify(image)

            best_panel_label = None
            best_panel_conf = None
            if det.panel_count > 0:
                best_idx = max(range(det.panel_count), key=lambda i: det.confidences[i])
                crop = crop_panel(image, tuple(det.boxes[best_idx]))
                panel_result = classifier.classify(crop)
                best_panel_label = panel_result.label
                best_panel_conf = panel_result.confidence

            assumed_gt_panel_count = 1  # see module docstring - disclosed assumption
            detection_correct = det.panel_count >= 1  # matches the assumed single-panel ground truth
            end_to_end_correct = detection_correct and best_panel_label == true_label

            records.append({
                "filename": img_path.name,
                "true_label": true_label,
                "assumed_ground_truth_panel_count": assumed_gt_panel_count,
                "predicted_panel_count": det.panel_count,
                "detection_correct": detection_correct,
                "whole_image_predicted_label": whole_result.label,
                "whole_image_confidence": whole_result.confidence,
                "whole_image_correct": whole_result.label == true_label,
                "best_panel_predicted_label": best_panel_label,
                "best_panel_confidence": best_panel_conf,
                "end_to_end_panel_correct": end_to_end_correct,
            })

    elapsed = time.time() - t0
    print(f"Evaluated {len(records)} images in {elapsed:.1f}s")

    y_true = [r["true_label"] for r in records]
    y_pred_whole = [r["whole_image_predicted_label"] for r in records]
    whole_acc = accuracy(y_true, y_pred_whole)
    whole_per_class = per_class_prf1(y_true, y_pred_whole, labels)
    whole_agg = macro_and_weighted_prf1(whole_per_class)
    whole_cm = confusion_matrix(y_true, y_pred_whole, labels)

    detection_correct_count = sum(1 for r in records if r["detection_correct"])
    detection_rate = detection_correct_count / len(records) if records else 0.0

    end_to_end_correct_count = sum(1 for r in records if r["end_to_end_panel_correct"])
    end_to_end_panel_accuracy = end_to_end_correct_count / len(records) if records else 0.0

    # Classification accuracy restricted to only the panels that were
    # actually (correctly) detected - i.e. conditional on detection success,
    # matching the task's "only evaluate classification for correctly
    # matched ground-truth panels" instruction.
    detected = [r for r in records if r["detection_correct"]]
    if detected:
        y_true_det = [r["true_label"] for r in detected]
        y_pred_det = [r["best_panel_predicted_label"] for r in detected]
        classification_given_detection_acc = accuracy(y_true_det, y_pred_det)
        classification_given_detection_per_class = per_class_prf1(y_true_det, y_pred_det, labels)
        classification_given_detection_agg = macro_and_weighted_prf1(classification_given_detection_per_class)
    else:
        classification_given_detection_acc = None
        classification_given_detection_per_class = None
        classification_given_detection_agg = None

    print(f"whole_image_classification_accuracy = {whole_acc:.4f}")
    print(f"detection_rate (panel_count>=1, assumed GT=1)  = {detection_rate:.4f} ({detection_correct_count}/{len(records)})")
    print(f"end_to_end_panel_accuracy (detect AND classify correct) = {end_to_end_panel_accuracy:.4f} ({end_to_end_correct_count}/{len(records)})")
    if classification_given_detection_acc is not None:
        print(f"classification_accuracy_given_detection_succeeded = {classification_given_detection_acc:.4f} (n={len(detected)})")

    summary = {
        "detector_sha256": yolo_sha,
        "classifier_sha256": mn_sha,
        "labels": labels,
        "dataset_split": args.split,
        "total_images": len(records),
        "assumptions": [
            "Ground-truth panel count is assumed to be exactly 1 per image "
            "(single-panel classification-dataset photos), not independently "
            "box-annotated.",
            "This dataset (SolNET/PVMD close-up photos) is a different visual "
            "domain than the YOLO detector's own training data (BDAPPV aerial "
            "rooftop imagery); detection performance here is not representative "
            "of YOLO's in-domain (aerial) performance - see the separate YOLO "
            "evaluation report for that.",
        ],
        "whole_image_classification": {
            "accuracy": whole_acc, "per_class": whole_per_class, "macro_weighted": whole_agg,
            "confusion_matrix": whole_cm,
        },
        "detection_rate_assumed_gt1": detection_rate,
        "end_to_end_panel_accuracy": end_to_end_panel_accuracy,
        "classification_accuracy_given_detection_succeeded": classification_given_detection_acc,
        "classification_given_detection_macro_weighted": classification_given_detection_agg,
        "wall_seconds": elapsed,
    }
    summary_path = output_dir / f"end_to_end_{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    per_image_csv = output_dir / f"end_to_end_{args.split}_per_image.csv"
    with per_image_csv.open("w", newline="") as f:
        fieldnames = ["filename", "true_label", "predicted_panel_count", "detection_correct",
                      "whole_image_predicted_label", "whole_image_confidence", "whole_image_correct",
                      "best_panel_predicted_label", "best_panel_confidence", "end_to_end_panel_correct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"\nWrote per-image CSV: {per_image_csv}")
    print(f"Wrote summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
