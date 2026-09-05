#!/usr/bin/env python3
"""training/evaluation/evaluate_yolo.py — Independent evaluation of the real,
released v1.0.0 YOLO detector against its untouched held-out test split.

Two distinct, clearly-labeled measurements are produced (they answer
different questions and must never be conflated):

1. Production-path metrics (Precision/Recall/TP/FP/FN/confidence stats):
   computed by running the actual application code
   (models.detector.SolarPanelDetector, via models.model_manager) at the
   real deployed configuration (conf=0.45, iou=0.50 from
   configs/settings.yaml), then greedily IoU-matching predictions to the
   real YOLO-format ground-truth labels. This answers "what does the
   deployed app actually produce."

2. mAP50 / mAP50-95: computed via ultralytics' own YOLO.val() on the same
   loaded checkpoint. mAP is a threshold-independent, precision-recall-curve
   integral - it is not something that can be validly derived from a single
   already-thresholded (conf=0.45) pass, so this uses the library's own
   internal low-confidence sweep, exactly like the original training run
   did. This is what's compared against the training-run's recorded
   mAP50/mAP50-95.

Never mocks the model. Never modifies weights, config, or ground truth.
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
    default_output_root,
    load_yolo_ground_truth,
    match_detections_to_ground_truth,
    sha256_file,
)
from utils.image_utils import unletterbox_box


def _write_data_yaml(data_root: Path, out_path: Path) -> Path:
    """Same shape as training/detection/train_yolo.py's _write_data_yaml -
    kept independent (not imported) so this evaluation script never depends
    on the training script's internals changing underneath it."""
    import yaml
    config = {
        "path": str(data_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {0: "solar panel"},
    }
    out_path.write_text(yaml.safe_dump(config))
    return out_path


def _run_production_path_eval(
    detector, data_root: Path, split: str, iou_threshold: float, limit: int | None = None
) -> dict:
    """Run models.detector.SolarPanelDetector.detect() (the real production
    code path) over every image in the split, matching against real
    ground-truth labels via greedy IoU matching.

    `limit`, when set, evaluates only the first N images (sorted, so still
    deterministic) - for a fast smoke check of this script itself, never
    for the recorded release metrics, which must always use the full split."""
    images_dir = data_root / split / "images"
    labels_dir = data_root / split / "labels"
    image_paths = sorted(images_dir.iterdir())
    if limit is not None:
        image_paths = image_paths[:limit]

    total_tp = total_fp = total_fn = 0
    all_confidences: list[float] = []
    matched_ious: list[float] = []
    per_image: list[dict] = []
    false_positive_examples: list[dict] = []
    false_negative_examples: list[dict] = []

    t0 = time.time()
    for i, img_path in enumerate(image_paths):
        image = Image.open(img_path).convert("RGB")
        result = detector.detect(image)

        # Boxes are in the letterboxed 640x640 canvas - map back to this
        # image's real pixel coordinates before comparing to ground truth,
        # using the exact same production utility utils/ui_helpers.py uses
        # to draw overlays on the original image.
        orig_size = (image.width, image.height)
        pred_boxes = [unletterbox_box(tuple(b), orig_size) for b in result.boxes]

        gt_boxes = load_yolo_ground_truth(labels_dir / f"{img_path.stem}.txt", image.width, image.height)

        tp, fp, fn, ious = match_detections_to_ground_truth(
            pred_boxes, result.confidences, gt_boxes, iou_threshold=iou_threshold
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_confidences.extend(result.confidences)
        matched_ious.extend(ious)

        per_image.append({
            "filename": img_path.name,
            "num_ground_truth": len(gt_boxes),
            "num_predicted": len(pred_boxes),
            "confidences": result.confidences,
            "tp": tp, "fp": fp, "fn": fn,
        })
        if fp > 0 and len(false_positive_examples) < 25:
            false_positive_examples.append({
                "filename": img_path.name, "num_ground_truth": len(gt_boxes),
                "predicted_boxes": pred_boxes, "confidences": result.confidences, "fp_count": fp,
            })
        if fn > 0 and len(false_negative_examples) < 25:
            false_negative_examples.append({
                "filename": img_path.name, "ground_truth_boxes": gt_boxes,
                "predicted_boxes": pred_boxes, "fn_count": fn,
            })

        if (i + 1) % 250 == 0:
            elapsed = time.time() - t0
            print(f"  ...{i + 1}/{len(image_paths)} images ({elapsed:.0f}s elapsed)", flush=True)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0

    return {
        "num_images": len(image_paths),
        "num_ground_truth_boxes": sum(r["num_ground_truth"] for r in per_image),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "confidence_stats": _confidence_stats(all_confidences),
        "matched_iou_stats": _confidence_stats(matched_ious) if matched_ious else None,
        "per_image": per_image,
        "false_positive_examples": false_positive_examples,
        "false_negative_examples": false_negative_examples,
        "wall_seconds": time.time() - t0,
    }


def _confidence_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    sorted_v = sorted(values)
    n = len(sorted_v)
    median = sorted_v[n // 2] if n % 2 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    return {
        "count": n, "mean": sum(values) / n, "median": median,
        "min": min(values), "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="yolo_prepared/-style root (train/val/test)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only: evaluate the first N images, not the full split.")
    parser.add_argument("--skip-map", action="store_true", help="Smoke-test only: skip the YOLO.val() mAP pass.")
    args = parser.parse_args()

    output_dir = args.output_dir or default_output_root()
    output_dir.mkdir(parents=True, exist_ok=True)

    from models.model_manager import model_manager
    from models.detector import SolarPanelDetector
    from utils.config import CFG

    yolo_weights = Path(CFG["models"]["yolo"]["weights"])
    conf = float(CFG["models"]["yolo"]["confidence_threshold"])
    iou_thresh = float(CFG["models"]["yolo"]["iou_threshold"])
    imgsz = int(CFG["models"]["yolo"]["image_size"])

    print(f"Evaluating {yolo_weights} (sha256={sha256_file(yolo_weights)[:16]}...) "
          f"at conf={conf}, iou={iou_thresh}, imgsz={imgsz}")

    detector = SolarPanelDetector()
    detector.set_model(model_manager.get_detector())

    print(f"\n[1/2] Production-path evaluation (models.detector.SolarPanelDetector, conf={conf})...")
    production_results = _run_production_path_eval(detector, args.data_root, args.split, iou_thresh, limit=args.limit)
    print(f"  precision={production_results['precision']:.4f} recall={production_results['recall']:.4f} "
          f"TP={production_results['true_positives']} FP={production_results['false_positives']} "
          f"FN={production_results['false_negatives']}")

    if args.skip_map:
        print("\n[2/2] Skipped (--skip-map).")
        map_results = None
    else:
        print(f"\n[2/2] mAP50/mAP50-95 via ultralytics YOLO.val() (library's own PR-curve integral, "
              f"not derivable from the single conf={conf} pass above)...")
        data_yaml = _write_data_yaml(args.data_root, output_dir / "data.yaml")
        raw_model = model_manager.get_detector()
        val_metrics = raw_model.val(data=str(data_yaml), split=args.split, imgsz=imgsz, iou=iou_thresh, plots=False, save_json=False)
        map_results = {
            "precision_at_val_default_conf": float(val_metrics.box.mp),
            "recall_at_val_default_conf": float(val_metrics.box.mr),
            "mAP50": float(val_metrics.box.map50),
            "mAP50_95": float(val_metrics.box.map),
        }
        print(f"  mAP50={map_results['mAP50']:.4f} mAP50-95={map_results['mAP50_95']:.4f}")

    # Full per-image records -> E: drive (large; not committed to git).
    per_image_csv = output_dir / f"yolo_{args.split}_per_image.csv"
    with per_image_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "num_ground_truth", "num_predicted", "tp", "fp", "fn"])
        writer.writeheader()
        for row in production_results["per_image"]:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    summary = {
        "weights_path": str(yolo_weights),
        "weights_sha256": sha256_file(yolo_weights),
        "config": {"confidence_threshold": conf, "iou_threshold": iou_thresh, "image_size": imgsz},
        "dataset_split": args.split,
        "production_path_metrics": {k: v for k, v in production_results.items() if k not in ("per_image",)},
        "map_metrics": map_results,
        "per_image_csv": str(per_image_csv),
    }
    summary_path = output_dir / f"yolo_{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote per-image CSV: {per_image_csv}")
    print(f"Wrote summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
