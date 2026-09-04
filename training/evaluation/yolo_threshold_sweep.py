#!/usr/bin/env python3
"""training/evaluation/yolo_threshold_sweep.py — YOLO confidence-threshold
operating-point analysis for the real, released v1.0.0 detector, run on the
VALIDATION split only (never the test split, so this analysis can never be
test-set threshold optimization).

Methodology (standard, legitimate technique - the same one ultralytics'
own YOLO.val() uses internally to build a PR curve): the raw ultralytics
model is queried ONCE per image at a very low confidence (conf=0.001,
matching ultralytics' own internal validation convention) with NMS at the
real production IoU (iou=0.50 from configs/settings.yaml - never varied;
this script only sweeps confidence). The resulting candidate detections are
then filtered post-hoc at each threshold in the grid and re-matched against
ground truth, avoiding 18 separate full-dataset inference passes while
still reflecting the real model + real production IoU setting.

Object-size buckets (tiny/small/medium/large) are derived from the
validation split's own real ground-truth box-area quartiles (computed by
this script from real data - not hand-picked), so the buckets are
transparent and reproducible, not cherry-picked to favor any particular
narrative.

Never touches the test split, never modifies configs/settings.yaml, never
retrains or modifies model weights.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image

from training.evaluation.common import (
    default_output_root,
    iou as _iou,
    load_yolo_ground_truth,
    sha256_file,
)
from utils.image_utils import pil_to_numpy, resize_for_yolo, unletterbox_box

_DEFAULT_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                  0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

_LOW_CONF = 0.001  # ultralytics' own internal validation convention


def _collect_candidates(raw_model, iou_thresh: float, imgsz: int, data_root: Path, split: str, limit: int | None = None):
    """One low-confidence pass per image; returns a list of per-image dicts
    with all candidate detections (post-NMS at the real production IoU) and
    real ground-truth boxes, in original-image pixel coordinates.

    `limit`, when set, collects only the first N images (sorted, so still
    deterministic) - for a fast smoke check of this script itself, never
    for the recorded threshold-selection analysis."""
    images_dir = data_root / split / "images"
    labels_dir = data_root / split / "labels"
    image_paths = sorted(images_dir.iterdir())
    if limit is not None:
        image_paths = image_paths[:limit]

    records = []
    t0 = time.time()
    for i, img_path in enumerate(image_paths):
        image = Image.open(img_path).convert("RGB")
        img_resized = resize_for_yolo(image)
        img_array = pil_to_numpy(img_resized)
        raw = raw_model(img_array, conf=_LOW_CONF, iou=iou_thresh, imgsz=imgsz, verbose=False)

        boxes, confs = [], []
        for pred in raw:
            if not pred.boxes:
                continue
            xyxy = pred.boxes.xyxy.cpu().numpy()
            conf_arr = pred.boxes.conf.cpu().numpy()
            for box, conf in zip(xyxy, conf_arr):
                orig_box = unletterbox_box(tuple(box.tolist()), (image.width, image.height))
                boxes.append(orig_box)
                confs.append(float(conf))

        gt_boxes = load_yolo_ground_truth(labels_dir / f"{img_path.stem}.txt", image.width, image.height)
        gt_areas = [
            ((x2 - x1) * (y2 - y1)) / (image.width * image.height)
            for (x1, y1, x2, y2) in gt_boxes
        ]
        records.append({
            "filename": img_path.name, "boxes": boxes, "confidences": confs,
            "gt_boxes": gt_boxes, "gt_areas": gt_areas,
        })
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(image_paths)} images ({time.time() - t0:.0f}s elapsed)", flush=True)

    print(f"  candidate collection done: {len(image_paths)} images in {time.time() - t0:.0f}s")
    return records


def _size_bucket_boundaries(records: list[dict]) -> dict:
    all_areas = sorted(a for r in records for a in r["gt_areas"])
    n = len(all_areas)
    if n == 0:
        return {"p25": 0.0, "p50": 0.0, "p75": 0.0}
    return {
        "p25": all_areas[int(n * 0.25)],
        "p50": all_areas[int(n * 0.50)],
        "p75": all_areas[int(n * 0.75)],
    }


def _bucket_of(area: float, boundaries: dict) -> str:
    if area < boundaries["p25"]:
        return "tiny"
    if area < boundaries["p50"]:
        return "small"
    if area < boundaries["p75"]:
        return "medium"
    return "large"


def _sweep(records: list[dict], grid: list[float], iou_thresh: float, boundaries: dict) -> dict:
    per_threshold = []
    for conf_thresh in grid:
        tp = fp = fn = 0
        bucket_tp = {"tiny": 0, "small": 0, "medium": 0, "large": 0}
        bucket_total = {"tiny": 0, "small": 0, "medium": 0, "large": 0}
        detection_count = 0
        for r in records:
            keep_idx = [i for i, c in enumerate(r["confidences"]) if c >= conf_thresh]
            pred_boxes = [r["boxes"][i] for i in keep_idx]
            pred_conf = [r["confidences"][i] for i in keep_idx]
            detection_count += len(pred_boxes)

            claimed = [False] * len(r["gt_boxes"])
            order = sorted(range(len(pred_boxes)), key=lambda i: pred_conf[i], reverse=True)
            for i in order:
                best_iou, best_j = 0.0, -1
                for j, gt in enumerate(r["gt_boxes"]):
                    if claimed[j]:
                        continue
                    cur = _iou(pred_boxes[i], gt)
                    if cur > best_iou:
                        best_iou, best_j = cur, j
                if best_j >= 0 and best_iou >= 0.5:
                    claimed[best_j] = True
                    tp += 1
                    bucket = _bucket_of(r["gt_areas"][best_j], boundaries)
                    bucket_tp[bucket] += 1
                else:
                    fp += 1
            for j, gt_area in enumerate(r["gt_areas"]):
                bucket = _bucket_of(gt_area, boundaries)
                bucket_total[bucket] += 1
            fn += claimed.count(False)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        recall_by_bucket = {
            b: (bucket_tp[b] / bucket_total[b] if bucket_total[b] else None)
            for b in bucket_total
        }
        per_threshold.append({
            "confidence_threshold": conf_thresh, "iou_threshold": iou_thresh,
            "tp": tp, "fp": fp, "fn": fn, "detection_count": detection_count,
            "gt_count": sum(bucket_total.values()),
            "precision": precision, "recall": recall, "f1": f1,
            "recall_by_size_bucket": recall_by_bucket,
            "gt_count_by_size_bucket": bucket_total,
        })
    return {"grid": grid, "per_threshold": per_threshold}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val", help="Must be val, never test, for threshold selection.")
    parser.add_argument("--grid", type=float, nargs="+", default=_DEFAULT_GRID)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only.")
    args = parser.parse_args()

    if args.split == "test":
        raise SystemExit(
            "Refusing to run threshold selection against the test split - this script is for "
            "the validation split only, so threshold analysis never becomes test-set optimization. "
            "Pass --split val (default)."
        )

    output_dir = args.output_dir or default_output_root()
    output_dir.mkdir(parents=True, exist_ok=True)

    from models.model_manager import model_manager
    from utils.config import CFG

    yolo_weights = Path(CFG["models"]["yolo"]["weights"])
    iou_thresh = float(CFG["models"]["yolo"]["iou_threshold"])
    imgsz = int(CFG["models"]["yolo"]["image_size"])
    production_conf = float(CFG["models"]["yolo"]["confidence_threshold"])

    print(f"YOLO threshold sweep on split={args.split} (weights sha256={sha256_file(yolo_weights)[:16]}...)")
    print(f"Fixed production iou={iou_thresh}, imgsz={imgsz} (never swept). Deployed conf={production_conf}.")

    raw_model = model_manager.get_detector()
    records = _collect_candidates(raw_model, iou_thresh, imgsz, args.data_root, args.split, limit=args.limit)

    boundaries = _size_bucket_boundaries(records)
    print(f"Size-bucket boundaries (from real {args.split}-split GT box-area quartiles): {boundaries}")

    sweep = _sweep(records, args.grid, iou_thresh, boundaries)

    best_f1 = max(sweep["per_threshold"], key=lambda r: r["f1"])
    print(f"\nBest-F1 threshold on {args.split}: conf={best_f1['confidence_threshold']} "
          f"(P={best_f1['precision']:.4f} R={best_f1['recall']:.4f} F1={best_f1['f1']:.4f})")
    deployed_row = next(r for r in sweep["per_threshold"] if abs(r["confidence_threshold"] - production_conf) < 1e-9) \
        if production_conf in args.grid else None
    if deployed_row:
        print(f"Deployed threshold conf={production_conf}: "
              f"P={deployed_row['precision']:.4f} R={deployed_row['recall']:.4f} F1={deployed_row['f1']:.4f}")

    for row in sweep["per_threshold"]:
        print(f"  conf={row['confidence_threshold']:.2f}  P={row['precision']:.4f}  R={row['recall']:.4f}  "
              f"F1={row['f1']:.4f}  TP={row['tp']}  FP={row['fp']}  FN={row['fn']}  det={row['detection_count']}")

    summary = {
        "weights_path": str(yolo_weights),
        "weights_sha256": sha256_file(yolo_weights),
        "split": args.split,
        "fixed_iou_threshold": iou_thresh,
        "fixed_imgsz": imgsz,
        "deployed_production_confidence_threshold": production_conf,
        "size_bucket_boundaries": boundaries,
        "size_bucket_derivation": (
            f"25th/50th/75th percentile of real ground-truth box normalized area "
            f"on the {args.split} split ({sum(len(r['gt_areas']) for r in records)} boxes total) - "
            "not hand-picked."
        ),
        "best_f1_threshold": best_f1,
        "sweep": sweep,
    }
    out_path = output_dir / f"yolo_threshold_sweep_{args.split}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
