#!/usr/bin/env python3
"""scripts/detect_dynamic_panels.py — Dynamic multi-panel detection CLI
(Phase 14A).

Runs the ground panel detector on a full camera image (or a directory of
them) and produces, per image:
  1. a machine-readable JSON report with a dynamically-sized `panels` list
     (P1..PN — N is never assumed, never hard-coded to 4),
  2. an optional annotated image (bounding boxes + panel id + confidence +
     "Panels detected: N"),
  3. geometry / overlap / size sanity diagnostics distinguishing valid vs
     suspicious detections, and an explicit `failed` status for images with
     zero accepted panel detections.

This is panel LOCALIZATION + spatial ordering only. No condition
classification (Clean/Dusty/BirdDrop/PhysicalDamage/ElectricalDamage) is
performed or inferred — see training/detection/ground_pv/panel_condition.py
for that separate (and separately-invoked) concern.

Does not touch, modify, or retrain any checkpoint. Read-only against the
model weights and the source images.

Examples
--------
    # Single image
    python scripts/detect_dynamic_panels.py \
        --source "D:/path/to/image.jpg" \
        --weights weights/yolo_solar_ground_research_v1.pt \
        --output-dir datasets/ground_pv_research/reports/dynamic_detection

    # Directory (batch / qualitative smoke test)
    python scripts/detect_dynamic_panels.py \
        --source "E:/Solar AI Project Data/first_party_photos/solar_real_images_6_holdout" \
        --weights weights/yolo_solar_ground_research_v1.pt \
        --output-dir research_runs/ground_pv/dynamic_detection_real_world \
        --save-annotated
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.dynamic_panel_detection import (  # noqa: E402
    DynamicPanelConfig,
    detect_dynamic_panels,
    draw_annotated,
)

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _iter_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in _IMAGE_EXT:
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Image file or directory (recursed).")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.45,
                        help="Panel-class confidence threshold (default 0.45 — matches the "
                             "documented real-world holdout convention; not silently lowered "
                             "to inflate panel count).")
    parser.add_argument("--iou", type=float, default=0.50, help="YOLO's own NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--row-tolerance", type=float, default=0.10,
                        help="Row-grouping tolerance as a fraction of image height.")
    parser.add_argument("--overlap-warning-iou", type=float, default=0.30,
                        help="Post-NMS residual-overlap diagnostic threshold.")
    parser.add_argument("--containment-warning-fraction", type=float, default=0.80)
    parser.add_argument("--min-relative-area", type=float, default=0.001)
    parser.add_argument("--max-relative-area", type=float, default=0.60)
    parser.add_argument("--min-aspect-ratio", type=float, default=0.20)
    parser.add_argument("--max-aspect-ratio", type=float, default=5.0)
    parser.add_argument("--size-inconsistency-ratio", type=float, default=3.0)
    parser.add_argument("--isolation-distance-multiplier", type=float, default=3.0)
    parser.add_argument("--save-annotated", action="store_true", help="Write an annotated image per input image.")
    parser.add_argument("--save-json", action="store_true", default=True,
                        help="Write a per-image JSON report (default: on).")
    parser.add_argument("--no-json", dest="save_json", action="store_false",
                        help="Skip per-image JSON (summary JSON is still written).")
    args = parser.parse_args(argv)

    if not args.weights.is_file():
        raise SystemExit(f"weights not found: {args.weights}")
    images = _iter_images(args.source)
    if not images:
        raise SystemExit(f"no images found under {args.source}")

    from ultralytics import YOLO
    from PIL import Image

    model = YOLO(str(args.weights))
    weights_sha256 = _sha256(args.weights)

    config = DynamicPanelConfig(
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        row_grouping_tolerance=args.row_tolerance,
        overlap_warning_iou=args.overlap_warning_iou,
        containment_warning_fraction=args.containment_warning_fraction,
        min_relative_area=args.min_relative_area,
        max_relative_area=args.max_relative_area,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        size_inconsistency_ratio=args.size_inconsistency_ratio,
        isolation_distance_multiplier=args.isolation_distance_multiplier,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    zero_detection: list[str] = []
    warning_images: list[str] = []

    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        report = detect_dynamic_panels(
            model, image, image_name=img_path.name, model_name=args.weights.name, config=config,
        )
        report["image_path"] = str(img_path)
        report["image_sha256"] = _sha256(img_path)

        if report["analysis_status"] == "failed":
            zero_detection.append(img_path.name)
        elif report["analysis_status"] == "warning":
            warning_images.append(img_path.name)

        records.append(report)
        if args.save_json:
            (args.output_dir / f"{img_path.stem}_dynamic_panels.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8",
            )
        if args.save_annotated:
            annotated = draw_annotated(image, report)
            annotated.save(args.output_dir / f"{img_path.stem}_dynamic_panels_annotated.jpg")

    panel_counts = [r["panel_count"] for r in records]
    summary = {
        "schema_version": "1.0",
        "phase": "14A — dynamic panel detection (localization + spatial ordering only)",
        "weights": str(args.weights),
        "weights_sha256": weights_sha256,
        "num_images": len(images),
        "num_images_ok": sum(1 for r in records if r["analysis_status"] == "ok"),
        "num_images_warning": len(warning_images),
        "num_images_failed_zero_detection": len(zero_detection),
        "images_zero_detection": zero_detection,
        "images_with_warnings": warning_images,
        "panel_count_by_image": {r["image"]: r["panel_count"] for r in records},
        "panel_count_min": min(panel_counts) if panel_counts else None,
        "panel_count_max": max(panel_counts) if panel_counts else None,
        "config": config.as_dict(),
        "records": records,
        "disclaimer": (
            "Dynamic panel LOCALIZATION and spatial ordering only — no condition "
            "classification (Clean/Dusty/BirdDrop/PhysicalDamage/ElectricalDamage). "
            "panel_count and panel IDs (P1..PN) are derived per-image; N is never "
            "assumed to be 4. Panel IDs are spatial identities for THIS capture only, "
            "not yet persistent physical identities across captures over time. "
            "Zero detections is reported as a failure to detect, not as 'no panels "
            "exist'. No accuracy/precision/recall/mAP is computed unless ground-truth "
            "annotations exist for the input images."
        ),
    }
    summary_path = args.output_dir / "dynamic_panel_detection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"images: {len(images)}   ok: {summary['num_images_ok']}   "
          f"warning: {summary['num_images_warning']}   "
          f"zero-detection: {summary['num_images_failed_zero_detection']}")
    for r in records:
        print(f"  {r['image']:<50} panel_count={r['panel_count']:<3} status={r['analysis_status']}")
    print(f"wrote per-image reports + summary to {args.output_dir}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
