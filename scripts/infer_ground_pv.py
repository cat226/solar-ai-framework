#!/usr/bin/env python3
"""scripts/infer_ground_pv.py — Research inference utility for the
ground-level PV defect detector.

Given arbitrary RGB rooftop / close-up images, this:
  1. runs the research detector (4 classes: panel / bird_drop / cracked / dusty),
  2. associates each defect detection with a panel detection
     (training/detection/ground_pv/panel_condition.assess_image),
  3. derives a per-panel RESEARCH visual condition assessment
     (Clean / Dust / BirdDrop / Damage — a panel may carry several),
  4. writes an annotated image and a per-image JSON record,
  5. records images with ZERO detections explicitly.

This is "visual defect detection" / "research visual condition assessment",
NOT a validated engineering or safety diagnosis.

Does not touch the production model.

Examples
--------
    python scripts/infer_ground_pv.py \
        --weights weights/yolo_solar_ground_research_v1.pt \
        --images "E:/Solar Real Images" \
        --output-dir research_runs/ground_pv/real_world_predictions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.class_map import RESEARCH_CLASS_ORDER  # noqa: E402
from training.detection.ground_pv.panel_condition import assess_image  # noqa: E402

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_COLORS = {
    "panel": (56, 176, 255), "bird_drop": (255, 92, 92),
    "cracked": (255, 179, 0), "dusty": (140, 200, 90),
}


def _iter_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in _IMAGE_EXT:
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXT)


def _draw(image, detections, assessment, out_path: Path) -> None:
    from PIL import ImageDraw
    im = image.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = _COLORS.get(d["class_name"], (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 12)), f"{d['class_name']} {d['confidence']:.2f}", fill=color)
    for pa in assessment["panels"]:
        x1, y1, x2, y2 = pa["panel_bbox"]
        draw.text((x1 + 2, y2 + 2), f"#{pa['panel_id']}: {'/'.join(pa['condition'])}", fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Image file or directory (recursed).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-assoc-iou", type=float, default=0.10)
    parser.add_argument("--no-images", action="store_true", help="Write JSON only, skip annotated images.")
    args = parser.parse_args(argv)

    if not args.weights.is_file():
        raise SystemExit(f"weights not found: {args.weights}")
    images = _iter_images(args.images)
    if not images:
        raise SystemExit(f"no images found under {args.images}")

    from ultralytics import YOLO
    from PIL import Image

    model = YOLO(str(args.weights))
    model_names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    zero_detection: list[str] = []

    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        res = model.predict(source=image, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        detections: list[dict] = []
        for b in res.boxes:
            cls_id = int(b.cls.item())
            name = model_names.get(cls_id, str(cls_id))
            xyxy = [float(v) for v in b.xyxy[0].tolist()]
            detections.append({
                "class_name": name,
                "class_id": cls_id,
                "confidence": float(b.conf.item()),
                "bbox": xyxy,
            })
        assessment = assess_image(detections, min_iou=args.min_assoc_iou)

        if not detections:
            zero_detection.append(img_path.name)

        record = {
            "image": img_path.name,
            "image_path": str(img_path),
            "image_size": [image.width, image.height],
            "num_detections": len(detections),
            "detections": detections,
            "detections_by_class": {
                c: sum(1 for d in detections if d["class_name"] == c) for c in RESEARCH_CLASS_ORDER
            },
            "panel_assessment": assessment,
        }
        records.append(record)
        (args.output_dir / f"{img_path.stem}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        if not args.no_images:
            _draw(image, detections, assessment, args.output_dir / f"{img_path.stem}_annotated.jpg")

    summary = {
        "weights": str(args.weights),
        "num_images": len(images),
        "num_images_with_detections": len(images) - len(zero_detection),
        "num_images_zero_detections": len(zero_detection),
        "images_zero_detections": zero_detection,
        "config": {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
                   "min_assoc_iou": args.min_assoc_iou},
        "records": records,
        "disclaimer": "Research visual condition assessment — not a validated diagnosis. "
                      "No ground truth; no TP/FP/FN reported.",
    }
    summary_path = args.output_dir / "inference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"images: {len(images)}   with detections: {summary['num_images_with_detections']}   "
          f"zero detections: {summary['num_images_zero_detections']}")
    print(f"wrote per-image JSON + annotated images to {args.output_dir}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
