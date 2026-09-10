#!/usr/bin/env python3
"""scripts/generate_annotation_proposals.py — Phase 14B.1 annotation-
assistance helper (optional, req 5: "if practical").

Runs the EXISTING ground panel detector
(weights/yolo_solar_ground_research_v1.pt, unmodified — reuses
training.detection.ground_pv.dynamic_panel_detection.run_panel_detector
from Phase 14A) over unannotated images and writes:

  1. an annotated JPEG per image with a large "PROPOSAL — NOT GROUND
     TRUTH" banner and every detected box drawn + labeled with its
     confidence, for quick human visual review, and
  2. a per-image JSON with the raw proposed boxes in the SAME
     normalized-YOLO coordinate convention as a real label file, so a
     human reviewer can copy/adjust them — but under a `.proposal.json`
     extension, in a SEPARATE directory from `annotations/`, specifically
     so it can never be mistaken for, or accidentally loaded as, a real
     `.txt` label file.

This script NEVER writes to the annotations/ label directory, NEVER sets
annotation_status, and NEVER trains or modifies anything. A proposal
becomes ground truth ONLY when a human draws/confirms the box themselves
in a real label file and the manifest entry is marked
annotation_status == "annotated_human_verified" — see
docs/FIRST_PARTY_COLLECTION_GUIDE.md.

Example
-------
    python scripts/generate_annotation_proposals.py \
        --images-dir "datasets/ground_pv_research/representative_panel_v1/raw/batch_001" \
        --weights weights/yolo_solar_ground_research_v1.pt \
        --output-dir datasets/ground_pv_research/representative_panel_v1/proposals/batch_001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.dynamic_panel_detection import (  # noqa: E402
    DynamicPanelConfig,
    run_panel_detector,
)

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_BANNER_COLOR = (255, 60, 60)
_BOX_COLOR = (255, 170, 0)


def _iter_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in _IMAGE_EXT:
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXT)


def _draw_proposal(image, detections) -> Any:
    from PIL import ImageDraw, ImageFont

    im = image.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for d in detections:
        x1, y1, x2, y2 = d.bbox
        draw.rectangle([x1, y1, x2, y2], outline=_BOX_COLOR, width=3)
        draw.text((x1 + 2, max(0, y1 - 16)), f"{d.confidence:.2f}", fill=_BOX_COLOR)

    banner_h = 28
    draw.rectangle([0, 0, im.width, banner_h], fill=(0, 0, 0))
    draw.text((6, 5), "PROPOSAL — NOT GROUND TRUTH (unverified model output)", fill=_BANNER_COLOR, font=font)
    return im


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.45,
                        help="Matches the documented real-world-holdout convention — see "
                             "docs/DYNAMIC_PANEL_DETECTION.md.")
    args = parser.parse_args(argv)

    if not args.weights.is_file():
        raise SystemExit(f"weights not found: {args.weights}")
    images = _iter_images(args.images_dir)
    if not images:
        raise SystemExit(f"no images found under {args.images_dir}")

    from ultralytics import YOLO
    from PIL import Image

    model = YOLO(str(args.weights))
    config = DynamicPanelConfig(confidence_threshold=args.conf)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        detections = run_panel_detector(
            model, image, conf=config.confidence_threshold, iou=config.iou_threshold, imgsz=config.imgsz,
        )

        annotated = _draw_proposal(image, detections)
        annotated.save(args.output_dir / f"{img_path.stem}.proposal.jpg")

        proposal = {
            "note": "PROPOSAL — NOT GROUND TRUTH. Raw, unverified model output. Every box must be "
                    "human-confirmed (or redrawn) in a real label file before it counts as an annotation.",
            "image": img_path.name,
            "model": args.weights.name,
            "confidence_threshold": args.conf,
            "proposed_boxes_yolo_format": [
                {
                    "class_id": 0, "class_name": "panel", "confidence": round(d.confidence, 4),
                    "cx": round(d.center_x / w, 6), "cy": round(d.center_y / h, 6),
                    "w": round(d.width / w, 6), "h": round(d.height / h, 6),
                }
                for d in detections
            ],
        }
        (args.output_dir / f"{img_path.stem}.proposal.json").write_text(
            json.dumps(proposal, indent=2), encoding="utf-8",
        )

    print(f"images: {len(images)}")
    print(f"wrote PROPOSAL (not ground truth) overlays + JSON to {args.output_dir}")
    print("Nothing here is a real annotation until a human confirms it in a label file "
          "and sets annotation_status='annotated_human_verified'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
