#!/usr/bin/env python3
"""scripts/infer_condition_on_panels.py — two-stage QUALITATIVE research
inference:

    original image
      -> STAGE 1: ground panel detector  (weights/yolo_solar_ground_research_v1.pt)
      -> panel crop  (per detected panel bbox, optional padding)
      -> STAGE 2: condition/defect model (weights/yolo_solar_condition_research_v1.pt)
      -> condition predictions per crop

The condition model is NOT a panel detector, so it is never run on the full
photograph and no claim is made that "the result represents panel-level
condition classification". It is run ONLY on the panel crops produced by
stage 1, and the full association chain is preserved.

QUALITATIVE ONLY. There is no ground truth for these images. This script does
NOT compute accuracy / precision / recall / F1 / mAP. A prediction is a raw
model output, NOT a fact. Absence of a detection is NOT "Clean". Images and
panels with zero detections are reported explicitly.

Hotspot is out of scope (needs a separate thermal/IR model) and is never
inferred from RGB.

    python scripts/infer_condition_on_panels.py \
        --panel-weights weights/yolo_solar_ground_research_v1.pt \
        --condition-weights weights/yolo_solar_condition_research_v1.pt \
        --images "D:/solar_ai_training_scratch/Solar Real Images" \
        --output-dir research_runs/ground_pv/condition_real_world
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_PANEL_COLOR = (56, 176, 255)
_COND_COLORS = {
    "CLEAN": (140, 200, 90), "DUSTY": (200, 170, 60), "BIRD_DROP": (255, 92, 92),
    "PHYSICAL_DAMAGE": (255, 140, 0), "ELECTRICAL_DAMAGE": (200, 60, 200),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _iter_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in _IMAGE_EXT:
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXT)


def _pad_box(x1, y1, x2, y2, pad, W, H):
    bw, bh = x2 - x1, y2 - y1
    return (max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh)),
            min(W, int(x2 + pad * bw)), min(H, int(y2 + pad * bh)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel-weights", type=Path, required=True)
    ap.add_argument("--condition-weights", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--panel-conf", type=float, default=0.25)
    ap.add_argument("--panel-iou", type=float, default=0.50)
    ap.add_argument("--panel-class", type=int, default=0, help="class id of 'panel' in the panel detector")
    ap.add_argument("--condition-conf", type=float, default=0.25)
    ap.add_argument("--condition-iou", type=float, default=0.50)
    ap.add_argument("--crop-pad", type=float, default=0.04)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args(argv)

    for w in (args.panel_weights, args.condition_weights):
        if not w.is_file():
            raise SystemExit(f"weights not found: {w}")
    images = _iter_images(args.images)
    if not images:
        raise SystemExit(f"no images found under {args.images}")

    from ultralytics import YOLO
    from PIL import Image, ImageDraw

    panel_model = YOLO(str(args.panel_weights))
    cond_model = YOLO(str(args.condition_weights))
    panel_names = panel_model.names if isinstance(panel_model.names, dict) else dict(enumerate(panel_model.names))
    cond_names = cond_model.names if isinstance(cond_model.names, dict) else dict(enumerate(cond_model.names))

    out = args.output_dir
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "annotated").mkdir(parents=True, exist_ok=True)
    (out / "per_image").mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    images_zero_panels: list[str] = []
    panels_zero_condition = 0
    total_panels = 0

    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        p_res = panel_model.predict(source=image, conf=args.panel_conf, iou=args.panel_iou,
                                    imgsz=args.imgsz, verbose=False)[0]
        panels = []
        for b in p_res.boxes:
            cid = int(b.cls.item())
            if cid != args.panel_class:
                continue
            xyxy = [float(v) for v in b.xyxy[0].tolist()]
            panels.append({"panel_bbox_xyxy": xyxy, "panel_confidence": float(b.conf.item())})
        panels.sort(key=lambda d: (-d["panel_confidence"], d["panel_bbox_xyxy"][0]))

        # also record any non-panel stage-1 detections (cracked/dusty) — literal, not fact
        stage1_other = []
        for b in p_res.boxes:
            cid = int(b.cls.item())
            if cid == args.panel_class:
                continue
            stage1_other.append({
                "class_name": panel_names.get(cid, str(cid)),
                "confidence": float(b.conf.item()),
                "bbox_xyxy": [float(v) for v in b.xyxy[0].tolist()],
            })

        if not panels:
            images_zero_panels.append(img_path.name)

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        panel_records = []
        for pi, p in enumerate(panels, 1):
            total_panels += 1
            x1, y1, x2, y2 = p["panel_bbox_xyxy"]
            cx1, cy1, cx2, cy2 = _pad_box(x1, y1, x2, y2, args.crop_pad, W, H)
            crop = image.crop((cx1, cy1, cx2, cy2))
            crop_name = f"{img_path.stem}__panel{pi:02d}.jpg"
            if not args.no_images:
                crop.save(out / "crops" / crop_name)

            c_res = cond_model.predict(source=crop, conf=args.condition_conf, iou=args.condition_iou,
                                       imgsz=args.imgsz, verbose=False)[0]
            preds = []
            for b in c_res.boxes:
                cid = int(b.cls.item())
                preds.append({
                    "class_name": cond_names.get(cid, str(cid)),
                    "class_id": cid,
                    "confidence": float(b.conf.item()),
                    "bbox_xyxy_in_crop": [float(v) for v in b.xyxy[0].tolist()],
                })
            preds.sort(key=lambda d: -d["confidence"])
            if not preds:
                panels_zero_condition += 1

            classes_present = sorted({d["class_name"] for d in preds})
            panel_records.append({
                "panel_id": pi,
                "panel_bbox_xyxy": p["panel_bbox_xyxy"],
                "panel_confidence": round(p["panel_confidence"], 4),
                "crop_bbox_xyxy": [cx1, cy1, cx2, cy2],
                "crop_file": crop_name,
                "condition_predictions": preds,
                "condition_classes_present": classes_present,
                "condition_detection_count": len(preds),
                "note": "raw model predictions on the panel crop — NOT a verified condition",
            })

            draw.rectangle([x1, y1, x2, y2], outline=_PANEL_COLOR, width=3)
            label = f"#{pi} " + ("/".join(classes_present) if classes_present else "(no condition prediction)")
            draw.text((x1 + 3, max(0, y1 - 12)), label, fill=_PANEL_COLOR)

        if not args.no_images:
            annotated.save(out / "annotated" / f"{img_path.stem}_annotated.jpg")

        record = {
            "image": img_path.name,
            "image_path": str(img_path),
            "image_size_wh": [W, H],
            "image_sha256": _sha256(img_path),
            "stage1_panel_detections": len(panels),
            "stage1_other_detections": stage1_other,
            "panels": panel_records,
            "qualitative_only": True,
        }
        records.append(record)
        (out / "per_image" / f"{img_path.stem}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    summary = {
        "pipeline": "image -> ground panel detector -> panel crop -> condition model",
        "panel_weights": str(args.panel_weights),
        "panel_weights_sha256": _sha256(args.panel_weights),
        "condition_weights": str(args.condition_weights),
        "condition_weights_sha256": _sha256(args.condition_weights),
        "config": {
            "panel_conf": args.panel_conf, "panel_iou": args.panel_iou,
            "condition_conf": args.condition_conf, "condition_iou": args.condition_iou,
            "crop_pad": args.crop_pad, "imgsz": args.imgsz,
        },
        "condition_classes": [cond_names[i] for i in sorted(cond_names)],
        "num_images": len(images),
        "num_images_with_zero_panels": len(images_zero_panels),
        "images_with_zero_panels": images_zero_panels,
        "total_panels_detected": total_panels,
        "panels_with_zero_condition_predictions": panels_zero_condition,
        "records": records,
        "disclaimer": (
            "QUALITATIVE ONLY. No ground truth. NO accuracy/precision/recall/F1/mAP is "
            "computed. A condition prediction (CLEAN / DUSTY / BIRD_DROP / PHYSICAL_DAMAGE / "
            "ELECTRICAL_DAMAGE) is a raw model output, NOT a fact. Absence of a detection is "
            "NOT 'Clean'. ELECTRICAL_DAMAGE carries a HIGH semantic risk — the source boxes "
            "visually correspond to cell cracks, not confirmed electrical failure. Hotspot is "
            "out of scope and never inferred from RGB."
        ),
    }
    (out / "condition_real_world_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"images: {len(images)}  images w/ 0 panels: {len(images_zero_panels)}  "
          f"panels: {total_panels}  panels w/ 0 condition preds: {panels_zero_condition}")
    print(f"summary: {out / 'condition_real_world_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
