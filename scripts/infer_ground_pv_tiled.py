#!/usr/bin/env python3
"""scripts/infer_ground_pv_tiled.py — RESEARCH EXPERIMENT (Phase 10).

Compare FULL-FRAME vs OVERLAPPING-TILE inference for the ground-PV panel
detector, then run the condition/defect detector through both chains.

    photo
      -> [A] full-frame ground panel detector   (weights/yolo_solar_ground_research_v1.pt)
         [B] tiled     ground panel detector    (same checkpoint, tiled)
      -> panel crops
      -> condition/defect detector              (weights/yolo_solar_condition_research_v1.pt)

Nothing is retrained. No checkpoint is modified. QUALITATIVE ONLY — there is
no ground truth for these images, so NO accuracy / precision / recall / F1 /
mAP / confusion matrix is computed. A prediction is a raw model output, not a
fact. Absence of a condition prediction does NOT mean "Clean".

ELECTRICAL_DAMAGE is NOT validated electrical-fault detection — the source
audit found those annotations visually correspond to cell cracks. Hotspot is
out of scope and never inferred from RGB.

    python scripts/infer_ground_pv_tiled.py \
        --panel-weights weights/yolo_solar_ground_research_v1.pt \
        --condition-weights weights/yolo_solar_condition_research_v1.pt \
        --images "E:/Solar AI Project Data/first_party_photos/solar_real_images_6_holdout" \
        --output-dir research_runs/ground_pv/tiled_real_world
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.tiled_inference import (  # noqa: E402
    Detection,
    merge_detections,
    tiled_detect,
)

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_PANEL_COLOR = (56, 176, 255)
_COND_COLORS = {
    "CLEAN": (140, 200, 90), "DUSTY": (210, 170, 50), "BIRD_DROP": (255, 92, 92),
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


def _full_frame_panels(model, image, *, conf, iou, imgsz, panel_class):
    res = model.predict(source=image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    dets, others = [], []
    for b in res.boxes:
        cid = int(b.cls.item())
        rec = {"class_id": cid, "class_name": str(names.get(cid, cid)),
               "confidence": round(float(b.conf.item()), 4),
               "bbox": [round(float(v), 2) for v in b.xyxy[0].tolist()]}
        (dets if cid == panel_class else others).append(rec)
    dets.sort(key=lambda d: (-d["confidence"], d["bbox"][0]))
    return dets, others


def _condition_on_panels(cond_model, image, panels, *, conf, iou, imgsz, crop_pad, out_crop_dir, stem, chain):
    from PIL import Image  # noqa: F401
    W, H = image.size
    cond_names = cond_model.names if isinstance(cond_model.names, dict) else dict(enumerate(cond_model.names))
    panel_records = []
    n_with_pred = 0
    class_counts = {c: 0 for c in _COND_COLORS}
    confs = []
    contained_flags = []
    for pi, p in enumerate(panels, 1):
        x1, y1, x2, y2 = p["bbox"]
        cx1, cy1, cx2, cy2 = _pad_box(x1, y1, x2, y2, crop_pad, W, H)
        crop = image.crop((cx1, cy1, cx2, cy2))
        crop_w, crop_h = crop.size
        crop_name = f"{stem}__{chain}__panel{pi:02d}.jpg"
        crop.save(out_crop_dir / crop_name)
        res = cond_model.predict(source=crop, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        preds = []
        for b in res.boxes:
            cid = int(b.cls.item())
            bx = [float(v) for v in b.xyxy[0].tolist()]
            cn = str(cond_names.get(cid, cid))
            inside = (bx[0] >= -1 and bx[1] >= -1 and bx[2] <= crop_w + 1 and bx[3] <= crop_h + 1)
            contained_flags.append(inside)
            preds.append({
                "class_name": cn, "class_id": cid,
                "confidence": round(float(b.conf.item()), 4),
                "bbox_xyxy_in_crop": [round(v, 2) for v in bx],
                "crop_size_wh": [crop_w, crop_h],
                "spatially_contained_in_crop": bool(inside),
            })
            if cn in class_counts:
                class_counts[cn] += 1
            confs.append(round(float(b.conf.item()), 4))
        preds.sort(key=lambda d: -d["confidence"])
        if preds:
            n_with_pred += 1
        panel_records.append({
            "panel_id": pi,
            "panel_bbox_xyxy": [round(v, 2) for v in p["bbox"]],
            "panel_confidence": p["confidence"],
            "panel_source_tiles": p.get("source_tiles", []),
            "crop_bbox_xyxy": [cx1, cy1, cx2, cy2],
            "crop_file": crop_name,
            "condition_predictions": preds,
            "condition_classes_present": sorted({d["class_name"] for d in preds}),
            "note": "raw model predictions on the panel crop - NOT a verified condition",
        })
    return {
        "chain": chain,
        "n_panels": len(panels),
        "n_panels_with_condition_prediction": n_with_pred,
        "n_panels_with_zero_prediction": len(panels) - n_with_pred,
        "condition_class_counts": class_counts,
        "condition_confidence_range": [min(confs), max(confs)] if confs else None,
        "condition_boxes_spatially_contained_in_crop": (
            f"{sum(contained_flags)}/{len(contained_flags)}" if contained_flags else "0/0"),
        "panels": panel_records,
    }


def _draw(image, panels, color, title):
    from PIL import ImageDraw, ImageFont
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None
    d.rectangle([0, 0, im.width - 1, 16], fill=(0, 0, 0))
    d.text((3, 3), title, fill=(255, 255, 255), font=font)
    for i, p in enumerate(panels, 1):
        x1, y1, x2, y2 = p["bbox"]
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        d.text((x1 + 3, max(18, y1 + 2)), f"#{i} p{p['confidence']:.2f}", fill=color, font=font)
    return im


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel-weights", type=Path, required=True)
    ap.add_argument("--condition-weights", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--panel-conf", type=float, default=0.45,
                    help="ground detector confidence (0.45 = the documented real-world holdout config)")
    ap.add_argument("--panel-iou", type=float, default=0.50)
    ap.add_argument("--panel-class", type=int, default=0)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--condition-conf", type=float, default=0.25, help="matches Phase 9")
    ap.add_argument("--condition-iou", type=float, default=0.50)
    ap.add_argument("--crop-pad", type=float, default=0.04)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args(argv)

    for w in (args.panel_weights, args.condition_weights):
        if not w.is_file():
            raise SystemExit(f"weights not found: {w}")
    images = _iter_images(args.images)
    if not images:
        raise SystemExit(f"no images under {args.images}")

    from ultralytics import YOLO
    from PIL import Image

    panel_model = YOLO(str(args.panel_weights))
    cond_model = YOLO(str(args.condition_weights))
    panel_names = panel_model.names if isinstance(panel_model.names, dict) else dict(enumerate(panel_model.names))

    out = args.output_dir
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "compare").mkdir(parents=True, exist_ok=True)
    (out / "per_image").mkdir(parents=True, exist_ok=True)

    records = []
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        stem = img_path.stem

        # A. full-frame
        ff_panels, ff_others = _full_frame_panels(
            panel_model, image, conf=args.panel_conf, iou=args.panel_iou,
            imgsz=args.imgsz, panel_class=args.panel_class)

        # B. tiled (panel class only)
        tiled = tiled_detect(
            panel_model, image, tile=args.tile, overlap=args.overlap,
            conf=args.panel_conf, nms_iou=args.panel_iou, tile_nms_iou=args.panel_iou,
            imgsz=args.imgsz, class_filter=args.panel_class)
        tl_panels = [d for d in tiled["detections"] if d["class_id"] == args.panel_class]
        tl_panels.sort(key=lambda d: (-d["confidence"], d["bbox"][0]))
        # non-panel tiled detections (cracked/dusty) for the crack-FP investigation
        tiled_all = tiled_detect(
            panel_model, image, tile=args.tile, overlap=args.overlap,
            conf=args.panel_conf, nms_iou=args.panel_iou, tile_nms_iou=args.panel_iou,
            imgsz=args.imgsz, class_filter=None)
        tl_others = [d for d in tiled_all["detections"] if d["class_id"] != args.panel_class]

        # condition chains
        ff_cond = _condition_on_panels(
            cond_model, image, ff_panels, conf=args.condition_conf, iou=args.condition_iou,
            imgsz=args.imgsz, crop_pad=args.crop_pad, out_crop_dir=out / "crops", stem=stem, chain="fullframe")
        tl_cond = _condition_on_panels(
            cond_model, image, tl_panels, conf=args.condition_conf, iou=args.condition_iou,
            imgsz=args.imgsz, crop_pad=args.crop_pad, out_crop_dir=out / "crops", stem=stem, chain="tiled")

        # side-by-side visual
        orig = image.convert("RGB").copy()
        from PIL import ImageDraw
        ImageDraw.Draw(orig).rectangle([0, 0, orig.width - 1, 16], fill=(0, 0, 0))
        ImageDraw.Draw(orig).text((3, 3), "original", fill=(255, 255, 255))
        a = _draw(image, ff_panels, _PANEL_COLOR, f"full-frame: {len(ff_panels)} panel(s)")
        b = _draw(image, [{"bbox": d["bbox"], "confidence": d["confidence"]} for d in tl_panels],
                  (120, 230, 120), f"tiled: {len(tl_panels)} panel(s)")
        combo = Image.new("RGB", (orig.width * 3 + 16, orig.height), (20, 20, 20))
        combo.paste(orig, (0, 0)); combo.paste(a, (orig.width + 8, 0)); combo.paste(b, (orig.width * 2 + 16, 0))
        combo.save(out / "compare" / f"{stem}__compare.jpg")

        ff_confs = [p["confidence"] for p in ff_panels]
        tl_confs = [p["confidence"] for p in tl_panels]
        rec = {
            "image": img_path.name,
            "image_path": str(img_path),
            "image_size_wh": [W, H],
            "image_sha256": _sha256(img_path),
            "tile_config": tiled["tile_config"],
            "full_frame": {
                "panel_count": len(ff_panels),
                "panel_confidences": ff_confs,
                "panel_confidence_range": [min(ff_confs), max(ff_confs)] if ff_confs else None,
                "panels": ff_panels,
                "non_panel_detections": ff_others,
            },
            "tiled": {
                "panel_count": len(tl_panels),
                "panel_confidences": tl_confs,
                "panel_confidence_range": [min(tl_confs), max(tl_confs)] if tl_confs else None,
                "n_tiles": tiled["tile_config"]["n_tiles"],
                "raw_detection_count_before_merge": tiled["raw_detection_count"],
                "panels": tl_panels,
                "non_panel_detections": tl_others,
                "per_tile": tiled["tiles"],
            },
            "condition_chain_full_frame": ff_cond,
            "condition_chain_tiled": tl_cond,
            "qualitative_only": True,
        }
        records.append(rec)
        (out / "per_image" / f"{stem}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    summary = {
        "experiment": "Phase 10 - tiled ground-panel inference vs full-frame + two-stage condition",
        "panel_weights": str(args.panel_weights),
        "panel_weights_sha256": _sha256(args.panel_weights),
        "condition_weights": str(args.condition_weights),
        "condition_weights_sha256": _sha256(args.condition_weights),
        "config": {
            "panel_conf": args.panel_conf, "panel_iou_nms": args.panel_iou,
            "tile": args.tile, "overlap": args.overlap,
            "condition_conf": args.condition_conf, "condition_iou": args.condition_iou,
            "crop_pad": args.crop_pad, "imgsz": args.imgsz,
        },
        "panel_classes": [panel_names[i] for i in sorted(panel_names)],
        "condition_classes": ["CLEAN", "DUSTY", "BIRD_DROP", "PHYSICAL_DAMAGE", "ELECTRICAL_DAMAGE"],
        "num_images": len(images),
        "per_image_comparison": [
            {
                "image": r["image"],
                "dims": r["image_size_wh"],
                "full_frame_panels": r["full_frame"]["panel_count"],
                "tiled_panels": r["tiled"]["panel_count"],
                "full_frame_conf_range": r["full_frame"]["panel_confidence_range"],
                "tiled_conf_range": r["tiled"]["panel_confidence_range"],
                "full_frame_non_panel": r["full_frame"]["non_panel_detections"],
                "tiled_non_panel": r["tiled"]["non_panel_detections"],
            }
            for r in records
        ],
        "records": records,
        "disclaimer": (
            "QUALITATIVE ONLY. No ground truth for these six owner photos. NO "
            "accuracy / precision / recall / F1 / mAP / confusion matrix is computed. "
            "A prediction is a raw model output, NOT a fact. Absence of a condition "
            "prediction does NOT mean Clean. ELECTRICAL_DAMAGE is NOT validated "
            "electrical-fault detection (source boxes visually correspond to cell "
            "cracks). Hotspot is out of scope and never inferred from RGB."
        ),
    }
    (out / "tiled_real_world_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"images: {len(images)}")
    for r in records:
        print(f"  {r['image']:<46} full-frame {r['full_frame']['panel_count']}  ->  tiled {r['tiled']['panel_count']}")
    print(f"summary: {out / 'tiled_real_world_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
