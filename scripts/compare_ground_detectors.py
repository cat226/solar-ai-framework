#!/usr/bin/env python3
"""scripts/compare_ground_detectors.py — RESEARCH (Phase 11).

Full-frame comparison of two ground PANEL detectors (v1 vs v2) on the six
stable owner photos, plus an optional tiled FALLBACK (tiling only when
full-frame returns 0-1 panels), plus an optional condition-chain pass.

QUALITATIVE ONLY. No ground truth for these six photos -> NO accuracy /
precision / recall / F1 / mAP / confusion matrix. A prediction is a raw model
output, not a fact. Absence of a condition prediction does NOT mean Clean.
ELECTRICAL_DAMAGE is NOT validated electrical-fault detection (source boxes
visually correspond to cell cracks). Hotspot is out of scope, never inferred.

    python scripts/compare_ground_detectors.py \
        --v1 weights/yolo_solar_ground_research_v1.pt \
        --v2 weights/yolo_solar_ground_research_v2.pt \
        --images "D:/solar_ai_training_scratch/Solar Real Images" \
        --output-dir research_runs/ground_pv/v2_real_world \
        --condition-weights weights/yolo_solar_condition_research_v1.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.tiled_inference import tiled_detect  # noqa: E402

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_V1_COLOR = (56, 176, 255)
_V2_COLOR = (120, 230, 120)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _iter_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in _IMAGE_EXT:
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXT)


def _full_frame(model, image, *, conf, iou, imgsz, panel_class):
    res = model.predict(source=image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    panels, others = [], []
    for b in res.boxes:
        cid = int(b.cls.item())
        rec = {"class_id": cid, "class_name": str(names.get(cid, cid)),
               "confidence": round(float(b.conf.item()), 4),
               "bbox": [round(float(v), 1) for v in b.xyxy[0].tolist()]}
        (panels if cid == panel_class else others).append(rec)
    panels.sort(key=lambda d: (-d["confidence"], d["bbox"][0]))
    return panels, others


def _area_frac(bbox, W, H):
    x1, y1, x2, y2 = bbox
    return round(max(0.0, x2 - x1) * max(0.0, y2 - y1) / (W * H), 3)


def _localization_notes(panels, W, H):
    """Heuristic qualitative flags (NOT metrics)."""
    if not panels:
        return ["0 panels"]
    notes = []
    fracs = [_area_frac(p["bbox"], W, H) for p in panels]
    notes.append(f"largest panel covers {max(fracs):.0%} of frame")
    # fragmentation: >=2 same-ish-x boxes stacked vertically covering one column
    xs = [(p["bbox"][0], p["bbox"][2]) for p in panels]
    stacked = sum(1 for i in range(len(xs)) for j in range(i + 1, len(xs))
                  if abs(xs[i][0] - xs[j][0]) < 0.15 * W and abs(xs[i][1] - xs[j][1]) < 0.15 * W)
    if stacked:
        notes.append(f"{stacked} vertically-stacked same-column box pair(s) (possible fragmentation)")
    # overlap groups
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0
    ov = sum(1 for i in range(len(panels)) for j in range(i + 1, len(panels))
             if _iou(panels[i]["bbox"], panels[j]["bbox"]) > 0.25)
    if ov:
        notes.append(f"{ov} overlapping panel-box pair(s) (possible group boxes)")
    covered = sum((p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]) for p in panels)
    notes.append(f"union box area ~{covered / (W * H):.0%} of frame (rough)")
    return notes


def _draw(image, panels, color, title):
    from PIL import ImageDraw
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width - 1, 16], fill=(0, 0, 0))
    d.text((3, 3), title, fill=(255, 255, 255))
    for i, p in enumerate(panels, 1):
        x1, y1, x2, y2 = p["bbox"]
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        d.text((x1 + 3, max(18, y1 + 2)), f"#{i} {p['confidence']:.2f}", fill=color)
    return im


def _condition_chain(cond_model, image, panels, *, conf, iou, imgsz, crop_pad, out_dir, stem, tag):
    W, H = image.size
    names = cond_model.names if isinstance(cond_model.names, dict) else dict(enumerate(cond_model.names))
    counts = {n: 0 for n in names.values()}
    n_pred = 0
    per_panel = []
    for pi, p in enumerate(panels, 1):
        x1, y1, x2, y2 = p["bbox"]
        bw, bh = x2 - x1, y2 - y1
        cx1 = max(0, int(x1 - crop_pad * bw)); cy1 = max(0, int(y1 - crop_pad * bh))
        cx2 = min(W, int(x2 + crop_pad * bw)); cy2 = min(H, int(y2 + crop_pad * bh))
        crop = image.crop((cx1, cy1, cx2, cy2))
        crop.save(out_dir / f"{stem}__{tag}__panel{pi:02d}.jpg")
        res = cond_model.predict(source=crop, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        preds = []
        for b in res.boxes:
            cn = str(names.get(int(b.cls.item()), int(b.cls.item())))
            preds.append({"class_name": cn, "confidence": round(float(b.conf.item()), 4)})
            if cn in counts:
                counts[cn] += 1
        preds.sort(key=lambda d: -d["confidence"])
        if preds:
            n_pred += 1
        per_panel.append({"panel_id": pi, "panel_confidence": p["confidence"],
                          "condition_predictions": preds})
    return {"tag": tag, "n_panels": len(panels), "n_panels_with_prediction": n_pred,
            "class_counts": {k: v for k, v in counts.items() if v}, "per_panel": per_panel}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", type=Path, required=True)
    ap.add_argument("--v2", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--panel-class", type=int, default=0)
    ap.add_argument("--fallback-tile", type=int, default=640)
    ap.add_argument("--fallback-overlap", type=float, default=0.25)
    ap.add_argument("--condition-weights", type=Path, default=None)
    ap.add_argument("--condition-conf", type=float, default=0.25)
    ap.add_argument("--condition-iou", type=float, default=0.50)
    ap.add_argument("--crop-pad", type=float, default=0.04)
    args = ap.parse_args(argv)

    for w in (args.v1, args.v2):
        if not w.is_file():
            raise SystemExit(f"weights not found: {w}")
    images = _iter_images(args.images)
    if not images:
        raise SystemExit(f"no images under {args.images}")

    from ultralytics import YOLO
    from PIL import Image

    m1 = YOLO(str(args.v1))
    m2 = YOLO(str(args.v2))
    cond = YOLO(str(args.condition_weights)) if args.condition_weights and args.condition_weights.is_file() else None

    out = args.output_dir
    (out / "compare").mkdir(parents=True, exist_ok=True)
    (out / "per_image").mkdir(parents=True, exist_ok=True)
    if cond:
        (out / "crops").mkdir(parents=True, exist_ok=True)

    records = []
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        stem = img_path.stem

        v1p, v1o = _full_frame(m1, image, conf=args.conf, iou=args.iou, imgsz=args.imgsz, panel_class=args.panel_class)
        v2p, v2o = _full_frame(m2, image, conf=args.conf, iou=args.iou, imgsz=args.imgsz, panel_class=args.panel_class)

        # optional tiled FALLBACK for v2 (only if full-frame v2 returned 0-1 panels)
        fallback = None
        if len(v2p) <= 1:
            td = tiled_detect(m2, image, tile=args.fallback_tile, overlap=args.fallback_overlap,
                              conf=args.conf, nms_iou=args.iou, tile_nms_iou=args.iou,
                              imgsz=args.imgsz, class_filter=args.panel_class)
            fb_panels = [d for d in td["detections"] if d["class_id"] == args.panel_class]
            fb_panels.sort(key=lambda d: -d["confidence"])
            fallback = {"triggered": True, "n_tiles": td["tile_config"]["n_tiles"],
                        "panel_count": len(fb_panels), "panels": fb_panels}

        a = _draw(image, v1p, _V1_COLOR, f"v1: {len(v1p)} panel(s)")
        b = _draw(image, v2p, _V2_COLOR, f"v2: {len(v2p)} panel(s)")
        from PIL import ImageDraw
        orig = image.convert("RGB").copy()
        ImageDraw.Draw(orig).rectangle([0, 0, orig.width - 1, 16], fill=(0, 0, 0))
        ImageDraw.Draw(orig).text((3, 3), "original", fill=(255, 255, 255))
        combo = Image.new("RGB", (W * 3 + 16, H), (20, 20, 20))
        combo.paste(orig, (0, 0)); combo.paste(a, (W + 8, 0)); combo.paste(b, (W * 2 + 16, 0))
        combo.save(out / "compare" / f"{stem}__v1_v2.jpg")

        rec = {
            "image": img_path.name, "image_size_wh": [W, H], "image_sha256": _sha256(img_path),
            "v1": {"panel_count": len(v1p), "panel_confidences": [p["confidence"] for p in v1p],
                   "panels": v1p, "non_panel": v1o, "localization_notes": _localization_notes(v1p, W, H)},
            "v2": {"panel_count": len(v2p), "panel_confidences": [p["confidence"] for p in v2p],
                   "panels": v2p, "non_panel": v2o, "localization_notes": _localization_notes(v2p, W, H)},
            "v2_tiled_fallback": fallback,
            "qualitative_only": True,
        }
        if cond:
            rec["condition_chain_v1"] = _condition_chain(
                cond, image, v1p, conf=args.condition_conf, iou=args.condition_iou,
                imgsz=args.imgsz, crop_pad=args.crop_pad, out_dir=out / "crops", stem=stem, tag="v1")
            rec["condition_chain_v2"] = _condition_chain(
                cond, image, v2p, conf=args.condition_conf, iou=args.condition_iou,
                imgsz=args.imgsz, crop_pad=args.crop_pad, out_dir=out / "crops", stem=stem, tag="v2")
        records.append(rec)
        (out / "per_image" / f"{stem}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    summary = {
        "experiment": "Phase 11 - ground detector v1 vs v2 (full-frame), + tiled fallback",
        "v1_weights": str(args.v1), "v1_sha256": _sha256(args.v1),
        "v2_weights": str(args.v2), "v2_sha256": _sha256(args.v2),
        "condition_weights": str(args.condition_weights) if cond else None,
        "condition_sha256": _sha256(args.condition_weights) if cond else None,
        "config": {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
                   "fallback_tile": args.fallback_tile, "fallback_overlap": args.fallback_overlap,
                   "condition_conf": args.condition_conf},
        "num_images": len(images),
        "per_image": [
            {"image": r["image"], "dims": r["image_size_wh"],
             "v1_panels": r["v1"]["panel_count"], "v2_panels": r["v2"]["panel_count"],
             "v1_conf": r["v1"]["panel_confidences"], "v2_conf": r["v2"]["panel_confidences"],
             "v1_non_panel": r["v1"]["non_panel"], "v2_non_panel": r["v2"]["non_panel"],
             "v2_fallback_panels": (r["v2_tiled_fallback"]["panel_count"] if r["v2_tiled_fallback"] else None)}
            for r in records
        ],
        "records": records,
        "disclaimer": (
            "QUALITATIVE ONLY. No ground truth for the six owner photos. NO accuracy / "
            "precision / recall / F1 / mAP / confusion matrix. Predictions are raw model "
            "outputs, not facts. Absence of a condition prediction does NOT mean Clean. "
            "ELECTRICAL_DAMAGE is NOT validated electrical-fault detection. Hotspot excluded."
        ),
    }
    (out / "v2_real_world_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"images: {len(images)}")
    for r in records:
        fb = r["v2_tiled_fallback"]
        fbs = f"  (fallback -> {fb['panel_count']})" if fb else ""
        print(f"  {r['image']:<46} v1={r['v1']['panel_count']}  v2={r['v2']['panel_count']}{fbs}")
    print(f"summary: {out / 'v2_real_world_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
