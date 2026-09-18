#!/usr/bin/env python3
"""scripts/propose_panels_sam.py - SAM-assisted DRAFT panel proposals.

Annotation AID only. Runs MobileSAM's automatic mask generator over each image
and writes candidate regions (polygon, bbox, shape stats) to a JSON file per
image. Nothing here is ground truth: a proposal only becomes a label after a
person (or a reviewer explicitly recorded as such) accepts it - see
scripts/build_panel_labels_from_review.py.

The SAM weights (mobile_sam.pt, Apache-2.0) are used for annotation
assistance only and are never part of any shipped model.

    python scripts/propose_panels_sam.py --images-dir <dir> --output-dir <dir> \
        --sam-weights mobile_sam.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_IMAGE_EXT = {".jpg", ".jpeg", ".png"}
_WORK_WIDTH = 1600  # SAM runs at 1024 internally; this only bounds mask memory


def _grid_predictor(points_per_side: int):
    """SAM Predictor whose automatic-mask grid density is configurable (ultralytics
    does not accept it as a predict() override)."""
    from ultralytics.models.sam import Predictor

    class _GridPredictor(Predictor):
        def generate(self, im, *args, **kwargs):
            kwargs.setdefault("points_stride", points_per_side)
            return super().generate(im, *args, **kwargs)

    return _GridPredictor


def _propose(model, image_bgr: np.ndarray, points_per_side: int) -> list[dict]:
    h, w = image_bgr.shape[:2]
    scale = _WORK_WIDTH / max(h, w) if max(h, w) > _WORK_WIDTH else 1.0
    small = cv2.resize(image_bgr, (int(w * scale), int(h * scale))) if scale != 1.0 else image_bgr
    sh, sw = small.shape[:2]
    result = model.predict(small, verbose=False, predictor=_grid_predictor(points_per_side))[0]
    if result.masks is None:
        return []
    masks = result.masks.data.cpu().numpy().astype(np.uint8)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    out: list[dict] = []
    for m in masks:
        if m.shape != (sh, sw):
            m = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST)
        area = float(m.sum()) / (sh * sw)
        if not 0.003 <= area <= 0.45:
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        solidity = cv2.contourArea(c) / max(cv2.contourArea(hull), 1.0)
        x, y, bw, bh = cv2.boundingRect(c)
        poly = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True).reshape(-1, 2)
        mean_h, mean_s, mean_v = (float(v) for v in cv2.mean(hsv, mask=m)[:3])
        out.append({
            "bbox_norm": [x / sw, y / sh, (x + bw) / sw, (y + bh) / sh],
            "area_frac": round(area, 5),
            "solidity": round(float(solidity), 4),
            "mean_hsv": [round(mean_h, 1), round(mean_s, 1), round(mean_v, 1)],
            "polygon_norm": [[round(px / sw, 5), round(py / sh, 5)] for px, py in poly.tolist()],
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--sam-weights", type=Path, required=True)
    ap.add_argument("--points-per-side", type=int, default=16)
    args = ap.parse_args(argv)

    from ultralytics import SAM

    model = SAM(str(args.sam_weights))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.images_dir.rglob("*") if p.suffix.lower() in _IMAGE_EXT)
    for i, p in enumerate(files, 1):
        dest = args.output_dir / f"{p.stem}.sam.json"
        if dest.is_file():
            continue
        t0 = time.time()
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        cands = _propose(model, img, args.points_per_side)
        dest.write_text(json.dumps({"image": p.name, "note": "PROPOSAL - NOT GROUND TRUTH",
                                    "candidates": cands}), encoding="utf-8")
        print(f"[{i}/{len(files)}] {p.name}: {len(cands)} candidates in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
