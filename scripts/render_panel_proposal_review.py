#!/usr/bin/env python3
"""scripts/render_panel_proposal_review.py - render numbered review sheets for
SAM panel proposals (see scripts/propose_panels_sam.py).

For every `<stem>.sam.json` this:
  * filters the raw candidates (size / shape / colour) and de-duplicates them,
  * writes `<stem>.filtered.json` with a stable numbering, and
  * writes `<stem>.review.jpg`: the image with each surviving candidate outlined
    and numbered, plus a labelled 10 % coordinate grid so boxes SAM missed or
    merged can be entered by hand.

Nothing here is ground truth. A reviewer records decisions (keep ids / add boxes)
in a decisions JSON consumed by scripts/build_panel_labels_from_review.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

MIN_AREA, MAX_AREA = 0.004, 0.45
MIN_SOLIDITY = 0.80
HUE_RANGE = (90.0, 135.0)   # OpenCV hue (0-180); panel blue
MIN_SAT = 25.0
DEDUP_IOU = 0.80


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def filter_candidates(cands: list[dict], min_sat: float = MIN_SAT) -> list[dict]:
    kept = []
    for c in cands:
        h, s, _ = c["mean_hsv"]
        if not MIN_AREA <= c["area_frac"] <= MAX_AREA:
            continue
        if c["solidity"] < MIN_SOLIDITY or s < min_sat or not HUE_RANGE[0] <= h <= HUE_RANGE[1]:
            continue
        kept.append(c)
    kept.sort(key=lambda c: (-c["solidity"], -c["area_frac"]))
    out: list[dict] = []
    for c in kept:
        if all(_iou(c["bbox_norm"], o["bbox_norm"]) < DEDUP_IOU for o in out):
            out.append(c)
    out.sort(key=lambda c: -c["area_frac"])
    for i, c in enumerate(out):
        c["id"] = i
    return out


def _draw(img: np.ndarray, cands: list[dict]) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for k in range(1, 10):
        x, y = int(w * k / 10), int(h * k / 10)
        cv2.line(out, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.line(out, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(out, f"{k / 10:.1f}", (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(out, f"{k / 10:.1f}", (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    rng = np.random.RandomState(7)
    for c in cands:
        col = tuple(int(v) for v in rng.randint(80, 255, 3))
        pts = (np.array(c["polygon_norm"]) * [w, h]).astype(np.int32)
        cv2.polylines(out, [pts], True, col, 2)
        x1, y1, x2, y2 = c["bbox_norm"]
        cx, cy = int((x1 + x2) / 2 * w), int((y1 + y2) / 2 * h)
        cv2.putText(out, str(c["id"]), (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5)
        cv2.putText(out, str(c["id"]), (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 2)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--proposals-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-side", type=int, default=1500)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these image stems")
    ap.add_argument("--min-sat", type=float, default=MIN_SAT,
                    help="minimum mean HSV saturation; lower it for pale, sky-reflecting (glare) panels")
    args = ap.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_stem = {p.stem: p for p in args.images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    for f in sorted(args.proposals_dir.glob("*.sam.json")):
        stem = f.name[: -len(".sam.json")]
        if args.only and stem not in args.only:
            continue
        cands = filter_candidates(json.loads(f.read_text(encoding="utf-8"))["candidates"], args.min_sat)
        (args.output_dir / f"{stem}.filtered.json").write_text(
            json.dumps({"image": by_stem[stem].name, "note": "PROPOSAL - NOT GROUND TRUTH",
                        "candidates": cands}), encoding="utf-8")
        img = cv2.imdecode(np.fromfile(str(by_stem[stem]), dtype=np.uint8), cv2.IMREAD_COLOR)
        s = args.max_side / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
        cv2.imencode(".jpg", _draw(img, cands), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(
            str(args.output_dir / f"{stem}.review.jpg"))
        print(f"{stem}: {len(cands)} candidates after filtering")
    return 0


if __name__ == "__main__":
    sys.exit(main())
