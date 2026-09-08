#!/usr/bin/env python3
"""scripts/analyze_ground_pv_dataset_scale.py — Phase 11 training-data failure
analysis for the ground PANEL detector.

Answers: does datasets/ground_pv_research/yolo/ contain enough examples that
resemble the Phase 10 real-world failures?
  * image 4  -> a single large foreground panel filling most of the frame
  * image 6  -> a wide multi-panel array
  * large / close-up panels in general
  * boundary-touching panels

Reads labels only (YOLO `class cx cy w h` normalised). Class 0 = panel.
Does NOT modify the dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PANEL_CLASS = 0
EDGE_EPS = 0.01  # within 1% of a border == touching


def _iter_split(root: Path, split: str):
    idir = root / split / "images"
    ldir = root / split / "labels"
    for img in sorted(idir.glob("*")):
        if img.suffix.lower() not in _IMG_EXT:
            continue
        lp = ldir / f"{img.stem}.txt"
        boxes = []
        if lp.is_file():
            for line in lp.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) != 5:
                    continue
                try:
                    cid = int(p[0]); cx, cy, w, h = (float(v) for v in p[1:])
                except ValueError:
                    continue
                boxes.append((cid, cx, cy, w, h))
        yield img.name, boxes


def _area_bucket(frac: float) -> str:
    if frac < 0.25:
        return "<25%"
    if frac < 0.50:
        return "25-50%"
    if frac < 0.75:
        return "50-75%"
    return ">75%"


def analyze(root: Path) -> dict:
    out: dict = {"dataset": str(root), "splits": {}, "totals": {}}
    all_panel_area = Counter()
    all_panels_per_img = Counter()
    all_big_foreground = 0        # image-4-like: >=1 panel >55% area AND <=2 panels
    all_wide_array = 0            # image-6-like: >=4 panels, none dominant (<40%)
    all_edge_panels = 0
    all_panel_boxes = 0
    all_images = 0
    all_images_with_panel = 0
    biggest_examples: list[tuple[float, str, str]] = []
    widearray_examples: list[tuple[int, str, str]] = []

    for split in ("train", "valid", "test"):
        if not (root / split / "images").is_dir():
            continue
        area_bucket = Counter()
        panels_per_img = Counter()
        edge_panels = 0
        panel_boxes = 0
        n_images = 0
        n_with_panel = 0
        big_fg = 0
        wide = 0
        cls_counter = Counter()
        for name, boxes in _iter_split(root, split):
            n_images += 1
            all_images += 1
            panels = [b for b in boxes if b[0] == PANEL_CLASS]
            for (cid, *_r) in boxes:
                cls_counter[cid] += 1
            npan = len(panels)
            panels_per_img[npan] += 1
            all_panels_per_img[npan] += 1
            if npan:
                n_with_panel += 1
                all_images_with_panel += 1
            areas = []
            for (_cid, cx, cy, w, h) in panels:
                frac = max(0.0, min(1.0, w)) * max(0.0, min(1.0, h))
                areas.append(frac)
                area_bucket[_area_bucket(frac)] += 1
                all_panel_area[_area_bucket(frac)] += 1
                panel_boxes += 1
                all_panel_boxes += 1
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2
                if x1 <= EDGE_EPS or y1 <= EDGE_EPS or x2 >= 1 - EDGE_EPS or y2 >= 1 - EDGE_EPS:
                    edge_panels += 1
                    all_edge_panels += 1
            if areas:
                mx = max(areas)
                if mx > 0.55 and npan <= 2:
                    big_fg += 1
                    all_big_foreground += 1
                    biggest_examples.append((round(mx, 3), split, name))
                if npan >= 4 and mx < 0.40:
                    wide += 1
                    all_wide_array += 1
                    widearray_examples.append((npan, split, name))
        out["splits"][split] = {
            "images": n_images,
            "images_with_panel": n_with_panel,
            "panel_boxes": panel_boxes,
            "class_instances": {("panel" if k == 0 else "cracked" if k == 1 else "dusty"): v
                                for k, v in sorted(cls_counter.items())},
            "panel_area_buckets": dict(area_bucket),
            "panels_per_image_hist": {str(k): v for k, v in sorted(panels_per_img.items())},
            "boundary_touching_panel_boxes": edge_panels,
            "image4_like_big_foreground_images": big_fg,
            "image6_like_wide_array_images": wide,
        }

    out["totals"] = {
        "images": all_images,
        "images_with_panel": all_images_with_panel,
        "panel_boxes": all_panel_boxes,
        "panel_area_buckets": dict(all_panel_area),
        "panel_area_buckets_pct": {k: round(100 * v / max(1, all_panel_boxes), 1)
                                   for k, v in all_panel_area.items()},
        "panels_per_image_hist": {str(k): v for k, v in sorted(all_panels_per_img.items())},
        "boundary_touching_panel_boxes": all_edge_panels,
        "boundary_touching_pct": round(100 * all_edge_panels / max(1, all_panel_boxes), 1),
        "image4_like_big_foreground_images": all_big_foreground,
        "image4_like_pct": round(100 * all_big_foreground / max(1, all_images), 2),
        "image6_like_wide_array_images": all_wide_array,
        "image6_like_pct": round(100 * all_wide_array / max(1, all_images), 2),
    }
    out["image4_like_examples_top"] = sorted(biggest_examples, reverse=True)[:15]
    out["image6_like_examples_top"] = sorted(widearray_examples, reverse=True)[:15]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path,
                    default=_REPO_ROOT / "datasets" / "ground_pv_research" / "yolo")
    ap.add_argument("--out-json", type=Path,
                    default=_REPO_ROOT / "datasets" / "ground_pv_research" / "reports" / "ground_dataset_scale_analysis.json")
    ap.add_argument("--out-csv", type=Path,
                    default=_REPO_ROOT / "datasets" / "ground_pv_research" / "reports" / "ground_dataset_scale_targets.csv")
    args = ap.parse_args(argv)

    res = analyze(args.dataset)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")

    # a per-image CSV of the image-4-like / image-6-like candidates (for
    # leakage-safe oversampling later — join on family via the traceability file)
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "split", "filename", "metric"])
        for (mx, split, name) in res["image4_like_examples_top"]:
            w.writerow(["image4_like_big_foreground", split, name, mx])
        for (npan, split, name) in res["image6_like_examples_top"]:
            w.writerow(["image6_like_wide_array", split, name, npan])

    t = res["totals"]
    print(json.dumps({
        "images": t["images"], "panel_boxes": t["panel_boxes"],
        "panel_area_buckets_pct": t["panel_area_buckets_pct"],
        "boundary_touching_pct": t["boundary_touching_pct"],
        "image4_like_images": t["image4_like_big_foreground_images"],
        "image6_like_images": t["image6_like_wide_array_images"],
        "panels_per_image_hist": t["panels_per_image_hist"],
    }, indent=2))
    print(f"\nwrote {args.out_json}\n      {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
