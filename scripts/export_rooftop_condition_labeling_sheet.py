#!/usr/bin/env python3
"""scripts/export_rooftop_condition_labeling_sheet.py - export per-panel crops + a CSV
for the OWNER to label (CLEAN / DUSTY / BIRD_DROP / PHYSICAL_DAMAGE).

Why this exists: panel LOCALISATION labels can be derived from geometry, but panel
CONDITION cannot be honestly read off a photo (dust is often invisible, bird drops and
cracks can be ambiguous). Only someone who knows the physical panels can label them.
The model's own guess is deliberately NOT included, so labels are not anchored on it.

Output (default datasets/ground_pv_research/rooftop_condition_v1/):
  crops/<split>/<crop_id>.jpg       padded crops of every labelled panel >= --min-area
  labeling_sheet.csv                crop_id,split,scene_id,source_image,panel_index,label,notes
Allowed `label` values: CLEAN, DUSTY, BIRD_DROP, PHYSICAL_DAMAGE, or blank / SKIP to omit.
Splits are inherited from the scene-disjoint panel dataset, so a scene never crosses splits.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
PAD = 0.04


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel-dataset", type=Path, default=_ROOT / "datasets/ground_pv_research/rooftop_panel_v1")
    ap.add_argument("--out", type=Path, default=_ROOT / "datasets/ground_pv_research/rooftop_condition_v1")
    ap.add_argument("--min-area", type=float, default=0.03, help="min box area as a fraction of the image")
    args = ap.parse_args(argv)

    rows = []
    with (args.panel_dataset / "split_manifest.csv").open(encoding="utf-8") as fh:
        meta = list(csv.DictReader(fh))
    for m in meta:
        stem, split = m["file"][:-4], m["split"]
        img = Image.open(args.panel_dataset / "images" / split / m["file"]).convert("RGB")
        W, H = img.size
        for i, line in enumerate(l for l in (args.panel_dataset / "labels" / split / f"{stem}.txt").read_text().splitlines() if l.strip()):
            _, x, y, w, h = map(float, line.split())
            if w * h < args.min_area:
                continue
            box = (max(0, int((x - w / 2 * (1 + PAD)) * W)), max(0, int((y - h / 2 * (1 + PAD)) * H)),
                   min(W, int((x + w / 2 * (1 + PAD)) * W)), min(H, int((y + h / 2 * (1 + PAD)) * H)))
            crop_id = f"{stem}_p{i:02d}"
            (args.out / "crops" / split).mkdir(parents=True, exist_ok=True)
            img.crop(box).save(args.out / "crops" / split / f"{crop_id}.jpg", quality=92)
            rows.append({"crop_id": crop_id, "split": split, "scene_id": m["scene_id"], "source_image": m["original"],
                         "panel_index": i, "label": "", "notes": ""})
    with (args.out / "labeling_sheet.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"exported {len(rows)} crops -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
