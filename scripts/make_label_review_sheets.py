#!/usr/bin/env python3
"""scripts/make_label_review_sheets.py — render the AI-proposed rooftop labels
as easy-to-scan contact sheets so the owner can confirm or correct them.

Each tile shows the PRISTINE photo, its index, proposed label and confidence
(owner-marked rows in yellow with the owner's recovered box drawn on top;
AI rows in white; low-confidence / UNSURE in orange). Reply with the indices
that are wrong and the right label, or edit the CSV's `label` column.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

_REPO = Path(__file__).resolve().parent.parent
_DIR = _REPO / "datasets/ground_pv_research/unified_condition_v1"
_COL = {"owner": (255, 230, 60), "high": (120, 230, 120), "medium": (255, 255, 255), "low": (255, 160, 60)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, default=Path("F:/Solar AI Consolidated Datasets/1_ground_level/representative_panel_v1/raw/batch_002_rooftop_2026_09_18"))
    ap.add_argument("--labels", type=Path, default=_DIR / "rooftop48_ai_proposed_labels.csv")
    ap.add_argument("--boxes", type=Path, default=_DIR / "owner_mark_boxes.json")
    ap.add_argument("--out-dir", type=Path, default=Path("F:/Solar AI Datasets/condition_unified_v1/review_sheets"))
    ap.add_argument("--per-sheet", type=int, default=6)
    args = ap.parse_args(argv)

    rows = list(csv.DictReader(args.labels.open(encoding="utf-8")))
    boxes = json.loads(args.boxes.read_text(encoding="utf-8"))["photos"] if args.boxes.exists() else {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    T, cols = 620, 3
    for s in range(0, len(rows), args.per_sheet):
        chunk = rows[s:s + args.per_sheet]
        rws = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * T, rws * (T + 46)), (24, 24, 24))
        d = ImageDraw.Draw(sheet)
        for i, r in enumerate(chunk):
            with Image.open(args.images / r["file"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                sc = (T - 6) / max(im.size)
                small = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.LANCZOS)
                dr = ImageDraw.Draw(small)
                for b in boxes.get(r["file"], {}).get("boxes", []):
                    dr.rectangle((b["x1"] * sc, b["y1"] * sc, b["x2"] * sc, b["y2"] * sc), outline=(255, 230, 60), width=2)
            x, y = (i % cols) * T, (i // cols) * (T + 46)
            sheet.paste(small, (x + 3, y + 3))
            col = _COL.get(r["confidence"], (255, 255, 255))
            d.text((x + 6, y + T - 2), f"#{r['index']}  {r['label']}  [{r['confidence']}]", fill=col)
            d.text((x + 6, y + T + 12), r["evidence"][:92], fill=(170, 170, 170))
        sheet.save(args.out_dir / f"review_{s // args.per_sheet + 1:02d}.jpg", quality=88)
    print(f"{len(rows)} photos -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
