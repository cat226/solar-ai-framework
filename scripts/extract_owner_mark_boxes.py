#!/usr/bin/env python3
"""scripts/extract_owner_mark_boxes.py — recover the owner's burned-in defect
boxes (black = BIRD_DROP, red = PHYSICAL_DAMAGE) by diffing each MARKED photo
against its PRISTINE original (same file name), and write them as JSON.

Read-only on both image folders. The output is the owner's own ground truth in
machine-readable form (it replaces the derived crops that were deleted from F:).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from training.detection.ground_pv.owner_mark_boxes import extract_boxes  # noqa: E402

_INK_TO_LABEL = {"black": "BIRD_DROP", "red": "PHYSICAL_DAMAGE"}  # owner legend, confirmed 2026-10-21
_EXT = {".jpg", ".jpeg", ".png"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pristine", type=Path, required=True)
    ap.add_argument("--marked", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=_REPO / "datasets/ground_pv_research/unified_condition_v1/owner_mark_boxes.json")
    args = ap.parse_args(argv)

    result, differing = {}, 0
    for p in sorted(args.pristine.iterdir()):
        m = args.marked / p.name
        if p.suffix.lower() not in _EXT or not m.exists():
            continue
        with Image.open(p) as a, Image.open(m) as b:
            a = ImageOps.exif_transpose(a).convert("RGB")
            b = ImageOps.exif_transpose(b).convert("RGB")
            if a.size != b.size:
                continue  # not the same photo - never guess a mapping
            if a.tobytes() == b.tobytes():
                continue
            differing += 1
            boxes = extract_boxes(a, b)
        if boxes:
            result[p.name] = {"image_size": [a.width, a.height], "boxes": [
                {**bx, "label": _INK_TO_LABEL[bx["ink"]]} for bx in boxes]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "1.0", "source": "owner burned-in marks, recovered by pristine-vs-marked diff",
               "legend": _INK_TO_LABEL, "n_photos_with_marks": len(result),
               "n_boxes": sum(len(v["boxes"]) for v in result.values()), "photos": result}
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"{differing} differing pairs -> {payload['n_photos_with_marks']} photos, {payload['n_boxes']} boxes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
