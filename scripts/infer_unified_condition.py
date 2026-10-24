#!/usr/bin/env python3
"""scripts/infer_unified_condition.py — run the unified image-level condition
classifier (scripts/train_unified_condition_classifier.py) over a folder of
photos and write a per-image CSV + a labelled contact sheet.

Intended use: a QUALITATIVE sanity check on the owner's pristine rooftop
photos. Those photos have no verified labels (the owner's marks cover only 6
of 48 images / 4 scenes), so nothing here is an accuracy figure. Point it at
the PRISTINE originals, never at the copies with burned-in owner boxes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageOps
from torchvision import models, transforms

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
_EXT = {".jpg", ".jpeg", ".png"}


def load_model(ckpt: Path):
    meta = json.loads(ckpt.with_suffix(".json").read_text(encoding="utf-8"))
    classes = meta["classes_index_order"]
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(classes))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.eval(), classes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=_REPO / "weights/mobilenet_condition_unified_v1.pth")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--contact-sheet", type=Path, default=None)
    ap.add_argument("--owner-marked", nargs="*", default=[],
                    help="file stems the owner marked as defective (anecdotal reference only)")
    args = ap.parse_args(argv)

    model, classes = load_model(args.checkpoint)
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in _EXT)
    rows = []
    with torch.no_grad():
        for p in files:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                pr = torch.softmax(model(tf(im).unsqueeze(0)), 1)[0].tolist()
            top = max(range(len(classes)), key=lambda i: pr[i])
            rows.append({"file": p.name, "pred": classes[top], "conf": round(pr[top], 3),
                         **{f"p_{c}": round(v, 3) for c, v in zip(classes, pr)},
                         "owner_marked": int(p.stem in set(args.owner_marked))})
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    dist = {c: sum(1 for r in rows if r["pred"] == c) for c in classes}
    marked = [r for r in rows if r["owner_marked"]]
    print("predictions:", dist)
    if marked:
        flagged = sum(1 for r in marked if r["pred"] != "CLEAN")
        print(f"owner-marked photos flagged non-CLEAN: {flagged}/{len(marked)} (anecdotal; {len(marked)} photos, few scenes)")
    if args.contact_sheet:
        T, cols = 260, 6
        sheet = Image.new("RGB", (cols * T, ((len(rows) + cols - 1) // cols) * (T + 18)), (25, 25, 25))
        d = ImageDraw.Draw(sheet)
        for i, (p, r) in enumerate(zip(files, rows)):
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((T - 4, T - 4))
                x, y = (i % cols) * T, (i // cols) * (T + 18)
                sheet.paste(im, (x + 2, y + 2))
            d.text((x + 4, y + T), f"{r['pred']} {r['conf']:.2f}{' *' if r['owner_marked'] else ''}", fill=(255, 220, 90))
        args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.contact_sheet, quality=85)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
