#!/usr/bin/env python3
"""scripts/infer_web_condition.py — run the Kaggle-fine-tuned ConvNeXt-Tiny web-only
condition classifier (web_condition_v2) on a folder of photos.

Preprocessing matches training: whole-frame Resize((288,288)), ImageNet mean/std,
probabilities averaged over the image and its horizontal flip. Class index order is
CANONICAL (CLEAN, DUSTY, BIRD_DROP, PHYSICAL_DAMAGE), read from the manifest next
to the checkpoint - never assumed.

RESEARCH ONLY: trained on web photos, never validated on the target rooftop camera.
A prediction is a raw model output, not a verified condition; do not use for alerts.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

_REPO = Path(__file__).resolve().parent.parent
_EXT = {".jpg", ".jpeg", ".png"}
_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
_SIZE = 288


def load(ckpt: Path):
    meta = json.loads(ckpt.with_suffix(".json").read_text(encoding="utf-8"))
    classes = meta["classes_index_order"]
    m = models.convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(768, len(classes))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.eval(), classes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=_REPO / "weights/convnext_tiny_condition_web_v2.pth")
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args(argv)

    model, classes = load(args.checkpoint)
    tf = transforms.Compose([transforms.Resize((_SIZE, _SIZE)), transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
    files = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in _EXT)
    rows = []
    with torch.no_grad():
        for p in files:
            with Image.open(p) as im:
                x = tf(ImageOps.exif_transpose(im).convert("RGB")).unsqueeze(0)
            pr = ((torch.softmax(model(x), 1) + torch.softmax(model(torch.flip(x, dims=[3])), 1)) / 2)[0].tolist()
            top = max(range(len(classes)), key=lambda i: pr[i])
            rows.append({"file": str(p), "pred": classes[top], "conf": round(pr[top], 3),
                         **{f"p_{c}": round(v, 3) for c, v in zip(classes, pr)}})
    for r in rows[:50]:
        print(f"{Path(r['file']).name[:48]:48s} {r['pred']:16s} {r['conf']:.2f}")
    if args.out_csv and rows:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} images; classes {classes}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
