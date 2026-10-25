#!/usr/bin/env python3
"""scripts/infer_web_ensemble.py — run the FINAL web-only condition model: a probability-averaged
ensemble of Kaggle-fine-tuned backbones (see weights/web_condition_ensemble_v1/ensemble.json).

Each member: whole-frame Resize((S,S)) with its own S, ImageNet mean/std, average of the image and its
horizontal flip. Class order is canonical (CLEAN, DUSTY, BIRD_DROP, PHYSICAL_DAMAGE), read from the manifest.

RESEARCH ONLY: trained on web photos, never validated on the target rooftop camera. A prediction is a raw
model output, not a verified condition; do not use for alerts.
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


def build_model(arch: str, n_classes: int) -> nn.Module:
    """Same head layout the Kaggle notebooks trained (weights=None: the fine-tuned state dict is loaded after)."""
    if arch.startswith("convnext"):
        ctor = {"convnext_tiny": models.convnext_tiny, "convnext_small": models.convnext_small,
                "convnext_base": models.convnext_base}[arch]
        m = ctor(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)
    elif arch == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    else:
        raise KeyError(f"unknown architecture {arch!r}")
    return m


class WebEnsemble:
    def __init__(self, manifest: Path):
        self.dir = manifest.parent
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        self.classes: list[str] = meta["classes_index_order"]
        self.members = []
        for mem in meta["members"]:
            m = build_model(mem["arch"], len(self.classes))
            m.load_state_dict(torch.load(self.dir / mem["file"], map_location="cpu"))
            tf = transforms.Compose([transforms.Resize((mem["size"], mem["size"])), transforms.ToTensor(),
                                     transforms.Normalize(_MEAN, _STD)])
            self.members.append((m.eval(), tf))

    @torch.no_grad()
    def predict_proba(self, image: Image.Image) -> list[float]:
        im = ImageOps.exif_transpose(image).convert("RGB")
        acc = torch.zeros(len(self.classes))
        for m, tf in self.members:
            x = tf(im).unsqueeze(0)
            acc += (torch.softmax(m(x), 1) + torch.softmax(m(torch.flip(x, dims=[3])), 1))[0] / 2
        return (acc / len(self.members)).tolist()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=_REPO / "weights/web_condition_ensemble_v1/ensemble.json")
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args(argv)

    ens = WebEnsemble(args.manifest)
    files = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in _EXT)
    rows = []
    for p in files:
        with Image.open(p) as im:
            pr = ens.predict_proba(im)
        top = max(range(len(pr)), key=lambda i: pr[i])
        rows.append({"file": str(p), "pred": ens.classes[top], "conf": round(pr[top], 3),
                     **{f"p_{c}": round(v, 3) for c, v in zip(ens.classes, pr)}})
    for r in rows[:50]:
        print(f"{Path(r['file']).name[:48]:48s} {r['pred']:16s} {r['conf']:.2f}")
    if args.out_csv and rows:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} images; {len(ens.members)} members; classes {ens.classes}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
