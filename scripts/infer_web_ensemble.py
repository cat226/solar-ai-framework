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

from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from training.detection.ground_pv.web_ensemble import WebEnsemble, build_model  # noqa: E402,F401
_EXT = {".jpg", ".jpeg", ".png"}


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
