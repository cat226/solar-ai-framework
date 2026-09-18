#!/usr/bin/env python3
"""Evaluate a panel detector on a rooftop_panel_v1 split (panel class only, IoU 0.5 / 0.5-0.95).

Works for both the legacy 3-class ground checkpoints (class 0 = panel is the only class
scored) and the new 1-class detector. Also reports per-image predicted-vs-labelled panel
counts at a deployment confidence, since 'does it find the 6 panels' is the user-facing
question.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent


def _labels(p: Path) -> np.ndarray:
    rows = [[float(v) for v in l.split()[1:5]] for l in p.read_text().splitlines() if l.strip()]
    a = np.array(rows).reshape(-1, 4)
    return np.c_[a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2]


def _iou(a, b):
    ix = np.maximum(0, np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]))
    iy = np.maximum(0, np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=_ROOT / "datasets/ground_pv_research/rooftop_panel_v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25, help="deployment confidence for count/PR")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    from ultralytics.utils.metrics import ap_per_class

    model = YOLO(str(args.weights))
    imgs = sorted((args.dataset / "images" / args.split).glob("*.jpg"))
    tp_rows, confs, n_gt = [], [], 0
    per_image, tp_dep, fp_dep = [], 0, 0
    ious = np.linspace(0.5, 0.95, 10)
    for im in imgs:
        gt = _labels(args.dataset / "labels" / args.split / f"{im.stem}.txt")
        r = model.predict(str(im), conf=0.001, iou=0.6, imgsz=args.imgsz, verbose=False)[0]
        b = r.boxes
        cls = b.cls.cpu().numpy().astype(int)
        m = cls == 0
        xy = b.xyxyn.cpu().numpy()[m]
        cf = b.conf.cpu().numpy()[m]
        order = np.argsort(-cf)
        xy, cf = xy[order], cf[order]
        tp = np.zeros((len(cf), len(ious)), bool)
        for ti, thr in enumerate(ious):
            used = set()
            for pi in range(len(cf)):
                if len(gt) == 0:
                    break
                iou = _iou(xy[pi], gt)
                for j in np.argsort(-iou):
                    if iou[j] < thr:
                        break
                    if j not in used:
                        used.add(j); tp[pi, ti] = True
                        break
        tp_rows.append(tp); confs.append(cf); n_gt += len(gt)
        dep = cf >= args.conf
        tp_dep += int(tp[dep, 0].sum()); fp_dep += int((~tp[dep, 0]).sum())
        per_image.append({"image": im.name, "labelled": int(len(gt)), "detected@conf": int(dep.sum()),
                          "matched@0.5": int(tp[dep, 0].sum())})
    tp_all = np.concatenate(tp_rows) if tp_rows else np.zeros((0, 10), bool)
    cf_all = np.concatenate(confs) if confs else np.zeros(0)
    res = ap_per_class(tp_all, cf_all, np.zeros(len(cf_all)), np.zeros(n_gt), plot=False)
    ap_arr = res[5]  # (nc, 10)
    out = {"weights": str(args.weights), "split": args.split, "images": len(imgs), "labelled_panels": n_gt,
           "mAP50": float(ap_arr[0, 0]), "mAP50_95": float(ap_arr[0].mean()),
           "deployment_conf": args.conf, "precision@conf": tp_dep / max(tp_dep + fp_dep, 1),
           "recall@conf": tp_dep / max(n_gt, 1), "per_image": per_image}
    print(json.dumps({k: v for k, v in out.items() if k != "per_image"}, indent=1))
    for p in per_image:
        print(f"  {p['image']}: labelled={p['labelled']} detected={p['detected@conf']} matched={p['matched@0.5']}")
    if args.out:
        args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
