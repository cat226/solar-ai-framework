#!/usr/bin/env python3
"""scripts/eval_backbones_grouped_cv.py — compare pretrained backbones on the
web-only 4-class condition pool with FAMILY-GROUPED stratified K-fold
cross-validation.

Why: the pool is small and the rare classes (BIRD_DROP / PHYSICAL_DAMAGE) have
only ~130 / ~50 families, so a single 15% test split gives n=19 / n=7. Every
image is predicted exactly once out-of-fold (OOF), giving the full class counts
for the recall intervals. Groups = content family, so a photo and its near-
twins never straddle train/test inside any fold.

Method: frozen ImageNet embeddings (whole-frame, + horizontal-flip average) ->
standardised -> balanced logistic regression. Cheap, deterministic, and a fair
head-to-head of backbones. Nothing here touches production weights.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision import models, transforms

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from training.detection.ground_pv.condition_taxonomy import CONDITION_4CLASS_ORDER  # noqa: E402

_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _backbone(name: str) -> nn.Module:
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1); m.classifier = nn.Identity()
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1); m.classifier = nn.Identity()
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); m.fc = nn.Identity()
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1); m.classifier[2] = nn.Identity()
    else:
        raise KeyError(name)
    return m.eval()


def embed(name: str, size: int, paths: list[Path], batch: int = 8) -> np.ndarray:
    model = _backbone(name)
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])
            f = (model(x) + model(torch.flip(x, dims=[3]))) / 2
            out.append(f.numpy())
            if (i // batch) % 80 == 0:
                print(f"    {name}@{size}: {i}/{len(paths)}", flush=True)
    return np.concatenate(out)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def summarize(y: np.ndarray, pred: np.ndarray, lineage: np.ndarray, classes: list[str]) -> dict:
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    per, f1s, recs = {}, [], []
    for i, c in enumerate(classes):
        tp, nt, npred = cm[i, i], cm[i].sum(), cm[:, i].sum()
        prec = tp / npred if npred else None
        rec = tp / nt if nt else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
        per[c] = {"n": int(nt), "recall": round(rec, 3), "recall_wilson95": wilson(int(tp), int(nt)),
                  "precision": None if prec is None else round(prec, 3), "f1": round(f1, 3)}
        f1s.append(f1); recs.append(rec)
    by_lin = {}
    for lin in sorted(set(lineage)):
        m = lineage == lin
        by_lin[lin] = {"n": int(m.sum()), "accuracy": round(float((y[m] == pred[m]).mean()), 3)}
    return {"accuracy": round(float((y == pred).mean()), 3), "balanced_accuracy": round(float(np.mean(recs)), 3),
            "macro_f1": round(float(np.mean(f1s)), 3), "per_class": per, "confusion_matrix": cm.tolist(),
            "by_lineage": by_lin}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("F:/Solar AI Datasets/condition_web_v2"))
    ap.add_argument("--report-dir", type=Path, default=_REPO / "datasets/ground_pv_research/web_condition_v2")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", nargs="+", default=["mobilenet_v2:224", "efficientnet_b0:224", "resnet50:224", "convnext_tiny:224", "convnext_tiny:320"])
    args = ap.parse_args(argv)

    torch.set_num_threads(8)
    rows = list(csv.DictReader((args.report_dir / "manifest_representatives.csv").open(encoding="utf-8")))
    classes = list(CONDITION_4CLASS_ORDER)
    paths = [args.dataset / r["out_rel"] for r in rows]
    y = np.array([classes.index(r["label"]) for r in rows])
    groups = np.array([int(r["family"]) for r in rows])
    lineage = np.array([r["lineage"] for r in rows])
    print(f"{len(rows)} images; classes {dict(Counter(r['label'] for r in rows))}", flush=True)

    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = list(skf.split(np.zeros(len(y)), y, groups))
    # every group must be wholly inside one fold's test set
    for tr, te in folds:
        assert not (set(groups[tr]) & set(groups[te])), "family leaked across a fold"

    results = {}
    for cfg in args.configs:
        name, size = cfg.split(":"); size = int(size)
        cache = args.dataset / f"emb_{name}_{size}.npy"
        t0 = time.time()
        if cache.exists():
            X = np.load(cache)
        else:
            print(f"embedding {cfg} ...", flush=True)
            X = embed(name, size, paths)
            np.save(cache, X)
        t_emb = time.time() - t0
        oof_p = np.zeros((len(y), len(classes)))
        for tr, te in folds:
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=3000, C=0.05, class_weight="balanced")
            clf.fit(sc.transform(X[tr]), y[tr])
            oof_p[te] = clf.predict_proba(sc.transform(X[te]))
        pred = oof_p.argmax(1)
        s = summarize(y, pred, lineage, classes)
        s["embed_seconds"] = round(t_emb, 1); s["feature_dim"] = int(X.shape[1])
        results[cfg] = s
        np.save(args.dataset / f"oof_{name}_{size}.npy", oof_p)
        pc = s["per_class"]
        print(f"{cfg:22s} acc={s['accuracy']:.3f} balacc={s['balanced_accuracy']:.3f} macroF1={s['macro_f1']:.3f} | "
              + " ".join(f"{c[:4]}={pc[c]['recall']:.2f}" for c in classes), flush=True)

    best = max(results, key=lambda c: results[c]["macro_f1"])
    out = {"protocol": f"{args.folds}-fold StratifiedGroupKFold (groups=content family), frozen embeddings + "
                       "balanced LogisticRegression(C=0.05) on standardised features; OOF predictions for every image",
           "n_images": len(rows), "class_counts": dict(Counter(r["label"] for r in rows)), "seed": args.seed,
           "configs": results, "best_by_macro_f1": best,
           "caveats": ["web-only pool; no rooftop images exist in this pipeline - this is NOT a rooftop-camera accuracy",
                       "C was fixed a priori (0.05), not tuned on OOF; backbone choice IS selected on OOF so the best "
                       "number is mildly optimistic",
                       "web sources share lineage (stock/ads photos): per-lineage accuracy is reported"]}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "backbone_grouped_cv.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("best:", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
