#!/usr/bin/env python3
"""scripts/train_final_web_condition.py — fit the FINAL research model of the
web-only pipeline: frozen ConvNeXt-Tiny embeddings (whole-frame resize,
+ horizontal-flip average) -> standardiser -> balanced logistic regression
(C=0.05), trained on ALL pool images (the recipe that won the family-grouped CV
in scripts/eval_backbones_grouped_cv.py).

Writes a joblib bundle + JSON metadata. Also provides `predict_paths()` for
inference. RESEARCH ONLY: trained on web photos, never validated on the target
rooftop camera; do not use for alerts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from training.detection.ground_pv.condition_taxonomy import CONDITION_4CLASS_ORDER  # noqa: E402

SIZE, BACKBONE, C = 320, "convnext_tiny", 0.05


def predict_paths(bundle_path: Path, paths: list[Path]) -> list[dict]:
    """Class + probabilities for each image path (loads the backbone on demand)."""
    from scripts.eval_backbones_grouped_cv import embed  # noqa: WPS433
    b = joblib.load(bundle_path)
    X = embed(b["backbone"], b["size"], paths)
    p = b["clf"].predict_proba(b["scaler"].transform(X))
    return [{"path": str(pa), "pred": b["classes"][int(r.argmax())], "conf": float(r.max()),
             **{f"p_{c}": float(v) for c, v in zip(b["classes"], r)}} for pa, r in zip(paths, p)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("F:/Solar AI Datasets/condition_web_v2"))
    ap.add_argument("--report-dir", type=Path, default=_REPO / "datasets/ground_pv_research/web_condition_v2")
    ap.add_argument("--out", type=Path, default=_REPO / "weights/convnext_lr_condition_web_v2.joblib")
    args = ap.parse_args(argv)

    rows = list(csv.DictReader((args.report_dir / "manifest_representatives.csv").open(encoding="utf-8")))
    classes = list(CONDITION_4CLASS_ORDER)
    y = np.array([classes.index(r["label"]) for r in rows])
    X = np.load(args.dataset / f"emb_{BACKBONE}_{SIZE}.npy")
    assert len(X) == len(y), "embedding cache does not match the manifest"
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=3000, C=C, class_weight="balanced").fit(sc.transform(X), y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"backbone": BACKBONE, "size": SIZE, "scaler": sc, "clf": clf, "classes": classes}, args.out)
    meta = {"classes_index_order": classes, "recipe": f"{BACKBONE}@{SIZE} frozen embeddings (+hflip avg) -> StandardScaler -> "
            f"LogisticRegression(C={C}, class_weight=balanced)", "n_train": len(y),
            "train_class_counts": {c: int((y == i).sum()) for i, c in enumerate(classes)},
            "cv_estimate": "see backbone_grouped_cv.json (5-fold family-grouped OOF, macro-F1 0.790)",
            "status": "RESEARCH ONLY - web photos only, never validated on the target rooftop camera",
            "bundle_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
