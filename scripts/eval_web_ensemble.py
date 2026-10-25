#!/usr/bin/env python3
"""scripts/eval_web_ensemble.py — score single models and probability-averaged ensembles from the
out-of-fold (OOF) predictions of several Kaggle runs on the SAME family-grouped folds.

Every OOF row was predicted by a model that never saw that photo's family, so averaging OOF
probabilities across runs is a legitimate estimate of the ensemble's out-of-sample performance.

Honesty: the HEADLINE is the plain mean of ALL members (no selection). `best_subset` is picked
on the same OOF scores, so it is optimistic and labelled as such.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from scripts.eval_backbones_grouped_cv import summarize  # noqa: E402
from training.detection.ground_pv.condition_taxonomy import CONDITION_4CLASS_ORDER  # noqa: E402


def score(probs: np.ndarray, y: np.ndarray, lineage: np.ndarray, classes: list[str]) -> dict:
    return summarize(y, probs.argmax(1), lineage, classes)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="name=path/to/oof_probs.npy")
    ap.add_argument("--folds", type=Path, default=Path("F:/Solar AI Datasets/kaggle_staging/web_condition_v2_clean_dataset/folds.csv"))
    ap.add_argument("--out", type=Path, default=_REPO / "datasets/ground_pv_research/web_condition_v2/ensemble_eval.json")
    args = ap.parse_args(argv)

    rows = list(csv.DictReader(args.folds.open(encoding="utf-8")))
    classes = list(CONDITION_4CLASS_ORDER)
    y = np.array([classes.index(r["label"]) for r in rows])
    lineage = np.array([r["lineage"] for r in rows])
    P = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        p = np.load(path)
        assert p.shape == (len(y), len(classes)), f"{name}: OOF shape {p.shape} does not match the folds file"
        assert np.allclose(p.sum(1), 1, atol=1e-3), f"{name}: rows are not probabilities"
        P[name] = p

    res = {"n": len(y), "members": {}, "ensembles": {}}
    for n, p in P.items():
        res["members"][n] = score(p, y, lineage, classes)
    names = list(P)
    full = score(np.mean([P[n] for n in names], 0), y, lineage, classes)
    res["ensembles"]["ALL_mean (headline, no selection)"] = {"members": names, **full}
    best = None
    for k in range(2, len(names)):
        for sub in itertools.combinations(names, k):
            s = score(np.mean([P[n] for n in sub], 0), y, lineage, classes)
            if best is None or s["macro_f1"] > best[1]["macro_f1"]:
                best = (sub, s)
    if best:
        res["ensembles"]["best_subset (selected on OOF -> optimistic)"] = {"members": list(best[0]), **best[1]}

    print(f"{'model':46s} acc    balacc macroF1 | " + " ".join(f"{c[:5]:>5s}" for c in classes))
    def line(name, s):
        pc = s["per_class"]
        print(f"{name:46s} {s['accuracy']:.3f}  {s['balanced_accuracy']:.3f}  {s['macro_f1']:.3f}  | "
              + " ".join(f"{pc[c]['recall']:.3f}" for c in classes))
    for n, s in res["members"].items():
        line(n, s)
    for n, s in res["ensembles"].items():
        line(n + " " + str(len(s["members"])) + "m", s)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
