#!/usr/bin/env python3
"""scripts/build_web_condition_kaggle_package.py — stage the web-only condition
pool (web_condition_v2) as a PRIVATE Kaggle dataset.

Contents: the 2,618 pool images, `folds.csv` (family-grouped 5-fold assignment,
computed HERE with StratifiedGroupKFold(seed=42) - identical to the local CV in
scripts/eval_backbones_grouped_cv.py - so the notebook never recomputes folds),
`manifest_representatives.csv`, `README.md`, `dataset-metadata.json`.

Local only: this script never talks to Kaggle. Upload is a separate, explicit
`kaggle datasets create` step. No first-party / rooftop image is included.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from training.detection.ground_pv.condition_taxonomy import CONDITION_4CLASS_ORDER  # noqa: E402

DATASET_ID = "edithstark/solar-ai-web-condition-v2"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("F:/Solar AI Datasets/condition_web_v2"))
    ap.add_argument("--report-dir", type=Path, default=_REPO / "datasets/ground_pv_research/web_condition_v2")
    ap.add_argument("--out", type=Path, default=Path("F:/Solar AI Datasets/kaggle_staging/web_condition_v2_dataset"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    rows = list(csv.DictReader((args.report_dir / "manifest_representatives.csv").open(encoding="utf-8")))
    classes = list(CONDITION_4CLASS_ORDER)
    y = np.array([classes.index(r["label"]) for r in rows])
    groups = np.array([int(r["family"]) for r in rows])
    fold = np.full(len(rows), -1)
    for k, (_, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed).split(np.zeros(len(y)), y, groups)):
        fold[te] = k
    assert (fold >= 0).all()
    # a family must be wholly inside one fold
    for g in set(groups.tolist()):
        assert len(set(fold[groups == g].tolist())) == 1

    if args.out.exists():
        shutil.rmtree(args.out)
    for r in rows:
        dst = args.out / r["out_rel"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.dataset / r["out_rel"], dst)
    with (args.out / "folds.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "family", "label", "lineage", "fold", "out_rel"])
        for r, f in zip(rows, fold):
            w.writerow([r["image_id"], r["family"], r["label"], r["lineage"], int(f), r["out_rel"]])
    shutil.copy2(args.report_dir / "manifest_representatives.csv", args.out / "manifest_representatives.csv")
    (args.out / "README.md").write_text(
        "# Solar AI web-only condition pool v2 (PRIVATE, research only)\n\n"
        "2,618 deduplicated web photos (CLEAN 1,559 / DUSTY 883 / BIRD_DROP 128 / PHYSICAL_DAMAGE 48) from three "
        "public sources (Faulty_solar_panel, Roboflow ahmed-tamer/solar-panels-dust-detection v2, GitHub "
        "suhasmaddali/Solar-Panels-Dust-Detection). Licences of ALL sources are unresolved: do not redistribute. "
        "`folds.csv` = family-grouped 5-fold CV assignment. No first-party/rooftop images.\n", encoding="utf-8")
    (args.out / "dataset-metadata.json").write_text(json.dumps({
        "title": "Solar AI web-only condition pool v2",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
        "subtitle": "Deduplicated web PV panel photos, 4 conditions, grouped CV folds",
        "description": "PRIVATE research dataset. 2,618 deduplicated web photos of PV panels labelled CLEAN/DUSTY/BIRD_DROP/"
                       "PHYSICAL_DAMAGE, assembled from three public sources with unresolved licences. Includes folds.csv "
                       "(family-grouped 5-fold). Not redistributable. No first-party images."}, indent=2), encoding="utf-8")
    files = [p for p in args.out.rglob("*") if p.is_file()]
    print(f"{len(rows)} images; folds {np.bincount(fold).tolist()}; {len(files)} files, "
          f"{sum(p.stat().st_size for p in files) / 1e6:.0f} MB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
