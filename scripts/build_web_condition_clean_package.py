#!/usr/bin/env python3
"""scripts/build_web_condition_clean_package.py — apply datasets/ground_pv_research/web_condition_v2/
label_overrides.csv (drops + relabels, original label kept in the file) to the web-only pool and stage a
NEW private Kaggle dataset `edithstark/solar-ai-web-condition-v2-clean`.

Folds are inherited unchanged from the v2 package (a family stays in its fold; dropped images just vanish),
so out-of-fold numbers stay comparable with the uncleaned run. Local only - never talks to Kaggle.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from training.detection.ground_pv.condition_taxonomy import CONDITION_4CLASS_ORDER  # noqa: E402

DATASET_ID = "edithstark/solar-ai-web-condition-v2-clean"


def apply_overrides(rows: list[dict], overrides: list[dict]) -> tuple[list[dict], dict]:
    """Return (new_rows, summary). Fails closed on unknown ids, stale original labels, bad actions/classes."""
    by_id = {r["image_id"]: r for r in rows}
    drop, relabel = set(), {}
    for o in overrides:
        iid = o["image_id"]
        if iid not in by_id:
            raise KeyError(f"override for unknown image_id {iid}")
        if by_id[iid]["label"] != o["original_label"]:
            raise ValueError(f"{iid}: file says original {o['original_label']} but pool has {by_id[iid]['label']}")
        if o["action"] == "drop":
            drop.add(iid)
        elif o["action"] == "relabel":
            if o["new_label"] not in CONDITION_4CLASS_ORDER:
                raise ValueError(f"{iid}: {o['new_label']} is not an active class")
            relabel[iid] = o["new_label"]
        else:
            raise ValueError(f"{iid}: unknown action {o['action']}")
    if drop & set(relabel):
        raise ValueError("an image is both dropped and relabelled")
    out = []
    for r in rows:
        if r["image_id"] in drop:
            continue
        r = dict(r)
        if r["image_id"] in relabel:
            r["label"] = relabel[r["image_id"]]
        out.append(r)
    return out, {"dropped": len(drop), "relabelled": len(relabel)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("F:/Solar AI Datasets/kaggle_staging/web_condition_v2_dataset"))
    ap.add_argument("--overrides", type=Path, default=_REPO / "datasets/ground_pv_research/web_condition_v2/label_overrides.csv")
    ap.add_argument("--out", type=Path, default=Path("F:/Solar AI Datasets/kaggle_staging/web_condition_v2_clean_dataset"))
    args = ap.parse_args(argv)

    rows = list(csv.DictReader((args.src / "folds.csv").open(encoding="utf-8")))
    new, summ = apply_overrides(rows, list(csv.DictReader(args.overrides.open(encoding="utf-8"))))
    if args.out.exists():
        shutil.rmtree(args.out)
    for r in new:
        old_rel = r["out_rel"]
        r["out_rel"] = f"{old_rel.split('/')[0]}/{r['label']}/{Path(old_rel).name}"
        dst = args.out / r["out_rel"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.src / old_rel, dst)
    with (args.out / "folds.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(new[0]))
        w.writeheader()
        w.writerows(new)
    shutil.copy2(args.overrides, args.out / "label_overrides.csv")
    (args.out / "README.md").write_text(
        "# Solar AI web-only condition pool v2-clean (PRIVATE, research only)\n\nv2 pool with 6 non-informative images dropped and "
        "5 labels corrected by AI visual review (label_overrides.csv, unverified by the owner). Same folds as v2. "
        "Licences of all sources unresolved; no first-party images.\n", encoding="utf-8")
    (args.out / "dataset-metadata.json").write_text(json.dumps({
        "title": "Solar AI web-only condition pool v2 clean", "id": DATASET_ID, "licenses": [{"name": "other"}],
        "subtitle": "Cleaned web PV panel photos, 4 conditions, grouped CV folds",
        "description": "PRIVATE research dataset: v2 pool minus 6 non-informative images with 5 AI-reviewed label corrections. "
                       "No first-party images; source licences unresolved; not redistributable."}, indent=2), encoding="utf-8")
    counts = {c: sum(1 for r in new if r["label"] == c) for c in CONDITION_4CLASS_ORDER}
    print(summ, len(new), counts, "->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
