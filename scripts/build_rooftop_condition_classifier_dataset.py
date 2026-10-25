#!/usr/bin/env python3
"""scripts/build_rooftop_condition_classifier_dataset.py - turn the OWNER-filled
labeling sheet (scripts/export_rooftop_condition_labeling_sheet.py) into an ImageFolder
dataset that extends the existing 4-class condition classifier dataset.

Refuses to build (exit 2) unless the labeled crops are enough to mean something:
  * every class has >= --min-train train crops from >= --min-scenes distinct scenes, and
  * every class has >= --min-eval crops in valid AND test.
No condition label is ever inferred here - blank / SKIP rows are dropped.
Output: <out>/{train,valid,test}/<CLASS>/*.jpg  (rooftop crops only, alphabetical class dirs)
        <out>/build_report.json
Merge with datasets/ground_pv_research/condition_4class_classifier_v1 in the training
notebook (rooftop crops oversampled in train only; valid/test stay rooftop-only so the
reported number is the target-domain number).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["BIRD_DROP", "CLEAN", "DUSTY", "PHYSICAL_DAMAGE"]


def summarize(rows: list[dict]) -> dict:
    per = {sp: Counter() for sp in ("train", "valid", "test")}
    scenes: dict[str, set] = defaultdict(set)
    for r in rows:
        per[r["split"]][r["label"]] += 1
        if r["split"] == "train":
            scenes[r["label"]].add(r["scene_id"])
    return {"per_split": {k: dict(v) for k, v in per.items()}, "train_scenes_per_class": {k: len(v) for k, v in scenes.items()}}


def readiness(rows: list[dict], min_train: int, min_eval: int, min_scenes: int) -> list[str]:
    s = summarize(rows)
    problems = []
    for c in CLASSES:
        if s["per_split"]["train"].get(c, 0) < min_train:
            problems.append(f"{c}: train crops {s['per_split']['train'].get(c, 0)} < {min_train}")
        if s["train_scenes_per_class"].get(c, 0) < min_scenes:
            problems.append(f"{c}: train scenes {s['train_scenes_per_class'].get(c, 0)} < {min_scenes}")
        for sp in ("valid", "test"):
            if s["per_split"][sp].get(c, 0) < min_eval:
                problems.append(f"{c}: {sp} crops {s['per_split'][sp].get(c, 0)} < {min_eval}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=_ROOT / "datasets/ground_pv_research/rooftop_condition_v1")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-train", type=int, default=20)
    ap.add_argument("--min-eval", type=int, default=5)
    ap.add_argument("--min-scenes", type=int, default=2)
    args = ap.parse_args(argv)
    out = args.out or args.root / "classifier"

    with (args.root / "labeling_sheet.csv").open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["label"].strip().upper() not in ("", "SKIP")]
    bad = sorted({r["label"] for r in rows if r["label"].strip().upper() not in CLASSES})
    if bad:
        raise SystemExit(f"unknown label(s) {bad}; allowed: {CLASSES} / blank / SKIP")
    for r in rows:
        r["label"] = r["label"].strip().upper()

    report = {"labeled_crops": len(rows), **summarize(rows)}
    problems = readiness(rows, args.min_train, args.min_eval, args.min_scenes)
    report["ready"], report["blocking_reasons"] = not problems, problems
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "build_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if problems:
        print(json.dumps(report, indent=1))
        print("\nNOT READY - not building a dataset from too few / too concentrated labels.")
        return 2
    for r in rows:
        dst = out / r["split"] / r["label"]
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.root / "crops" / r["split"] / f"{r['crop_id']}.jpg", dst / f"{r['crop_id']}.jpg")
    print(json.dumps(report, indent=1))
    print(f"\nbuilt {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
