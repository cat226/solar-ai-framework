#!/usr/bin/env python3
"""scripts/build_rooftop_condition_mix_package.py - Phase 19-B package builder.

The rooftop condition crops alone are far too few/skewed to train a standalone
classifier (scripts/build_rooftop_condition_classifier_dataset.py correctly
refuses - see its output). This instead packages the AI-reviewed rooftop crops
as their own small Kaggle dataset, to be MIXED into the existing, much larger
`solar-ai-ground-condition-4class-classifier-v1` (Susan-derived) training set
at train time, oversampled - the same MIX strategy that worked for the Phase 19
panel detector (recall 0.125->0.81-0.94 with no regression on the base dataset).

Labels are `annotated_ai_assisted_unverified` in spirit (visual review by an
AI reviewer, not the property owner) - disclosed in every output file.
Non-CLEAN examples are extremely few here (this rooftop batch happens to be
mostly clean); this package exists for DOMAIN adaptation (this camera, this
lighting, this panel brand's texture), not to teach new class definitions.

Output: kaggle/rooftop_condition_dataset/{train,valid,test}/{CLASS}/*.jpg
        + dataset-metadata.json + build_report.json (counts, disclosure)
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["BIRD_DROP", "CLEAN", "DUSTY", "PHYSICAL_DAMAGE"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=_ROOT / "datasets/ground_pv_research/rooftop_condition_v1")
    ap.add_argument("--out", type=Path, default=_ROOT / "kaggle/rooftop_condition_dataset")
    args = ap.parse_args(argv)

    with (args.root / "labeling_sheet.csv").open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["label"].strip().upper() not in ("", "SKIP")]
    bad = sorted({r["label"] for r in rows if r["label"].strip().upper() not in CLASSES})
    if bad:
        raise SystemExit(f"unknown label(s) {bad}")

    if args.out.exists():
        shutil.rmtree(args.out)
    counts = Counter()
    for r in rows:
        label = r["label"].strip().upper()
        dst = args.out / r["split"] / label
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.root / "crops" / r["split"] / f"{r['crop_id']}.jpg", dst / f"{r['crop_id']}.jpg")
        counts[(r["split"], label)] += 1

    report = {
        "role": "DOMAIN-ADAPTATION TOP-UP ONLY - too few/skewed to be a standalone dataset "
                "(see scripts/build_rooftop_condition_classifier_dataset.py's own refusal)",
        "label_provenance": "AI visual review (Claude), NOT the property owner - disclosed, not owner-verified",
        "clean_definition": "no visible dust/bird-drop/damage in the photo; does not rule out dust invisible at this resolution",
        "counts": {f"{sp}/{cls}": n for (sp, cls), n in sorted(counts.items())},
        "total_labeled": len(rows),
    }
    (args.out / "build_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (args.out / "dataset-metadata.json").write_text(json.dumps({
        "title": "Solar AI rooftop condition top-up v1",
        "id": "edithstark/solar-ai-rooftop-condition-topup-v1",
        "licenses": [{"name": "other"}],
        "subtitle": "Phase 19-B first-party rooftop condition crops, AI-reviewed, domain top-up only",
        "description": "PRIVATE. AI-visually-reviewed (not owner-verified) CLEAN/DUSTY/BIRD_DROP/"
                        "PHYSICAL_DAMAGE crops from one rooftop, 241 total, heavily skewed toward CLEAN. "
                        "Too few/skewed to train a standalone classifier - meant to be mixed into the "
                        "much larger solar-ai-ground-condition-4class-classifier-v1 (Susan-derived) "
                        "training set for domain adaptation only.",
    }, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
