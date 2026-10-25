#!/usr/bin/env python3
"""scripts/build_condition_4class_dataset.py — derive the active 4-class
condition dataset (`condition_4class_v1`) from the existing 5-class
research dataset (`datasets/ground_pv_research/condition/`, read-only).

Removes every ELECTRICAL_DAMAGE annotation (never relabels it as
PHYSICAL_DAMAGE or any other class); drops images whose annotations were
entirely ELECTRICAL_DAMAGE; preserves the existing leakage-clean
train/valid/test split exactly (a class-filtered subset of a leakage-clean
split stays leakage-clean by construction — no family/split logic is
recomputed).

Example
-------
    python scripts/build_condition_4class_dataset.py --overwrite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.build_condition_4class_dataset import build  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path,
                        default=_REPO_ROOT / "datasets" / "ground_pv_research" / "condition")
    parser.add_argument("--derived-dir", type=Path,
                        default=_REPO_ROOT / "datasets" / "ground_pv_research" / "condition_4class_v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    rep = build(args.source_dir, args.derived_dir, overwrite=args.overwrite)

    print(f"source images (5-class): {rep.source_image_count}")
    print(f"derived images (4-class): {rep.derived_image_count}")
    print(f"ELECTRICAL_DAMAGE instances removed: {rep.electrical_damage_instances_removed}")
    print(f"images dropped (all-electrical-damage): {rep.electrical_damage_only_images_dropped}")
    print(f"split counts: {rep.final_split_counts}")
    print(f"quality gates all pass: {rep.quality_gates.get('all_pass')}")
    print(f"\nreport: {args.derived_dir / 'reports' / 'condition_4class_build_report.md'}")
    return 0 if rep.quality_gates.get("all_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
