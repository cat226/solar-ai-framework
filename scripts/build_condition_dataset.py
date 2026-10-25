"""scripts/build_condition_dataset.py — CLI wrapper around
training.detection.ground_pv.build_condition_dataset.

PERSONAL, NON-COMMERCIAL RESEARCH ONLY. Builds the unified 5-class
condition/defect research dataset (CLEAN / DUSTY / BIRD_DROP /
PHYSICAL_DAMAGE / ELECTRICAL_DAMAGE) from the audited S1 + S3 candidate
sources. Does NOT train anything.

    python scripts/build_condition_dataset.py --overwrite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.detection.ground_pv.build_condition_dataset import build  # noqa: E402

_CD = _ROOT / "datasets" / "ground_pv_research" / "candidate_defects" / "raw"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s1", type=Path, default=_CD / "S1_detection-for-defects_v2")
    ap.add_argument("--s3", type=Path, default=_CD / "S3_dusty-and-bird-droppings_v1")
    ap.add_argument("--out", type=Path,
                    default=_ROOT / "datasets" / "ground_pv_research" / "condition")
    ap.add_argument("--existing-yolo", type=Path,
                    default=_ROOT / "datasets" / "ground_pv_research" / "yolo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dhash-hamming", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rep = build(
        args.s1, args.s3, args.out,
        existing_yolo_dir=args.existing_yolo,
        seed=args.seed, dhash16_merge_hamming=args.dhash_hamming,
        overwrite=args.overwrite,
    )
    g = rep.quality_gates
    print(f"derived images        : {rep.derived_image_count}")
    print(f"splits                : {rep.final_split_counts}")
    print(f"class image counts    : {rep.class_image_counts.get('total')}")
    print(f"electrical_damage     : {rep.electrical_damage_images} images "
          f"/ {rep.electrical_damage_instances} boxes (semantic_risk=HIGH)")
    print(f"no family crosses split: {rep.no_family_crosses_splits}")
    print(f"quality gates all pass : {g.get('all_pass')}")
    print(f"output                : {args.out}")
    return 0 if g.get("all_pass") and rep.no_family_crosses_splits else 1


if __name__ == "__main__":
    raise SystemExit(main())
