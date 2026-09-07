#!/usr/bin/env python3
"""scripts/remediate_ground_pv_dataset.py — Build the derived, leakage-clean,
3-class research-training dataset from the audited raw Roboflow dataset.

The raw dataset stays IMMUTABLE. This only reads it and writes a separate
derived tree + split manifests + a remediation report.

Steps (see training/detection/ground_pv/remediate.py):
  polygon-to-bbox; drop bird_drop (audit: 9 imgs / ~3 photos); de-dup the
  1 exact cross-split image; group near-duplicate FAMILIES before splitting
  (exact SHA-256 + source basename + 16x16 dHash); assign whole families to
  ~70/20/10 train/valid/test; preserve the official split separately.

Outputs:
  datasets/ground_pv_research/yolo/                       (derived dataset, gitignored)
  datasets/ground_pv_research/splits/official_split/       (provenance manifest)
  datasets/ground_pv_research/splits/leakage_clean_split/  (split manifest)
  datasets/ground_pv_research/reports/remediation_report.{md,json}   (committed)

Examples
--------
    python scripts/remediate_ground_pv_dataset.py
    python scripts/remediate_ground_pv_dataset.py --overwrite --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.remediate import (  # noqa: E402
    DEFAULT_DHASH16_MERGE_HAMMING,
    remediate,
)

try:
    from training.detection.ground_pv.paths import (  # noqa: E402
        RAW_DATASET_DIR, NORMALIZED_YOLO_DIR, SPLITS_DIR, REPORTS_DIR,
    )
except Exception:  # noqa: BLE001
    _d = _REPO_ROOT / "datasets" / "ground_pv_research"
    RAW_DATASET_DIR = _d / "raw" / "panel_solar_bw945_v1"
    NORMALIZED_YOLO_DIR, SPLITS_DIR, REPORTS_DIR = _d / "yolo", _d / "splits", _d / "reports"


_REPO_RAW = _REPO_ROOT / "datasets" / "ground_pv_research" / "raw" / "panel_solar_bw945_v1"
# The raw download normally lives under the E: data root, but E: has been
# unreliable so it may instead be at the repo-relative path. Prefer
# whichever actually exists.
_DEFAULT_RAW = (
    RAW_DATASET_DIR if (RAW_DATASET_DIR / "data.yaml").is_file()
    else _REPO_RAW if (_REPO_RAW / "data.yaml").is_file()
    else RAW_DATASET_DIR
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=_DEFAULT_RAW)
    p.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "datasets/ground_pv_research/yolo",
                   help="Derived dataset dir. Default is the repo-relative datasets/ path "
                        "(where the raw download already lives), NOT the E: data root.")
    p.add_argument("--splits-dir", type=Path, default=_REPO_ROOT / "datasets/ground_pv_research/splits")
    p.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--valid", type=float, default=0.20)
    p.add_argument("--test", type=float, default=0.10)
    p.add_argument("--dhash16-merge-hamming", type=int, default=DEFAULT_DHASH16_MERGE_HAMMING)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    if not (args.raw_dir / "data.yaml").is_file():
        sys.stderr.write(f"raw dataset not found at {args.raw_dir}\n")
        return 2

    print(f"Remediating {args.raw_dir}\n  -> {args.out_dir}", flush=True)
    rep = remediate(
        args.raw_dir, args.out_dir, args.splits_dir, args.reports_dir,
        ratios={"train": args.train, "valid": args.valid, "test": args.test},
        seed=args.seed, dhash16_merge_hamming=args.dhash16_merge_hamming,
        overwrite=args.overwrite,
    )
    print(f"\n  raw images        : {rep.raw_image_count}")
    print(f"  derived images    : {rep.derived_image_count}")
    print(f"  images removed    : {rep.images_removed_total} (exact dup: {rep.exact_duplicates_removed})")
    print(f"  near-dup families : {rep.family_grouping['total_families_over_kept']} "
          f"(largest {rep.family_grouping['largest_family_images']} imgs)")
    print(f"  split counts      : {rep.final_split_counts}")
    print(f"  unique fams/split : {rep.unique_groups_per_split}")
    print(f"  class counts      : {rep.class_counts['total']}")
    print(f"  polygon->bbox     : {rep.polygon_to_box_conversions}")
    print(f"  bird_drop excluded: {rep.bird_drop_annotations_excluded} annots / "
          f"{rep.bird_drop_images_affected} imgs")
    print(f"  NO GROUP CROSSES SPLITS: {rep.no_group_crosses_splits}")
    print(f"  residual x-split near-dup pairs: {rep.residual_cross_split_near_dup_pairs}")
    print(f"\n  report: {args.reports_dir / 'remediation_report.md'}")
    return 0 if rep.no_group_crosses_splits else 1


if __name__ == "__main__":
    raise SystemExit(main())
