#!/usr/bin/env python3
"""scripts/audit_ground_pv_dataset.py — Run the full 23-point audit on the
downloaded ground-level PV research dataset and write:

    datasets/ground_pv_research/reports/dataset_audit.md
    datasets/ground_pv_research/reports/dataset_audit.json

Exit code is non-zero when the audit fails closed (structure broken,
annotations invalid, >5% corrupt images, cross-split exact leakage, ...).
The raw dataset is never modified.

Examples
--------
    python scripts/audit_ground_pv_dataset.py
    python scripts/audit_ground_pv_dataset.py --dataset-dir datasets/ground_pv_research/raw/panel_solar_bw945_v1
    python scripts/audit_ground_pv_dataset.py --near-dup-hamming 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.audit import (  # noqa: E402
    NEAR_DUP_HAMMING,
    TINY_OBJECT_AREA_FRAC,
    run_audit,
    write_reports,
)

try:
    from training.detection.ground_pv.paths import RAW_DATASET_DIR as DEFAULT_DATASET_DIR
    from training.detection.ground_pv.paths import REPORTS_DIR as DEFAULT_REPORTS_DIR
except Exception:  # noqa: BLE001
    DEFAULT_DATASET_DIR = _REPO_ROOT / "datasets" / "ground_pv_research" / "raw" / "panel_solar_bw945_v1"
    DEFAULT_REPORTS_DIR = _REPO_ROOT / "datasets" / "ground_pv_research" / "reports"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--near-dup-hamming", type=int, default=NEAR_DUP_HAMMING)
    parser.add_argument("--tiny-object-area-frac", type=float, default=TINY_OBJECT_AREA_FRAC)
    parser.add_argument("--max-corruption-fraction", type=float, default=0.05)
    args = parser.parse_args(argv)

    if not args.dataset_dir.is_dir():
        sys.stderr.write(
            f"dataset directory not found: {args.dataset_dir}\n"
            "Run  python scripts/download_ground_pv_dataset.py  first "
            f"(needs the ROBOFLOW_API_KEY environment variable).\n"
        )
        return 2

    print(f"Auditing {args.dataset_dir} ...", flush=True)
    result = run_audit(
        args.dataset_dir,
        near_dup_hamming=args.near_dup_hamming,
        tiny_object_area_frac=args.tiny_object_area_frac,
        max_corruption_fraction=args.max_corruption_fraction,
    )
    json_path, md_path = write_reports(result, args.reports_dir)

    payload = result.as_dict()
    t = payload["totals"]
    print(f"\n  images={t['images']} labels={t['labels']} boxes={t['boxes']}")
    print(f"  class instances: {result.class_instance_counts}")
    print(f"  images per class: {result.images_per_class}")
    print(f"  corrupt={t['corrupt_images']}  exact-dup-groups={t['exact_duplicate_groups']}  "
          f"near-dup-pairs={t['near_duplicate_pairs']}")
    print(f"  cross-split exact leakage={t['cross_split_exact_leakage']}  "
          f"near leakage={t['cross_split_near_leakage']}")
    print(f"\n  reports: {md_path}")
    print(f"           {json_path}")

    if result.fail_closed_reasons:
        print("\nAUDIT FAILED CLOSED:")
        for r in result.fail_closed_reasons:
            print(f"  - {r}")
        return 1
    print("\nAUDIT PASSED (mechanical). Human review of the report is still required "
          "before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
