#!/usr/bin/env python3
"""scripts/verify_condition_4class_checkpoint.py — Phase 15B post-retrieval
verification: checks a downloaded training_manifest.json + its checkpoint
file are internally consistent (recorded checkpoint_sha256 matches the
actual file, all required reproducibility fields present, taxonomy is
exactly the active 4-class set).

    python scripts/verify_condition_4class_checkpoint.py \
        --manifest research_runs/ground_pv/condition_4class_kaggle/training_manifest.json \
        --checkpoint weights/yolo_solar_condition_4class_research_v2.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.verify_condition_checkpoint import verify_from_files  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        raise SystemExit(f"manifest not found: {args.manifest}")

    problems = verify_from_files(args.manifest, args.checkpoint)
    if problems:
        print("VERIFICATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("VERIFICATION OK: checkpoint and manifest are internally consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
