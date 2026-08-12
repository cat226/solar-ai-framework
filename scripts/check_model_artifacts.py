#!/usr/bin/env python3
"""scripts/check_model_artifacts.py — Validate required model artifacts exist.

This script checks whether the three trained model artifacts expected by the
Solar AI Framework are present in the weights/ directory.

It is intentionally read-only and never downloads or generates anything.

Exit codes:
    0 — all artifacts present
    1 — one or more artifacts missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    """Resolve the project root directory.

    By default this is the parent of the scripts/ directory.
    Can be overridden with the --root flag for testing or custom layouts.
    """
    parser = argparse.ArgumentParser(
        description="Check that required model artifacts exist."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project root directory (default: auto-detected from script location).",
    )
    args = parser.parse_args()
    return args.root


_REQUIRED_ARTIFACTS: list[tuple[str, Path]] = [
    ("yolo_solar.pt", None),
    ("mobilenet_solar.pth", None),
    ("xgboost_solar.joblib", None),
]


def check_artifacts(project_root: Path) -> int:
    """Check each required artifact and print status.

    Args:
        project_root: Root directory of the project.

    Returns:
        0 if all artifacts exist, 1 otherwise.
    """
    weights_dir = project_root / "weights"
    paths = [(name, weights_dir / name) for name, _ in _REQUIRED_ARTIFACTS]

    all_ok = True
    for name, path in paths:
        if path.exists():
            print(f"[OK]    {path.relative_to(project_root)}")
        else:
            print(f"[MISSING] {path.relative_to(project_root)}")
            all_ok = False

    if all_ok:
        print("\nAll required model artifacts are present.")
        return 0
    else:
        print("\nMissing model artifacts.")
        print("Obtain the trained models and place them in weights/.")
        print("See weights/README.md for details.")
        return 1


if __name__ == "__main__":
    root = _resolve_project_root()
    sys.exit(check_artifacts(root))
