"""training/detection/ground_pv/paths.py — Default on-disk locations for the
ground-level PV research dataset.

Per the project storage policy (see training/cloud/base/storage_paths.py),
large local Solar AI data lives on the E: drive
(``E:/Solar AI Training Images/...``), not the low-space C:/D: drives the
repo sits on. This module resolves the ground-PV dataset locations to that
root when it is available, and falls back to repo-relative paths otherwise
(CI, another machine, a contributor without an E: drive).

Small, git-tracked outputs (the audit report, manifests) always stay in the
repository at ``datasets/ground_pv_research/reports/`` — this module is only
about where the *bulk* image data lives.

Override with the ``SOLAR_AI_DATA_ROOT`` environment variable (honored by
storage_paths.py) or the ``--dest`` / ``--dataset-dir`` CLI flags.
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

try:  # storage_paths is local-only tooling; safe to import here (never on Kaggle)
    from training.cloud.base.storage_paths import SOLAR_AI_DATA_ROOT
except Exception:  # noqa: BLE001
    SOLAR_AI_DATA_ROOT = None  # type: ignore[assignment]

# Bulk data root: <E:/Solar AI Training Images>/ground_pv_research when the
# drive is available, else <repo>/datasets/ground_pv_research.
if SOLAR_AI_DATA_ROOT is not None:
    GROUND_PV_DATA_ROOT: Path = Path(SOLAR_AI_DATA_ROOT) / "ground_pv_research"
else:
    GROUND_PV_DATA_ROOT = _REPO_ROOT / "datasets" / "ground_pv_research"

RAW_DATASET_DIR: Path = GROUND_PV_DATA_ROOT / "raw" / "panel_solar_bw945_v1"
NORMALIZED_YOLO_DIR: Path = GROUND_PV_DATA_ROOT / "yolo"
SPLITS_DIR: Path = GROUND_PV_DATA_ROOT / "splits"
REAL_WORLD_HOLDOUT_DIR: Path = GROUND_PV_DATA_ROOT / "real_world_holdout"

# Reports stay in the repo (small, committed).
REPORTS_DIR: Path = _REPO_ROOT / "datasets" / "ground_pv_research" / "reports"


def describe() -> dict[str, str]:
    return {
        "solar_ai_data_root": str(SOLAR_AI_DATA_ROOT) if SOLAR_AI_DATA_ROOT else None,
        "ground_pv_data_root": str(GROUND_PV_DATA_ROOT),
        "raw_dataset_dir": str(RAW_DATASET_DIR),
        "normalized_yolo_dir": str(NORMALIZED_YOLO_DIR),
        "reports_dir": str(REPORTS_DIR),
    }
