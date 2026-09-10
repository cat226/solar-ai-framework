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

# ---------------------------------------------------------------------------
# Phase 14B — representative deployment-domain panel dataset.
#
# Unlike GROUND_PV_DATA_ROOT above, this checks that SOLAR_AI_DATA_ROOT is
# actually a REACHABLE, MOUNTED directory (not just a configured string)
# before using it. The E: drive has repeatedly proven unstable across this
# project's history (disconnects mid-operation, goes read-only, or is simply
# absent - see Phase 13A/13B memory), so a bulk-data root for a NEW dataset
# area is only trusted here if it demonstrably exists right now; otherwise
# this falls back to the repo-relative D: location (same drive the repo
# itself lives on, which has had adequate free space in practice) rather
# than silently writing under a path that may vanish mid-write.
# ---------------------------------------------------------------------------


def _resolve_representative_panel_root() -> Path:
    if SOLAR_AI_DATA_ROOT is not None and Path(SOLAR_AI_DATA_ROOT).exists():
        return Path(SOLAR_AI_DATA_ROOT) / "ground_pv_research" / "representative_panel_v1"
    return _REPO_ROOT / "datasets" / "ground_pv_research" / "representative_panel_v1"


REPRESENTATIVE_PANEL_V1_ROOT: Path = _resolve_representative_panel_root()
REPRESENTATIVE_PANEL_V1_RAW_DIR: Path = REPRESENTATIVE_PANEL_V1_ROOT / "raw"
REPRESENTATIVE_PANEL_V1_ANNOTATIONS_DIR: Path = REPRESENTATIVE_PANEL_V1_ROOT / "annotations"
REPRESENTATIVE_PANEL_V1_DERIVED_DIR: Path = REPRESENTATIVE_PANEL_V1_ROOT / "derived"

# Small, git-tracked outputs (manifests, splits, reports) always stay in the
# repository, exactly like REPORTS_DIR above, regardless of where the bulk
# raw/annotations data resolved to.
REPRESENTATIVE_PANEL_V1_MANIFESTS_DIR: Path = (
    _REPO_ROOT / "datasets" / "ground_pv_research" / "representative_panel_v1" / "manifests"
)
REPRESENTATIVE_PANEL_V1_SPLITS_DIR: Path = (
    _REPO_ROOT / "datasets" / "ground_pv_research" / "representative_panel_v1" / "splits"
)
REPRESENTATIVE_PANEL_V1_REPORTS_DIR: Path = (
    _REPO_ROOT / "datasets" / "ground_pv_research" / "representative_panel_v1" / "reports"
)


def describe() -> dict[str, str]:
    return {
        "solar_ai_data_root": str(SOLAR_AI_DATA_ROOT) if SOLAR_AI_DATA_ROOT else None,
        "ground_pv_data_root": str(GROUND_PV_DATA_ROOT),
        "raw_dataset_dir": str(RAW_DATASET_DIR),
        "normalized_yolo_dir": str(NORMALIZED_YOLO_DIR),
        "reports_dir": str(REPORTS_DIR),
        "representative_panel_v1_root": str(REPRESENTATIVE_PANEL_V1_ROOT),
        "representative_panel_v1_root_reachable": Path(SOLAR_AI_DATA_ROOT).exists() if SOLAR_AI_DATA_ROOT else False,
        "representative_panel_v1_manifests_dir": str(REPRESENTATIVE_PANEL_V1_MANIFESTS_DIR),
        "representative_panel_v1_reports_dir": str(REPRESENTATIVE_PANEL_V1_REPORTS_DIR),
    }
