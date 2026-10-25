"""tests/test_ground_pv_paths.py — default on-disk locations for the ground-PV
dataset resolve to the E: data root when available and fall back to
repo-relative paths otherwise (CI / no E: drive)."""
from __future__ import annotations

import importlib
from pathlib import Path


def test_paths_module_exposes_expected_names():
    from training.detection.ground_pv import paths
    for name in ("GROUND_PV_DATA_ROOT", "RAW_DATASET_DIR", "NORMALIZED_YOLO_DIR",
                 "SPLITS_DIR", "REAL_WORLD_HOLDOUT_DIR", "REPORTS_DIR"):
        assert isinstance(getattr(paths, name), Path)


def test_raw_dir_ends_with_canonical_dataset_name():
    from training.detection.ground_pv import paths
    assert paths.RAW_DATASET_DIR.name == "panel_solar_bw945_v1"
    assert paths.RAW_DATASET_DIR.parent.name == "raw"


def test_reports_dir_always_stays_in_the_repo():
    from training.detection.ground_pv import paths
    # small, git-tracked outputs never move to the bulk-data drive
    assert paths.REPORTS_DIR.as_posix().endswith("datasets/ground_pv_research/reports")


def test_fallback_to_repo_when_no_data_root(monkeypatch):
    import training.cloud.base.storage_paths as sp
    import training.detection.ground_pv.paths as paths
    monkeypatch.setattr(sp, "SOLAR_AI_DATA_ROOT", None, raising=False)
    reloaded = importlib.reload(paths)
    try:
        assert "datasets" in reloaded.GROUND_PV_DATA_ROOT.as_posix()
        assert reloaded.RAW_DATASET_DIR.name == "panel_solar_bw945_v1"
    finally:
        importlib.reload(paths)  # restore real module state for other tests


def test_data_root_used_when_present(monkeypatch, tmp_path):
    import training.cloud.base.storage_paths as sp
    import training.detection.ground_pv.paths as paths
    monkeypatch.setattr(sp, "SOLAR_AI_DATA_ROOT", tmp_path / "Solar AI Project Data", raising=False)
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.GROUND_PV_DATA_ROOT == tmp_path / "Solar AI Project Data" / "ground_level"
        assert reloaded.RAW_DATASET_DIR.parent.parent == reloaded.GROUND_PV_DATA_ROOT
    finally:
        importlib.reload(paths)
