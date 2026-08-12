"""Tests for non-fabricating model artifact diagnostics."""

from pathlib import Path

from models.model_manager import ModelManager


def test_artifact_status_reports_configured_paths_without_creating_files():
    manager = ModelManager()
    status = manager.artifact_status

    assert set(status) == {"YOLO", "MobileNet", "XGBoost"}
    for entry in status.values():
        assert isinstance(entry["path"], str)
        assert isinstance(entry["exists"], bool)
        assert Path(entry["path"]).is_file() == entry["exists"]


def test_missing_artifacts_are_reported_as_missing_not_loaded():
    manager = ModelManager()
    status = manager.artifact_status

    # The repository intentionally does not contain trained weights.
    # This test verifies the diagnostic contract without fabricating artifacts.
    if not any(item["exists"] for item in status.values()):
        assert manager.loaded_models == {
            "YOLO": False,
            "MobileNet": False,
            "XGBoost": False,
        }
