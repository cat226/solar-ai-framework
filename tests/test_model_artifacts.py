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

    # YOLO/XGBoost each have exactly one configured artifact, so "exists"
    # tracks that one path directly.
    for name in ("YOLO", "XGBoost"):
        entry = status[name]
        assert Path(entry["path"]).is_file() == entry["exists"]

    # MobileNet is deliberately different: "exists" reports whether a usable
    # classifier is available at all (either the v1 release artifact or the
    # future six-class one), not only whether the one path shown here (the
    # future six-class artifact) is present - see ModelManager.artifact_status
    # and mobilenet_status for the full v1/six-class breakdown.
    mn = status["MobileNet"]
    mn_status = manager.mobilenet_status
    assert mn["exists"] == (mn_status["v1_exists"] or mn_status["six_class_exists"])


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
