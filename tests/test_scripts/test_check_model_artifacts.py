"""tests/test_scripts/test_check_model_artifacts.py - Tests for the artifact validator.

Covers:
- All artifacts present
- One artifact missing
- Multiple artifacts missing
- Paths resolve independently of current working directory
- Correct exit status
- Correct filenames in output
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "check_model_artifacts.py"


def _run_validator(project_root: Path) -> tuple[int, str]:
    """Run the validator script with an explicit --root and return (exit_code, stdout)."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--root", str(project_root)],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout


class TestArtifactValidator:
    """check_model_artifacts.py reports required artifact status."""

    def test_all_artifacts_present_returns_zero(self, tmp_path: Path):
        weights = tmp_path / "weights"
        weights.mkdir()
        for name in ("yolo_solar.pt", "mobilenet_solar.pth", "xgboost_solar.joblib"):
            (weights / name).write_text("fake")

        exit_code, stdout = _run_validator(tmp_path)
        assert exit_code == 0
        assert "[OK]" in stdout
        assert "All required model artifacts are present" in stdout

    def test_one_artifact_missing_returns_nonzero(self, tmp_path: Path):
        weights = tmp_path / "weights"
        weights.mkdir()
        (weights / "yolo_solar.pt").write_text("fake")
        (weights / "mobilenet_solar.pth").write_text("fake")
        # xgboost_solar.joblib is missing

        exit_code, stdout = _run_validator(tmp_path)
        assert exit_code == 1
        assert "[MISSING]" in stdout
        assert "xgboost_solar.joblib" in stdout

    def test_multiple_artifacts_missing_returns_nonzero(self, tmp_path: Path):
        weights = tmp_path / "weights"
        weights.mkdir()
        # No artifacts present

        exit_code, stdout = _run_validator(tmp_path)
        assert exit_code == 1
        assert stdout.count("[MISSING]") == 3
        assert "yolo_solar.pt" in stdout
        assert "mobilenet_solar.pth" in stdout
        assert "xgboost_solar.joblib" in stdout

    def test_paths_resolve_from_project_root_not_cwd(self, tmp_path: Path):
        """Validator uses --root, so CWD does not affect path resolution."""
        real_root = _SCRIPT.parent.parent
        real_weights = real_root / "weights"
        real_weights.mkdir(exist_ok=True)
        for name in ("yolo_solar.pt", "mobilenet_solar.pth", "xgboost_solar.joblib"):
            (real_weights / name).write_text("fake")

        try:
            exit_code, stdout = _run_validator(real_root)
            assert exit_code == 0
            assert "[OK]" in stdout
        finally:
            for name in ("yolo_solar.pt", "mobilenet_solar.pth", "xgboost_solar.joblib"):
                (real_weights / name).unlink(missing_ok=True)

    def test_no_files_created(self, tmp_path: Path):
        weights = tmp_path / "weights"
        weights.mkdir()

        _run_validator(tmp_path)

        assert not any(weights.iterdir()), "Validator must not create files"

    def test_output_contains_expected_filenames(self, tmp_path: Path):
        weights = tmp_path / "weights"
        weights.mkdir()
        (weights / "yolo_solar.pt").write_text("fake")
        # mobilenet and xgboost missing

        exit_code, stdout = _run_validator(tmp_path)
        assert exit_code == 1
        assert "yolo_solar.pt" in stdout
        assert "mobilenet_solar.pth" in stdout
        assert "xgboost_solar.joblib" in stdout
