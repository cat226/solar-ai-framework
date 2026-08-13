"""Tests for explicit liveness versus inference-readiness semantics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scripts.check_runtime_readiness as readiness


class _ReadyManager:
    @property
    def artifact_status(self):
        return {
            "YOLO": {"exists": True},
            "MobileNet": {"exists": True},
            "XGBoost": {"exists": True},
        }


class _MissingManager:
    @property
    def artifact_status(self):
        return {
            "YOLO": {"exists": False},
            "MobileNet": {"exists": True},
            "XGBoost": {"exists": False},
        }


def test_readiness_reports_ready_when_all_artifacts_exist(monkeypatch, capsys):
    monkeypatch.setattr(readiness, "model_manager", _ReadyManager())

    assert readiness.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "inference_readiness": "ready",
        "liveness": "ok",
        "missing_artifacts": [],
    }


def test_readiness_reports_not_ready_without_fabricating_artifacts(monkeypatch, capsys):
    monkeypatch.setattr(readiness, "model_manager", _MissingManager())

    assert readiness.main() == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["liveness"] == "ok"
    assert payload["inference_readiness"] == "not_ready"
    assert payload["missing_artifacts"] == ["YOLO", "XGBoost"]


def test_direct_execution_from_repo_root_works():
    """Running ``python scripts/check_runtime_readiness.py`` from the repo root
    must not raise ModuleNotFoundError and must report not_ready without weights."""
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "check_runtime_readiness.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2, (
        f"Expected exit code 2 (not_ready), got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "ModuleNotFoundError" not in result.stderr
    assert "No module named 'models'" not in result.stderr

    payload = json.loads(result.stdout)
    assert payload["liveness"] == "ok"
    assert payload["inference_readiness"] == "not_ready"
    assert set(payload["missing_artifacts"]) == {"YOLO", "MobileNet", "XGBoost"}
