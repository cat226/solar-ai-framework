"""Tests for explicit liveness versus inference-readiness semantics."""

from __future__ import annotations

import json

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
    monkeypatch.setattr(readiness, "ModelManager", _ReadyManager)

    assert readiness.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "inference_readiness": "ready",
        "liveness": "ok",
        "missing_artifacts": [],
    }


def test_readiness_reports_not_ready_without_fabricating_artifacts(monkeypatch, capsys):
    monkeypatch.setattr(readiness, "ModelManager", _MissingManager)

    assert readiness.main() == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["liveness"] == "ok"
    assert payload["inference_readiness"] == "not_ready"
    assert payload["missing_artifacts"] == ["YOLO", "XGBoost"]
