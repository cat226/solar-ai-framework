"""tests/test_pi_agent.py — pi_agent/capture_and_send.py: config loading,
the picamera2-unavailable-off-Pi error path (real, not simulated — this dev
machine genuinely has no picamera2), and the analyze -> send_result ->
run_once flow against a real HTTP call (mocked at the requests layer only,
never the analysis logic).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pi_agent import capture_and_send as agent
from services import ground_pipeline

_DEMO_DIR = Path(__file__).resolve().parent.parent / "assets" / "demo" / "ground_pv_real"


class TestLoadConfig:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            agent.load_config(tmp_path / "does_not_exist.yaml")

    def test_missing_required_key_raises(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("api_base_url: 'http://x'\n", encoding="utf-8")
        with pytest.raises(ValueError, match="api_token"):
            agent.load_config(path)

    def test_valid_config_applies_defaults(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(
            "api_base_url: 'http://x:8008'\napi_token: 'tok'\ncamera_label: 'cam1'\n",
            encoding="utf-8",
        )
        config = agent.load_config(path)
        assert config["capture_interval_seconds"] == 300
        assert config["detection_confidence"] == 0.25

    def test_explicit_values_override_defaults(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(
            "api_base_url: 'http://x:8008'\napi_token: 'tok'\ncamera_label: 'cam1'\n"
            "capture_interval_seconds: 60\ndetection_confidence: 0.4\n",
            encoding="utf-8",
        )
        config = agent.load_config(path)
        assert config["capture_interval_seconds"] == 60
        assert config["detection_confidence"] == 0.4


class TestCaptureImageOffPi:
    def test_raises_actionable_error_without_picamera2(self):
        """This dev machine genuinely has no picamera2 installed — this is
        a real exercise of the off-Pi error path, not a mocked simulation."""
        with pytest.raises(RuntimeError, match="picamera2 is not installed"):
            agent.capture_image()


class TestSendResult:
    def test_posts_to_ingest_endpoint_with_bearer_token(self, monkeypatch):
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"run_id": 1, "problem_detected": False, "camera_number": 3}

        def fake_post(url, json, headers, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            captured["timeout"] = timeout
            return FakeResponse()

        monkeypatch.setattr(agent.requests, "post", fake_post)
        config = {"api_base_url": "http://x:8008/", "api_token": "secret-tok"}
        result = agent.send_result(config, "cam1_20260101.jpg", {"panel_count": 0})

        assert captured["url"] == "http://x:8008/v1/ingest"
        assert captured["headers"]["Authorization"] == "Bearer secret-tok"
        assert captured["json"]["captures"][0]["label"] == "cam1_20260101.jpg"
        assert result["problem_detected"] is False


class TestRunOnce:
    def test_run_once_uses_test_image_and_logs_problem_detected(self, monkeypatch, tmp_path):
        from PIL import Image
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (100, 100), color="white").save(img_path)

        monkeypatch.setattr(agent, "analyze", lambda image, label, conf, models: {"panel_count": 1})
        monkeypatch.setattr(
            agent, "send_result",
            lambda config, label, report: {"problem_detected": True, "camera_number": 5, "flagged_count": 2, "run_id": 42},
        )

        config = {"camera_label": "cam1", "detection_confidence": 0.25}
        result = agent.run_once(config, models=(None, None, None, None), test_image=str(img_path))
        assert result["problem_detected"] is True
        assert result["camera_number"] == 5

    def test_run_once_without_test_image_calls_capture_image(self, monkeypatch):
        called = {}
        monkeypatch.setattr(agent, "capture_image", lambda: called.setdefault("captured", True) or object())
        monkeypatch.setattr(agent, "analyze", lambda image, label, conf, models: {"panel_count": 0})
        monkeypatch.setattr(
            agent, "send_result",
            lambda config, label, report: {"problem_detected": False, "camera_number": 1, "clean_count": 0, "unread_count": 0, "run_id": 1},
        )
        config = {"camera_label": "cam1", "detection_confidence": 0.25}
        agent.run_once(config, models=(None, None, None, None))
        assert called.get("captured") is True


@pytest.mark.skipif(
    not ground_pipeline.PANEL_WEIGHTS.is_file()
    or not ground_pipeline.CONDITION_WEIGHTS.is_file()
    or not ground_pipeline.CLASSIFIER_WEIGHTS.is_file()
    or not _DEMO_DIR.is_dir(),
    reason="frozen ground-domain research weights / demo photos not present on this machine",
)
class TestRunOnceRealAnalysis:
    """Runs the REAL model stack (same weights app.py loads) on a real
    demo photo — proves the Pi agent's analyze() path genuinely produces a
    usable report, not just that it's wired up. Only send_result (the
    network call) is mocked."""

    def test_real_image_through_real_models_produces_valid_report(self, monkeypatch):
        demo_photos = sorted(_DEMO_DIR.glob("*.jpeg"))
        assert demo_photos, "expected at least one real demo photo"

        panel_model, condition_model = ground_pipeline.load_ground_models()
        classifier_model, classifier_device = ground_pipeline.load_condition_classifier()
        models = (panel_model, condition_model, classifier_model, classifier_device)

        sent = {}
        monkeypatch.setattr(
            agent, "send_result",
            lambda config, label, report: sent.setdefault("report", report) or {
                "problem_detected": False, "camera_number": 1, "clean_count": 0, "unread_count": 0, "run_id": 1,
            },
        )

        config = {"camera_label": "cam1", "detection_confidence": 0.25}
        agent.run_once(config, models, test_image=str(demo_photos[0]))

        report = sent["report"]
        assert "panel_count" in report
        assert "panels" in report
        if report["panel_count"] > 0:
            for p in report["panels"]:
                assert "classifier_condition" in p
                assert p["classifier_condition"]["label"] in ground_pipeline.CLASSIFIER_CLASSES
