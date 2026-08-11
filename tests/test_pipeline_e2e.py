"""tests/test_pipeline_e2e.py - End-to-end pipeline validation tests.

These tests validate complete realistic scenarios through the full backend
pipeline, using real orchestration and real deterministic stages with
controlled model/API boundaries.

Real components:
- Input validation
- Image validation/conversion
- services.physics.compute_physics
- services.feature_engineering.build_feature_dataframe
- services.recommendation.generate_recommendations
- Pipeline orchestration and PipelineResult assembly

Simulated boundaries (no real weights/API):
- Model inference (detector, classifier, predictor)
- External weather API (fetch_weather)

Design rules:
- deterministic
- no network
- no real model weights
- no GPU required
- realistic domain scenarios
- complete output validation
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from PIL import Image

from services.pipeline import run_pipeline
from services.recommendation import Severity
from utils.exceptions import SolarAIError


# ---------------------------------------------------------------------------
# Lightweight stubs for heavy optional dependencies
# ---------------------------------------------------------------------------

if "torch" not in sys.modules:
    torch_stub = MagicMock()
    torch_stub.nn = MagicMock()
    torch_stub.nn.functional = MagicMock()
    torch_stub.device = MagicMock()
    torch_stub.cuda = MagicMock()
    torch_stub.cuda.is_available = MagicMock(return_value=False)
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = torch_stub.nn
    sys.modules["torch.nn.functional"] = torch_stub.nn.functional
if "torchvision" not in sys.modules:
    tv_stub = MagicMock()
    tv_stub.transforms = MagicMock()
    sys.modules["torchvision"] = tv_stub
    sys.modules["torchvision.transforms"] = tv_stub.transforms


# ---------------------------------------------------------------------------
# Sentinel exception for E2E tests
# ---------------------------------------------------------------------------

class SentinelE2EException(SolarAIError):
    """Distinctive SolarAIError subclass for identifying stage failures."""
    pass


# ---------------------------------------------------------------------------
# Domain object factories
# ---------------------------------------------------------------------------

def _make_detection(panel_count=1, confidence=0.9):
    from models.detector import DetectionResult
    return DetectionResult(
        boxes=[[10.0, 10.0, 210.0, 190.0]],
        confidences=[confidence],
        class_ids=[0],
        panel_count=panel_count,
        best_confidence=confidence,
        detection_successful=True,
    )


def _make_classification(label="Clean", confidence=0.95):
    from models.classifier import ClassificationResult
    return ClassificationResult(
        label=label,
        class_id=0,
        confidence=confidence,
        probabilities={
            "Clean": confidence,
            "Dusty": 0.02,
            "Bird-Drop": 0.01,
            "Electrical-Damage": 0.01,
            "Physical-Damage": 0.005,
            "Hotspot": 0.005,
        },
        classification_successful=True,
    )


def _make_weather(city="Chennai", fetch_successful=True, **kwargs):
    from services.weather import WeatherData
    defaults = {
        "ambient_temp_c": 25.0,
        "humidity_pct": 50.0,
        "wind_speed_ms": 2.0,
        "cloud_cover_pct": 30.0,
        "pressure_hpa": 1013.25,
        "latitude": 13.08,
        "longitude": 80.27,
        "timestamp": None,
        "description": "clear sky",
    }
    defaults.update(kwargs)
    return WeatherData(
        city=city,
        fetch_successful=fetch_successful,
        **defaults,
    )


def _make_prediction(loss_pct=5.0, output_w=380.0):
    from models.predictor import PredictionResult
    return PredictionResult(
        efficiency_loss_pct=loss_pct,
        estimated_output_w=output_w,
        prediction_successful=True,
    )


def _make_recommendation(severity="OK", summary="Panel is healthy"):
    from services.recommendation import RecommendationReport
    return RecommendationReport(
        recommendations=[],
        overall_severity=Severity(severity),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------

def _make_mock_detector(detection_result=None):
    detector = MagicMock()
    detector.detect.return_value = detection_result or _make_detection()
    return detector


def _make_mock_classifier(classification_result=None):
    clf = MagicMock()
    clf.classify.return_value = classification_result or _make_classification()
    return clf


def _make_mock_predictor(prediction_result=None):
    predictor = MagicMock()
    predictor.predict.return_value = prediction_result or _make_prediction()
    return predictor


# ---------------------------------------------------------------------------
# Patching helper
# ---------------------------------------------------------------------------

def _patch_external(monkeypatch, detection=None, classification=None,
                    weather=None, prediction=None):
    """Patch only external/heavy boundaries; real stages use production code."""
    mm = MagicMock()
    mm.get_detector.return_value = MagicMock()
    mm.get_classifier.return_value = MagicMock()
    mm.get_predictor.return_value = MagicMock()

    detector = _make_mock_detector(detection)
    clf = _make_mock_classifier(classification)
    predictor = _make_mock_predictor(prediction)
    weather_data = weather or _make_weather()

    monkeypatch.setattr("services.pipeline.model_manager", mm)
    monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: detector)
    monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: clf)
    monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: predictor)
    monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: weather_data)
    # Real implementations for physics, features, recommendations are used


# ---------------------------------------------------------------------------
# A. Realistic scenario: healthy panel
# ---------------------------------------------------------------------------

class TestE2EHealthyPanel:
    """Complete pipeline for a healthy solar panel."""

    def test_healthy_panel_produces_ok_recommendations(self, monkeypatch):
        img = Image.new("RGB", (224, 224), (100, 150, 200))
        detection = _make_detection(panel_count=2, confidence=0.95)
        classification = _make_classification(label="Clean", confidence=0.98)
        weather = _make_weather(city="Chennai", ambient_temp_c=28.0, cloud_cover_pct=20.0)
        prediction = _make_prediction(loss_pct=3.0, output_w=388.0)

        _patch_external(monkeypatch, detection=detection, classification=classification,
                        weather=weather, prediction=prediction)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert result.recommendations.overall_severity in (Severity.OK, Severity.INFO)

    def test_healthy_panel_output_w接近_rated_power(self, monkeypatch):
        img = Image.new("RGB", (224, 224), (120, 160, 210))
        detection = _make_detection(panel_count=1, confidence=0.92)
        classification = _make_classification(label="Clean", confidence=0.97)
        weather = _make_weather(city="Chennai", ambient_temp_c=25.0, cloud_cover_pct=10.0)
        prediction = _make_prediction(loss_pct=2.0, output_w=392.0)

        _patch_external(monkeypatch, detection=detection, classification=classification,
                        weather=weather, prediction=prediction)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert result.efficiency_prediction.estimated_output_w > 380.0


# ---------------------------------------------------------------------------
# B. Realistic scenario: dusty panel
# ---------------------------------------------------------------------------

class TestE2EDustyPanel:
    """Complete pipeline for a dusty solar panel."""

    def test_dusty_panel_generates_warning_recommendation(self, monkeypatch):
        img = Image.new("RGB", (224, 224), (140, 140, 140))
        detection = _make_detection(panel_count=1, confidence=0.88)
        classification = _make_classification(label="Dusty", confidence=0.91)
        weather = _make_weather(city="Chennai", ambient_temp_c=30.0, cloud_cover_pct=40.0)
        prediction = _make_prediction(loss_pct=12.0, output_w=352.0)

        _patch_external(monkeypatch, detection=detection, classification=classification,
                        weather=weather, prediction=prediction)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert result.recommendations.overall_severity in (Severity.WARNING, Severity.CRITICAL)
        assert any("Dust" in r.message or "cleaning" in r.action.lower()
                   for r in result.recommendations.recommendations)

    def test_dusty_panel_physics_soiling_reflected_in_features(self, monkeypatch):
        img = Image.new("RGB", (224, 224), (150, 150, 150))
        detection = _make_detection(panel_count=1, confidence=0.85)
        classification = _make_classification(label="Dusty", confidence=0.90)
        weather = _make_weather(city="Chennai", ambient_temp_c=32.0, cloud_cover_pct=35.0)

        _patch_external(monkeypatch, detection=detection, classification=classification,
                        weather=weather)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert result.physics_data.soiling_ratio < 1.0
        assert result.feature_dataframe["soiling_ratio"].iloc[0] == pytest.approx(
            result.physics_data.soiling_ratio
        )


# ---------------------------------------------------------------------------
# C. Realistic scenario: hot weather conditions
# ---------------------------------------------------------------------------

class TestE2EHotWeather:
    """Complete pipeline under high-temperature conditions."""

    def test_hot_weather_increases_module_temperature(self, monkeypatch):
        img = Image.new("RGB", (224, 224), (255, 200, 150))
        detection = _make_detection(panel_count=1, confidence=0.90)
        classification = _make_classification(label="Clean", confidence=0.95)
        weather = _make_weather(city="Chennai", ambient_temp_c=42.0, cloud_cover_pct=10.0,
                               wind_speed_ms=1.0)
        prediction = _make_prediction(loss_pct=8.0, output_w=368.0)

        _patch_external(monkeypatch, detection=detection, classification=classification,
                        weather=weather, prediction=prediction)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert result.physics_data.temp_loss_pct > 0.0

    def test_hot_weather_may_trigger_hotspot_recommendation(self, monkeypatch):
        img = Image.new("RGB", (224, 224), (250, 180, 120))
        detection = _make_detection(panel_count=1, confidence=0.92)
        classification = _make_classification(label="Hotspot", confidence=0.89)
        weather = _make_weather(city="Chennai", ambient_temp_c=40.0, cloud_cover_pct=5.0)

        _patch_external(monkeypatch, detection=detection, classification=classification,
                        weather=weather)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert result.recommendations.overall_severity == Severity.CRITICAL


# ---------------------------------------------------------------------------
# D. Realistic scenario: cold/cloudy weather
# ---------------------------------------------------------------------------

class TestE2ECloudyWeather:
    """Complete pipeline under low-irradiance conditions."""

    def test_cloudy_weather_low_irradiance(self, monkeypatch):
        img = Image.new("RGB", (224, 224), (180, 190, 200))
        detection = _make_detection(panel_count=1, confidence=0.87)
        classification = _make_classification(label="Clean", confidence=0.93)
        weather = _make_weather(city="Chennai", ambient_temp_c=18.0, cloud_cover_pct=90.0)
        prediction = _make_prediction(loss_pct=5.0, output_w=380.0)

        _patch_external(monkeypatch, detection=detection, classification=classification,
                        weather=weather, prediction=prediction)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert result.physics_data.irradiance_wm2 < 300.0
        from services.feature_engineering import build_features
        raw_features = build_features(
            weather=result.weather_data,
            physics=result.physics_data,
            classification=result.classification_result,
            detection=result.detection_result,
        )
        assert raw_features["cloud_factor"].iloc[0] < 0.9


# ---------------------------------------------------------------------------
# E. Complete output validation
# ---------------------------------------------------------------------------

class TestE2ECompleteOutputValidation:
    """Every PipelineResult field is populated and coherent."""

    def test_successful_result_has_complete_fields(self, monkeypatch):
        img = Image.new("RGB", (224, 224), (110, 140, 170))
        detection = _make_detection(panel_count=1, confidence=0.90)
        classification = _make_classification(label="Clean", confidence=0.96)
        weather = _make_weather(city="Chennai")
        prediction = _make_prediction(loss_pct=4.0, output_w=384.0)

        _patch_external(monkeypatch, detection=detection, classification=classification,
                        weather=weather, prediction=prediction)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert result.city == "Chennai"
        assert result.error_message == ""
        assert result.error_type == ""
        assert result.processing_time >= 0.0

        assert result.detection_result is not None
        assert result.classification_result is not None
        assert result.weather_data is not None
        assert result.physics_data is not None
        assert result.feature_dataframe is not None
        assert result.efficiency_prediction is not None
        assert result.recommendations is not None

    def test_processing_time_is_non_negative(self, monkeypatch):
        img = Image.new("RGB", (224, 224), (100, 100, 100))
        _patch_external(monkeypatch)

        result = run_pipeline(image=img, city="Chennai")
        assert result.processing_time >= 0.0


# ---------------------------------------------------------------------------
# F. Weather fallback with real downstream stages
# ---------------------------------------------------------------------------

class TestE2EWeatherFallback:
    """Weather API fallback produces valid physics and recommendations."""

    def test_weather_fallback_produces_valid_physics(self, monkeypatch):
        img = Image.new("RGB", (224, 224), (130, 150, 180))
        weather_fallback = _make_weather(fetch_successful=False)
        detection = _make_detection(panel_count=1, confidence=0.91)
        classification = _make_classification(label="Clean", confidence=0.94)

        _patch_external(monkeypatch, weather=weather_fallback,
                        detection=detection, classification=classification)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert result.weather_data.fetch_successful is False
        assert result.physics_data is not None
        assert result.feature_dataframe is not None
        assert result.recommendations is not None


# ---------------------------------------------------------------------------
# G. Determinism
# ---------------------------------------------------------------------------

class TestE2EDeterminism:
    """Repeated E2E runs produce identical results."""

    def test_repeated_runs_identical(self, monkeypatch):
        img = Image.new("RGB", (224, 224), (115, 145, 175))
        detection = _make_detection(panel_count=2, confidence=0.89)
        classification = _make_classification(label="Dusty", confidence=0.87)
        weather = _make_weather(city="Chennai", ambient_temp_c=27.0, cloud_cover_pct=35.0)
        prediction = _make_prediction(loss_pct=10.0, output_w=360.0)

        _patch_external(monkeypatch, detection=detection, classification=classification,
                        weather=weather, prediction=prediction)

        r1 = run_pipeline(image=img, city="Chennai")
        r2 = run_pipeline(image=img, city="Chennai")
        assert r1.status == r2.status == "SUCCESS"
        assert r1.physics_data.irradiance_wm2 == r2.physics_data.irradiance_wm2
        assert r1.physics_data.module_temp_c == r2.physics_data.module_temp_c
        assert r1.feature_dataframe.equals(r2.feature_dataframe)
        assert r1.efficiency_prediction.efficiency_loss_pct == r2.efficiency_prediction.efficiency_loss_pct


# ---------------------------------------------------------------------------
# H. Input immutability
# ---------------------------------------------------------------------------

class TestE2EInputImmutability:
    """Input image is not mutated by the pipeline."""

    def test_rgb_image_not_mutated(self, monkeypatch):
        img = Image.new("RGB", (224, 224), (120, 130, 140))
        original_mode = img.mode
        _patch_external(monkeypatch)

        run_pipeline(image=img, city="Chennai")
        assert img.mode == original_mode

    def test_rgba_image_conversion_does_not_mutate_original(self, monkeypatch):
        img = Image.new("RGBA", (224, 224), (255, 0, 0, 128))
        original_mode = img.mode
        _patch_external(monkeypatch)

        run_pipeline(image=img, city="Chennai")
        assert img.mode == original_mode
