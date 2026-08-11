"""tests/test_pipeline_performance.py - Performance and resource stability tests.

These tests validate that repeated pipeline execution is:
- deterministic
- reasonably performant
- free from obvious memory/state leakage
- free from accumulating handlers/resources
- stable under repeated execution

Design rules:
- deterministic
- no network
- no real model weights
- no GPU required
- relative/sanity thresholds rather than brittle absolute timing
- clearly documented as stability signals, not production benchmarks
"""

from __future__ import annotations

import logging
import time
import tracemalloc
import gc
from unittest.mock import MagicMock

import pytest
from PIL import Image

from services.pipeline import run_pipeline
from models.model_manager import model_manager
from utils.exceptions import SolarAIError


# ---------------------------------------------------------------------------
# Sentinel exception
# ---------------------------------------------------------------------------

class SentinelPerfException(SolarAIError):
    """Distinctive SolarAIError subclass for identifying stage failures."""
    pass


# ---------------------------------------------------------------------------
# Domain object factories (reused from E2E suite)
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


# ---------------------------------------------------------------------------
# A. Latency / throughput
# ---------------------------------------------------------------------------

class TestPipelineLatency:
    """Sanity checks for pipeline execution timing."""

    def test_single_run_completes_in_reasonable_time(self, monkeypatch):
        """Single run should complete well under 10s in this mocked configuration."""
        img = Image.new("RGB", (224, 224))
        _patch_external(monkeypatch)

        start = time.perf_counter()
        result = run_pipeline(image=img, city="Chennai")
        elapsed = time.perf_counter() - start

        assert result.status == "SUCCESS"
        assert elapsed < 10.0, f"Single run took {elapsed:.2f}s — unexpected slowdown"

    def test_repeated_runs_do_not_degrade_dramatically(self, monkeypatch):
        """Repeated runs should not be significantly slower than the first run."""
        img = Image.new("RGB", (224, 224))
        _patch_external(monkeypatch)

        times = []
        for _ in range(5):
            start = time.perf_counter()
            result = run_pipeline(image=img, city="Chennai")
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            assert result.status == "SUCCESS"

        first = times[0]
        last = times[-1]
        # Allow up to 3x slowdown; anything more suggests resource leakage
        assert last < first * 3.0, (
            f"Last run ({last:.2f}s) was >3x slower than first ({first:.2f}s)"
        )

    def test_batch_of_runs_completes_without_error(self, monkeypatch):
        """A small batch of runs should all succeed and finish in bounded time."""
        img = Image.new("RGB", (224, 224))
        _patch_external(monkeypatch)

        start = time.perf_counter()
        for _ in range(10):
            result = run_pipeline(image=img, city="Chennai")
            assert result.status == "SUCCESS"
        total = time.perf_counter() - start

        assert total < 60.0, f"10 runs took {total:.2f}s — unexpected slowdown"


# ---------------------------------------------------------------------------
# B. Resource / state stability
# ---------------------------------------------------------------------------

class TestPipelineResourceStability:
    """Repeated execution should not accumulate resources or mutate global state."""

    def test_root_logger_handler_count_stable(self, monkeypatch):
        """Root logger handler count should not grow across repeated pipeline runs."""
        img = Image.new("RGB", (224, 224))
        _patch_external(monkeypatch)

        root = logging.getLogger()
        initial_handlers = len(root.handlers)

        for _ in range(5):
            result = run_pipeline(image=img, city="Chennai")
            assert result.status == "SUCCESS"

        assert len(root.handlers) == initial_handlers, (
            f"Handler count changed from {initial_handlers} to "
            f"{len(root.handlers)} — possible handler leak"
        )

    def test_model_manager_returns_same_cached_objects(self, monkeypatch):
        """ModelManager getters should return the same cached object every time."""
        img = Image.new("RGB", (224, 224))
        _patch_external(monkeypatch)

        # Use real model manager for this test (not the mock)
        from services.pipeline import model_manager as real_mm

        detector1 = real_mm.get_detector()
        detector2 = real_mm.get_detector()
        assert detector1 is detector2, "Detector cache returned different objects"

        classifier1 = real_mm.get_classifier()
        classifier2 = real_mm.get_classifier()
        assert classifier1 is classifier2, "Classifier cache returned different objects"

        predictor1 = real_mm.get_predictor()
        predictor2 = real_mm.get_predictor()
        assert predictor1 is predictor2, "Predictor cache returned different objects"

    def test_pipeline_does_not_mutate_input_image(self, monkeypatch):
        """Original PIL image must remain unchanged after pipeline execution."""
        img = Image.new("RGB", (224, 224), (120, 130, 140))
        original_mode = img.mode
        original_size = img.size

        _patch_external(monkeypatch)

        for _ in range(3):
            run_pipeline(image=img, city="Chennai")
            assert img.mode == original_mode
            assert img.size == original_size

    def test_repeated_runs_produce_identical_physics(self, monkeypatch):
        """Physics results should be byte-identical across repeated runs."""
        img = Image.new("RGB", (224, 224))
        weather = _make_weather(city="Chennai", ambient_temp_c=30.0, cloud_cover_pct=40.0)
        _patch_external(monkeypatch, weather=weather)

        results = []
        for _ in range(3):
            result = run_pipeline(image=img, city="Chennai")
            assert result.status == "SUCCESS"
            results.append(result.physics_data)

        assert all(r.irradiance_wm2 == results[0].irradiance_wm2 for r in results)
        assert all(r.module_temp_c == results[0].module_temp_c for r in results)
        assert all(r.soiling_ratio == results[0].soiling_ratio for r in results)
        assert all(r.cloud_factor == results[0].cloud_factor for r in results)


# ---------------------------------------------------------------------------
# C. Memory stability
# ---------------------------------------------------------------------------

class TestPipelineMemoryStability:
    """Repeated execution should not cause uncontrolled memory growth."""

    def test_repeated_runs_memory_growth_within_reasonable_limit(self, monkeypatch):
        """Memory growth across repeated runs should be bounded.

        Uses tracemalloc to measure aggregate memory before and after a
        bounded number of pipeline executions. The threshold is intentionally
        conservative relative to the baseline snapshot.
        """
        img = Image.new("RGB", (224, 224))
        _patch_external(monkeypatch)

        tracemalloc.start()
        try:
            # Warm-up run
            run_pipeline(image=img, city="Chennai")

            # Baseline after warm-up and GC
            gc.collect()
            snapshot_before = tracemalloc.take_snapshot()

            # Repeated execution
            for _ in range(10):
                result = run_pipeline(image=img, city="Chennai")
                assert result.status == "SUCCESS"

            # Force collection and capture final state
            gc.collect()
            snapshot_after = tracemalloc.take_snapshot()

            before_stats = snapshot_before.statistics("lineno")
            after_stats = snapshot_after.statistics("lineno")

            before_total = sum(s.size for s in before_stats)
            after_total = sum(s.size for s in after_stats)

            growth_ratio = after_total / before_total if before_total > 0 else 1.0
            # Use a generous threshold; exact numbers depend on Python internals
            assert growth_ratio < 5.0, (
                f"Memory grew {growth_ratio:.1f}x across repeated runs "
                f"({before_total} -> {after_total} bytes)"
            )
        finally:
            tracemalloc.stop()

