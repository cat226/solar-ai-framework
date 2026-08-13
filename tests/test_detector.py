"""tests/test_detector.py - Deterministic tests for models/detector.py.

Covers the YOLO detection wrapper:

A. Initialization and model injection
B. Validation and error handling
C. Successful detection
D. Result parsing from YOLO output
E. Configuration behavior
F. Logging behavior
G. Determinism

Design rules honoured:
- no real YOLO weights
- no network access
- deterministic mocks for YOLO inference
- existing conftest fixtures preferred
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from models.detector import DetectionResult, SolarPanelDetector
from utils.exceptions import ModelLoadError, PredictionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yolo_result(boxes=None, confs=None, cls_ids=None):
    """Create a mock YOLO prediction result with the expected structure."""
    pred = MagicMock()
    pred.boxes = MagicMock()
    
    if boxes is not None:
        pred.boxes.xyxy.cpu().numpy.return_value = np.array(boxes, dtype=np.float32)
    else:
        pred.boxes.xyxy.cpu().numpy.return_value = np.empty((0, 4), dtype=np.float32)
    
    if confs is not None:
        pred.boxes.conf.cpu().numpy.return_value = np.array(confs, dtype=np.float32)
    else:
        pred.boxes.conf.cpu().numpy.return_value = np.array([], dtype=np.float32)
    
    if cls_ids is not None:
        pred.boxes.cls.cpu().numpy.return_value = np.array(cls_ids, dtype=np.int32)
    else:
        pred.boxes.cls.cpu().numpy.return_value = np.array([], dtype=np.int32)
    
    return pred


def _make_mock_yolo_model():
    """Return a mock YOLO model that can be configured per test."""
    model = MagicMock()
    return model


# ---------------------------------------------------------------------------
# A. Initialization and model injection
# ---------------------------------------------------------------------------

class TestInitializationAndInjection:
    """SolarPanelDetector lifecycle basics."""

    def test_fresh_detector_has_no_model(self):
        detector = SolarPanelDetector()
        assert detector._model is None

    def test_set_model_stores_model(self):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        detector.set_model(fake_model)
        assert detector._model is fake_model

    def test_set_model_none_raises_model_load_error(self):
        detector = SolarPanelDetector()
        with pytest.raises(ModelLoadError, match="YOLO"):
            detector.set_model(None)

    def test_set_model_none_does_not_store_model(self):
        detector = SolarPanelDetector()
        with pytest.raises(ModelLoadError):
            detector.set_model(None)
        assert detector._model is None


# ---------------------------------------------------------------------------
# B. Validation and error handling
# ---------------------------------------------------------------------------

class TestValidationAndErrorHandling:
    """detect() validates state before inference."""

    def test_detect_before_set_model_raises_model_load_error(self):
        detector = SolarPanelDetector()
        img = Image.new("RGB", (640, 640))
        with pytest.raises(ModelLoadError, match="YOLO"):
            detector.detect(img)

    def test_detect_after_set_model_succeeds(self):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        fake_model.return_value = []  # empty results
        detector.set_model(fake_model)
        
        img = Image.new("RGB", (640, 640))
        result = detector.detect(img)
        assert isinstance(result, DetectionResult)

    def test_yolo_inference_exception_raises_prediction_error(self):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        fake_model.side_effect = RuntimeError("CUDA out of memory")
        detector.set_model(fake_model)
        
        img = Image.new("RGB", (640, 640))
        with pytest.raises(PredictionError, match="YOLO") as exc_info:
            detector.detect(img)
        assert "CUDA out of memory" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# C. Successful detection
# ---------------------------------------------------------------------------

class TestSuccessfulDetection:
    """detect() returns DetectionResult with parsed YOLO outputs."""

    def test_detect_returns_detection_result(self):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        fake_model.return_value = [_make_yolo_result()]
        detector.set_model(fake_model)
        
        img = Image.new("RGB", (640, 640))
        result = detector.detect(img)
        assert isinstance(result, DetectionResult)

    def test_detect_empty_results_returns_empty_detection(self):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        fake_model.return_value = [_make_yolo_result()]
        detector.set_model(fake_model)
        
        img = Image.new("RGB", (640, 640))
        result = detector.detect(img)
        assert result.panel_count == 0
        assert result.best_confidence == 0.0
        assert result.detection_successful is False
        assert result.boxes == []
        assert result.confidences == []
        assert result.class_ids == []

    def test_detect_single_panel_parsed_correctly(self):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        
        boxes = [[10.0, 20.0, 100.0, 200.0]]
        confs = [0.92]
        cls_ids = [0]
        fake_model.return_value = [_make_yolo_result(boxes, confs, cls_ids)]
        
        detector.set_model(fake_model)
        img = Image.new("RGB", (640, 640))
        result = detector.detect(img)
        
        assert result.panel_count == 1
        assert result.best_confidence == pytest.approx(0.92)
        assert result.detection_successful is True
        assert result.boxes == [[10.0, 20.0, 100.0, 200.0]]
        assert result.confidences == pytest.approx([0.92])
        assert result.class_ids == [0]

    def test_detect_multiple_panels_parsed_correctly(self):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        
        boxes = [[10.0, 20.0, 100.0, 200.0], [200.0, 100.0, 400.0, 300.0]]
        confs = [0.92, 0.85]
        cls_ids = [0, 0]
        fake_model.return_value = [
            _make_yolo_result(boxes, confs, cls_ids),
        ]
        
        detector.set_model(fake_model)
        img = Image.new("RGB", (640, 640))
        result = detector.detect(img)
        
        assert result.panel_count == 2
        assert result.best_confidence == pytest.approx(0.92)
        assert result.detection_successful is True
        assert len(result.boxes) == 2
        assert len(result.confidences) == 2
        assert len(result.class_ids) == 2

    def test_detect_multiple_predictions_in_raw_output(self):
        """YOLO may return multiple prediction objects; all should be merged."""
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        
        pred1 = _make_yolo_result([[10.0, 20.0, 100.0, 200.0]], [0.92], [0])
        pred2 = _make_yolo_result([[200.0, 100.0, 400.0, 300.0]], [0.85], [0])
        fake_model.return_value = [pred1, pred2]
        
        detector.set_model(fake_model)
        img = Image.new("RGB", (640, 640))
        result = detector.detect(img)
        
        assert result.panel_count == 2
        assert result.best_confidence == pytest.approx(0.92)
        assert len(result.boxes) == 2


# ---------------------------------------------------------------------------
# D. Configuration behavior
# ---------------------------------------------------------------------------

class TestConfigurationBehavior:
    """detect() uses config values for YOLO parameters."""

    def test_detect_uses_configured_confidence_threshold(self, project_config):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        fake_model.return_value = [_make_yolo_result()]
        detector.set_model(fake_model)
        
        img = Image.new("RGB", (640, 640))
        detector.detect(img)
        
        expected_conf = float(project_config["models"]["yolo"]["confidence_threshold"])
        fake_model.assert_called_once()
        call_kwargs = fake_model.call_args[1]
        assert call_kwargs["conf"] == pytest.approx(expected_conf)

    def test_detect_uses_configured_iou_threshold(self, project_config):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        fake_model.return_value = [_make_yolo_result()]
        detector.set_model(fake_model)
        
        img = Image.new("RGB", (640, 640))
        detector.detect(img)
        
        expected_iou = float(project_config["models"]["yolo"]["iou_threshold"])
        call_kwargs = fake_model.call_args[1]
        assert call_kwargs["iou"] == pytest.approx(expected_iou)

    def test_detect_uses_configured_image_size(self, project_config):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        fake_model.return_value = [_make_yolo_result()]
        detector.set_model(fake_model)
        
        img = Image.new("RGB", (640, 640))
        detector.detect(img)
        
        expected_size = int(project_config["models"]["yolo"]["image_size"])
        call_kwargs = fake_model.call_args[1]
        assert call_kwargs["imgsz"] == expected_size

    def test_detect_passes_verbose_false(self):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        fake_model.return_value = [_make_yolo_result()]
        detector.set_model(fake_model)
        
        img = Image.new("RGB", (640, 640))
        detector.detect(img)
        
        call_kwargs = fake_model.call_args[1]
        assert call_kwargs["verbose"] is False


# ---------------------------------------------------------------------------
# E. Logging behavior
# ---------------------------------------------------------------------------

class TestLoggingBehavior:
    """detect() emits expected log messages."""

    def test_info_log_before_inference(self, caplog):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        fake_model.return_value = [_make_yolo_result()]
        detector.set_model(fake_model)
        
        img = Image.new("RGB", (640, 640))
        import logging
        with caplog.at_level(logging.INFO):
            detector.detect(img)
        
        assert "Running YOLO inference" in caplog.text

    def test_info_log_after_successful_detection(self, caplog):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        boxes = [[10.0, 20.0, 100.0, 200.0]]
        confs = [0.92]
        cls_ids = [0]
        fake_model.return_value = [_make_yolo_result(boxes, confs, cls_ids)]
        detector.set_model(fake_model)
        
        img = Image.new("RGB", (640, 640))
        import logging
        with caplog.at_level(logging.INFO):
            detector.detect(img)
        
        assert "Detection complete" in caplog.text
        assert "1 panel(s) detected" in caplog.text


# ---------------------------------------------------------------------------
# F. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Repeated detections with same mock produce equivalent results."""

    def test_repeated_detections_are_equivalent(self):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        boxes = [[10.0, 20.0, 100.0, 200.0]]
        confs = [0.92]
        cls_ids = [0]
        fake_model.return_value = [_make_yolo_result(boxes, confs, cls_ids)]
        detector.set_model(fake_model)
        
        img = Image.new("RGB", (640, 640))
        r1 = detector.detect(img)
        r2 = detector.detect(img)
        
        assert r1.panel_count == r2.panel_count
        assert r1.best_confidence == r2.best_confidence
        assert r1.detection_successful == r2.detection_successful
        assert r1.boxes == r2.boxes
        assert fake_model.call_count == 2

    def test_model_called_once_per_detect(self):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        fake_model.return_value = [_make_yolo_result()]
        detector.set_model(fake_model)
        
        img = Image.new("RGB", (640, 640))
        detector.detect(img)
        detector.detect(img)
        
        assert fake_model.call_count == 2


# ---------------------------------------------------------------------------
# G. Malformed model output validation
# ---------------------------------------------------------------------------

class TestMalformedOutputValidation:
    """Detector rejects non-finite or out-of-range model outputs."""

    def test_nan_confidence_raises_prediction_error(self):
        import math
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        boxes = [[10.0, 20.0, 100.0, 200.0]]
        confs = [float("nan")]
        cls_ids = [0]
        fake_model.return_value = [_make_yolo_result(boxes, confs, cls_ids)]
        detector.set_model(fake_model)

        img = Image.new("RGB", (640, 640))
        with pytest.raises(PredictionError, match="non-finite"):
            detector.detect(img)

    def test_inf_confidence_raises_prediction_error(self):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        boxes = [[10.0, 20.0, 100.0, 200.0]]
        confs = [float("inf")]
        cls_ids = [0]
        fake_model.return_value = [_make_yolo_result(boxes, confs, cls_ids)]
        detector.set_model(fake_model)

        img = Image.new("RGB", (640, 640))
        with pytest.raises(PredictionError, match="non-finite"):
            detector.detect(img)

    def test_negative_confidence_raises_prediction_error(self):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        boxes = [[10.0, 20.0, 100.0, 200.0]]
        confs = [-0.1]
        cls_ids = [0]
        fake_model.return_value = [_make_yolo_result(boxes, confs, cls_ids)]
        detector.set_model(fake_model)

        img = Image.new("RGB", (640, 640))
        with pytest.raises(PredictionError, match="out of range"):
            detector.detect(img)

    def test_confidence_above_one_raises_prediction_error(self):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        boxes = [[10.0, 20.0, 100.0, 200.0]]
        confs = [1.1]
        cls_ids = [0]
        fake_model.return_value = [_make_yolo_result(boxes, confs, cls_ids)]
        detector.set_model(fake_model)

        img = Image.new("RGB", (640, 640))
        with pytest.raises(PredictionError, match="out of range"):
            detector.detect(img)

    def test_nan_box_coordinate_raises_prediction_error(self):
        detector = SolarPanelDetector()
        fake_model = _make_mock_yolo_model()
        boxes = [[float("nan"), 20.0, 100.0, 200.0]]
        confs = [0.9]
        cls_ids = [0]
        fake_model.return_value = [_make_yolo_result(boxes, confs, cls_ids)]
        detector.set_model(fake_model)

        img = Image.new("RGB", (640, 640))
        with pytest.raises(PredictionError, match="non-finite"):
            detector.detect(img)