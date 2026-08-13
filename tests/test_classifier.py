"""tests/test_classifier.py - Deterministic tests for models/classifier.py.

Covers the MobileNet classification wrapper:

A. Initialization and device selection
B. Model injection
C. Missing model handling
D. Successful classification
E. Label and confidence handling
F. Configuration-driven labels
G. Inference exception handling
H. Logging behavior
I. Determinism

Design rules honoured:
- no real MobileNet weights
- no network access
- no GPU required
- deterministic mocks for torch/torchvision
- existing conftest fixtures preferred
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import pytest
from PIL import Image

from models.classifier import ClassificationResult, SolarFaultClassifier
from utils.config import CFG
from utils.exceptions import ModelLoadError, PredictionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(has_params=True, device=None):
    """Create a mock torch.nn.Module with optional parameters."""
    model = MagicMock()
    if has_params:
        param = MagicMock()
        param.device = device or torch.device("cpu")
        model.parameters.return_value = iter([param])
    else:
        model.parameters.return_value = iter([])
    return model


@contextmanager
def _classification_mocks(model, probs_array):
    """Context manager that sets up model and softmax mocks for a classification test.
    
    Usage:
        with _classification_mocks(model, probs):
            result = clf.classify(img)
    """
    import models.classifier as classifier_module
    
    mock_logits = MagicMock()
    mock_probs = MagicMock()
    
    # Model forward returns logits
    model.return_value = mock_logits
    
    # Patch softmax in the classifier module's namespace
    patcher = patch.object(classifier_module.F, "softmax", return_value=mock_probs)
    mock_softmax = patcher.start()
    mock_softmax.return_value = mock_probs
    
    # probs.squeeze(0).cpu().numpy() returns numpy array
    mock_probs.squeeze.return_value.cpu.return_value.numpy.return_value = probs_array
    
    try:
        yield
    finally:
        patcher.stop()


# ---------------------------------------------------------------------------
# A. Initialization and device selection
# ---------------------------------------------------------------------------

class TestInitializationAndDevice:
    """SolarFaultClassifier initialization behavior."""

    def test_fresh_classifier_has_no_model(self):
        clf = SolarFaultClassifier()
        assert clf._model is None

    def test_default_device_is_cpu_when_cuda_unavailable(self, monkeypatch):
        import models.classifier as classifier_module
        
        fake_torch = MagicMock()
        fake_torch.device = MagicMock()
        fake_torch.cuda = MagicMock()
        fake_torch.cuda.is_available = MagicMock(return_value=False)
        monkeypatch.setattr(classifier_module, "torch", fake_torch)
        
        clf = SolarFaultClassifier()
        assert fake_torch.device.call_args[0][0] == "cpu"

    def test_device_set_from_model_parameters(self, monkeypatch):
        fake_device = MagicMock()
        model = _make_mock_model(has_params=True, device=fake_device)
        
        clf = SolarFaultClassifier()
        clf.set_model(model)
        
        assert clf._device is fake_device

    def test_device_unchanged_when_model_has_no_parameters(self, monkeypatch):
        import models.classifier as classifier_module
        
        fake_torch = MagicMock()
        fake_torch.device = MagicMock(return_value="cpu_device")
        fake_torch.cuda = MagicMock()
        fake_torch.cuda.is_available = MagicMock(return_value=False)
        monkeypatch.setattr(classifier_module, "torch", fake_torch)
        
        model = _make_mock_model(has_params=False)
        clf = SolarFaultClassifier()
        clf.set_model(model)
        
        assert clf._device == "cpu_device"


# ---------------------------------------------------------------------------
# B. Model injection
# ---------------------------------------------------------------------------

class TestModelInjection:
    """set_model() behavior."""

    def test_set_model_stores_model(self):
        clf = SolarFaultClassifier()
        model = _make_mock_model()
        clf.set_model(model)
        assert clf._model is model

    def test_set_model_none_raises_model_load_error(self):
        clf = SolarFaultClassifier()
        with pytest.raises(ModelLoadError, match="MobileNet"):
            clf.set_model(None)

    def test_set_model_none_does_not_corrupt_state(self):
        clf = SolarFaultClassifier()
        with pytest.raises(ModelLoadError):
            clf.set_model(None)
        assert clf._model is None


# ---------------------------------------------------------------------------
# C. Missing model handling
# ---------------------------------------------------------------------------

class TestMissingModelHandling:
    """classify() validates model is set before inference."""

    def test_classify_before_set_model_raises_model_load_error(self):
        clf = SolarFaultClassifier()
        img = Image.new("RGB", (224, 224))
        with pytest.raises(ModelLoadError, match="MobileNet"):
            clf.classify(img)


# ---------------------------------------------------------------------------
# D. Successful classification
# ---------------------------------------------------------------------------

class TestSuccessfulClassification:
    """classify() returns ClassificationResult with correct structure."""

    def test_classify_returns_classification_result(self):
        model = _make_mock_model()
        clf = SolarFaultClassifier()
        clf.set_model(model)
        
        probs = np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        with _classification_mocks(model, probs):
            img = Image.new("RGB", (224, 224))
            result = clf.classify(img)
        
        assert isinstance(result, ClassificationResult)

    def test_classify_sets_successful_flag(self):
        model = _make_mock_model()
        clf = SolarFaultClassifier()
        clf.set_model(model)
        
        probs = np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        with _classification_mocks(model, probs):
            img = Image.new("RGB", (224, 224))
            result = clf.classify(img)
        
        assert result.classification_successful is True

    def test_classify_identifies_highest_probability_class(self):
        model = _make_mock_model()
        clf = SolarFaultClassifier()
        clf.set_model(model)
        
        probs = np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        with _classification_mocks(model, probs):
            img = Image.new("RGB", (224, 224))
            result = clf.classify(img)
        
        assert result.class_id == 1
        assert result.label == "Dusty"

    def test_classify_confidence_matches_predicted_class(self):
        model = _make_mock_model()
        clf = SolarFaultClassifier()
        clf.set_model(model)
        
        probs = np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        with _classification_mocks(model, probs):
            img = Image.new("RGB", (224, 224))
            result = clf.classify(img)
        
        assert result.confidence == pytest.approx(0.7)

    def test_classify_probabilities_sum_to_one(self):
        model = _make_mock_model()
        clf = SolarFaultClassifier()
        clf.set_model(model)
        
        probs = np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        with _classification_mocks(model, probs):
            img = Image.new("RGB", (224, 224))
            result = clf.classify(img)
        
        assert pytest.approx(sum(result.probabilities.values())) == 1.0

    def test_classify_all_labels_present_in_probabilities(self):
        model = _make_mock_model()
        clf = SolarFaultClassifier()
        clf.set_model(model)
        
        probs = np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        with _classification_mocks(model, probs):
            img = Image.new("RGB", (224, 224))
            result = clf.classify(img)
        
        expected_labels = CFG["classification"]["labels"]
        assert set(result.probabilities.keys()) == set(expected_labels)


# ---------------------------------------------------------------------------
# E. Label and confidence handling
# ---------------------------------------------------------------------------

class TestLabelAndConfidenceHandling:
    """Label mapping and confidence edge cases."""

    def test_classify_uses_configured_labels(self, project_config):
        model = _make_mock_model()
        clf = SolarFaultClassifier()
        clf.set_model(model)
        
        probs = np.array([0.9, 0.05, 0.01, 0.01, 0.02, 0.01], dtype=np.float32)
        with _classification_mocks(model, probs):
            img = Image.new("RGB", (224, 224))
            result = clf.classify(img)
        
        expected_labels = project_config["classification"]["labels"]
        assert result.label == expected_labels[0]

    def test_classify_different_class_has_different_confidence(self):
        model = _make_mock_model()
        clf = SolarFaultClassifier()
        clf.set_model(model)
        
        probs = np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        with _classification_mocks(model, probs):
            img = Image.new("RGB", (224, 224))
            result = clf.classify(img)
        
        assert result.label == "Dusty"
        assert result.confidence == pytest.approx(0.7)
        assert result.class_id == 1


# ---------------------------------------------------------------------------
# F. Inference exception handling
# ---------------------------------------------------------------------------

class TestInferenceExceptionHandling:
    """Inference errors are wrapped in PredictionError."""

    def test_model_forward_exception_raises_prediction_error(self):
        model = _make_mock_model()
        model.side_effect = RuntimeError("CUDA out of memory")
        
        clf = SolarFaultClassifier()
        clf.set_model(model)
        
        img = Image.new("RGB", (224, 224))
        with pytest.raises(PredictionError, match="MobileNet") as exc_info:
            clf.classify(img)
        
        assert "CUDA out of memory" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

    def test_softmax_exception_raises_prediction_error(self):
        import models.classifier as classifier_module
        
        model = _make_mock_model()
        clf = SolarFaultClassifier()
        clf.set_model(model)
        
        # Make model return valid logits but softmax fail
        mock_logits = MagicMock()
        model.return_value = mock_logits
        
        with patch.object(classifier_module.F, "softmax", side_effect=RuntimeError("softmax failed")):
            img = Image.new("RGB", (224, 224))
            with pytest.raises(PredictionError, match="MobileNet"):
                clf.classify(img)


# ---------------------------------------------------------------------------
# G. Logging behavior
# ---------------------------------------------------------------------------

class TestLoggingBehavior:
    """classify() emits expected log messages."""

    def test_info_log_before_inference(self, caplog):
        model = _make_mock_model()
        clf = SolarFaultClassifier()
        clf.set_model(model)
        
        probs = np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        with _classification_mocks(model, probs):
            img = Image.new("RGB", (224, 224))
            import logging
            with caplog.at_level(logging.INFO):
                clf.classify(img)
        
        assert "Running MobileNet classification inference" in caplog.text

    def test_info_log_after_successful_classification(self, caplog):
        model = _make_mock_model()
        clf = SolarFaultClassifier()
        clf.set_model(model)
        
        probs = np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        with _classification_mocks(model, probs):
            img = Image.new("RGB", (224, 224))
            import logging
            with caplog.at_level(logging.INFO):
                clf.classify(img)
        
        assert "Classification complete" in caplog.text
        assert "Dusty" in caplog.text
        assert "confidence=" in caplog.text


# ---------------------------------------------------------------------------
# H. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Repeated classifications with same mock produce equivalent results."""

    def test_repeated_classifications_are_equivalent(self):
        model = _make_mock_model()
        clf = SolarFaultClassifier()
        clf.set_model(model)
        
        probs = np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        with _classification_mocks(model, probs):
            img = Image.new("RGB", (224, 224))
            r1 = clf.classify(img)
        
        with _classification_mocks(model, probs):
            img = Image.new("RGB", (224, 224))
            r2 = clf.classify(img)
        
        assert r1.label == r2.label
        assert r1.class_id == r2.class_id
        assert r1.confidence == r2.confidence
        assert r1.probabilities == r2.probabilities
        assert model.call_count == 2


# ---------------------------------------------------------------------------
# I. Configuration behavior
# ---------------------------------------------------------------------------

class TestConfigurationBehavior:
    """Classifier uses config for labels."""

    def test_labels_from_config(self, project_config):
        model = _make_mock_model()
        clf = SolarFaultClassifier()
        clf.set_model(model)
        
        probs = np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        with _classification_mocks(model, probs):
            img = Image.new("RGB", (224, 224))
            result = clf.classify(img)
        
        expected_labels = project_config["classification"]["labels"]
        assert set(result.probabilities.keys()) == set(expected_labels)


# ---------------------------------------------------------------------------
# J. Malformed model output validation
# ---------------------------------------------------------------------------

class TestMalformedOutputValidation:
    """Classifier rejects non-finite or out-of-range model outputs."""

    def test_nan_probability_raises_prediction_error(self):
        model = _make_mock_model()
        clf = SolarFaultClassifier()
        clf.set_model(model)

        probs = np.array([0.1, float("nan"), 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        with _classification_mocks(model, probs):
            img = Image.new("RGB", (224, 224))
            with pytest.raises(PredictionError, match="non-finite"):
                clf.classify(img)

    def test_inf_probability_raises_prediction_error(self):
        model = _make_mock_model()
        clf = SolarFaultClassifier()
        clf.set_model(model)

        probs = np.array([0.1, float("inf"), 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        with _classification_mocks(model, probs):
            img = Image.new("RGB", (224, 224))
            with pytest.raises(PredictionError, match="non-finite"):
                clf.classify(img)

    def test_negative_probability_raises_prediction_error(self):
        model = _make_mock_model()
        clf = SolarFaultClassifier()
        clf.set_model(model)

        probs = np.array([-0.1, 0.9, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        with _classification_mocks(model, probs):
            img = Image.new("RGB", (224, 224))
            with pytest.raises(PredictionError, match="negative values"):
                clf.classify(img)