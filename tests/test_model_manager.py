"""tests/test_model_manager.py - Deterministic tests for models/model_manager.py.

Covers the model lifecycle manager as a lazy-loading, caching gateway:

A. Singleton behavior
B. Lazy loading / caching
C. Device resolution
D. Missing dependency errors
E. Missing weight file errors
F. preload_all()
G. loaded_models property
H. Determinism

Design rules honoured:
- no real model weights
- no network access
- no GPU required
- deterministic mocks for torch/ultralytics/joblib
- existing conftest fixtures preferred
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from models.model_manager import ModelManager, model_manager
from utils.config import CFG
from utils.exceptions import ModelLoadError


# ---------------------------------------------------------------------------
# Lightweight stubs for heavy optional dependencies
# ---------------------------------------------------------------------------

def _install_stubs():
    """Install lightweight mocks for torch/ultralytics/joblib if missing."""
    if "torch" not in sys.modules:
        torch_stub = MagicMock()
        torch_stub.device = MagicMock()
        torch_stub.cuda = MagicMock()
        torch_stub.cuda.is_available = MagicMock(return_value=False)
        torch_stub.nn = MagicMock()
        torch_stub.nn.Linear = MagicMock()
        sys.modules["torch"] = torch_stub
        sys.modules["torch.nn"] = torch_stub.nn
        sys.modules["torch.nn.functional"] = MagicMock()
    if "ultralytics" not in sys.modules:
        sys.modules["ultralytics"] = MagicMock()
    if "joblib" not in sys.modules:
        sys.modules["joblib"] = MagicMock()


_install_stubs()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_torch():
    """Return a fake torch module with device/cuda/models."""
    t = MagicMock()
    t.device = MagicMock()
    t.cuda = MagicMock()
    t.cuda.is_available = MagicMock(return_value=False)
    t.nn = MagicMock()
    t.nn.Linear = MagicMock(return_value=MagicMock())
    return t


def _make_fake_yolo():
    return MagicMock()


def _make_fake_joblib():
    j = MagicMock()
    j.load = MagicMock(return_value=MagicMock())
    return j


# ---------------------------------------------------------------------------
# A. Singleton behavior
# ---------------------------------------------------------------------------

class TestSingletonBehavior:
    """Module-level model_manager is a shared singleton."""

    def test_module_level_singleton_exists(self):
        assert model_manager is not None
        assert isinstance(model_manager, ModelManager)

    def test_new_instance_is_independent(self):
        mm = ModelManager()
        assert mm is not model_manager
        assert isinstance(mm, ModelManager)


# ---------------------------------------------------------------------------
# B. Lazy loading / caching
# ---------------------------------------------------------------------------

class TestLazyLoadingAndCaching:
    """Models are loaded only on first access and cached thereafter."""

    def test_detector_loaded_on_first_get(self, monkeypatch):
        fake_yolo = _make_fake_yolo()
        monkeypatch.setitem(sys.modules, "ultralytics", fake_yolo)

        mm = ModelManager()
        # Simulate weights file existing
        monkeypatch.setattr("models.model_manager._YOLO_WEIGHTS", MagicMock(exists=MagicMock(return_value=True)))

        first = mm.get_detector()
        second = mm.get_detector()
        assert first is second
        assert fake_yolo.YOLO.called

    def test_classifier_loaded_on_first_get(self, monkeypatch):
        fake_torch = _make_fake_torch()
        fake_models = MagicMock()
        fake_models.mobilenet_v2 = MagicMock(return_value=MagicMock())
        fake_torch.models = fake_models
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "torchvision", MagicMock(models=fake_models))

        mm = ModelManager()
        monkeypatch.setattr("models.model_manager._MN_WEIGHTS", MagicMock(exists=MagicMock(return_value=True)))

        first = mm.get_classifier()
        second = mm.get_classifier()
        assert first is second
        assert fake_models.mobilenet_v2.called

    def test_predictor_loaded_on_first_get(self, monkeypatch):
        fake_joblib = _make_fake_joblib()
        monkeypatch.setitem(sys.modules, "joblib", fake_joblib)

        mm = ModelManager()
        monkeypatch.setattr("models.model_manager._XGB_WEIGHTS", MagicMock(exists=MagicMock(return_value=True)))

        first = mm.get_predictor()
        second = mm.get_predictor()
        assert first is second
        assert fake_joblib.load.called


# ---------------------------------------------------------------------------
# C. Device resolution
# ---------------------------------------------------------------------------

class TestDeviceResolution:
    """Torch device is resolved lazily and cached."""

    def test_cpu_when_cuda_unavailable(self, monkeypatch):
        fake_torch = _make_fake_torch()
        fake_torch.cuda.is_available.return_value = False
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        mm = ModelManager()
        device = mm._resolve_device()
        assert device == fake_torch.device.return_value
        assert fake_torch.device.call_args[0][0] == "cpu"

    def test_device_cached_after_first_resolve(self, monkeypatch):
        fake_torch = _make_fake_torch()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        mm = ModelManager()
        mm._resolve_device()
        mm._resolve_device()  # second call
        assert fake_torch.device.call_count == 1


# ---------------------------------------------------------------------------
# D. Missing dependency errors
# ---------------------------------------------------------------------------

class TestMissingDependencyErrors:
    """Missing heavy dependencies raise ModelLoadError with actionable messages."""

    def test_missing_torch_for_classifier(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.delitem(sys.modules, "torch", raising=False)

        mm = ModelManager()
        monkeypatch.setattr("models.model_manager._MN_WEIGHTS", MagicMock(exists=MagicMock(return_value=True)))

        with pytest.raises(ModelLoadError, match="MobileNet"):
            mm.get_classifier()

    def test_missing_torch_in_resolve_device_raises_model_load_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.delitem(sys.modules, "torch", raising=False)

        mm = ModelManager()

        with pytest.raises(ModelLoadError, match="torch") as exc_info:
            mm._resolve_device()

        assert "torch is not installed" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert mm._device is None

    def test_resolve_device_failure_preserves_manager_state(self, monkeypatch):
        fake_torch = _make_fake_torch()

        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.delitem(sys.modules, "torch", raising=False)

        mm = ModelManager()

        with pytest.raises(ModelLoadError, match="torch"):
            mm._resolve_device()

        assert mm._device is None

        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        device = mm._resolve_device()
        assert device == fake_torch.device.return_value
        assert mm._device is device

    def test_missing_ultralytics_for_detector(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ultralytics", None)

        mm = ModelManager()
        monkeypatch.setattr("models.model_manager._YOLO_WEIGHTS", MagicMock(exists=MagicMock(return_value=True)))

        with pytest.raises(ModelLoadError, match="YOLO"):
            mm.get_detector()

    def test_missing_joblib_for_predictor(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "joblib", None)

        mm = ModelManager()
        monkeypatch.setattr("models.model_manager._XGB_WEIGHTS", MagicMock(exists=MagicMock(return_value=True)))

        with pytest.raises(ModelLoadError, match="XGBoost"):
            mm.get_predictor()


# ---------------------------------------------------------------------------
# E. Missing weight file errors
# ---------------------------------------------------------------------------

class TestMissingWeightFileErrors:
    """Missing weight files raise ModelLoadError with path information."""

    def test_detector_missing_weights(self, monkeypatch):
        fake_yolo = _make_fake_yolo()
        monkeypatch.setitem(sys.modules, "ultralytics", fake_yolo)

        mm = ModelManager()
        monkeypatch.setattr("models.model_manager._YOLO_WEIGHTS", MagicMock(exists=MagicMock(return_value=False)))

        with pytest.raises(ModelLoadError, match="YOLO"):
            mm.get_detector()

    def test_classifier_missing_weights(self, monkeypatch):
        fake_torch = _make_fake_torch()
        fake_models = MagicMock()
        fake_models.mobilenet_v2 = MagicMock(return_value=MagicMock())
        fake_torch.models = fake_models
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "torchvision", MagicMock(models=fake_models))

        mm = ModelManager()
        monkeypatch.setattr("models.model_manager._MN_WEIGHTS", MagicMock(exists=MagicMock(return_value=False)))

        with pytest.raises(ModelLoadError, match="MobileNet"):
            mm.get_classifier()

    def test_predictor_missing_weights(self, monkeypatch):
        fake_joblib = _make_fake_joblib()
        monkeypatch.setitem(sys.modules, "joblib", fake_joblib)

        mm = ModelManager()
        monkeypatch.setattr("models.model_manager._XGB_WEIGHTS", MagicMock(exists=MagicMock(return_value=False)))

        with pytest.raises(ModelLoadError, match="XGBoost"):
            mm.get_predictor()


# ---------------------------------------------------------------------------
# F. preload_all()
# ---------------------------------------------------------------------------

class TestPreloadAll:
    """preload_all() eagerly loads all three models in sequence."""

    def test_preload_all_loads_all_models(self, monkeypatch):
        fake_yolo = _make_fake_yolo()
        fake_torch = _make_fake_torch()
        fake_models = MagicMock()
        fake_models.mobilenet_v2 = MagicMock(return_value=MagicMock())
        fake_torch.models = fake_models
        fake_joblib = _make_fake_joblib()

        monkeypatch.setitem(sys.modules, "ultralytics", fake_yolo)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "torchvision", MagicMock(models=fake_models))
        monkeypatch.setitem(sys.modules, "joblib", fake_joblib)

        mm = ModelManager()
        monkeypatch.setattr("models.model_manager._YOLO_WEIGHTS", MagicMock(exists=MagicMock(return_value=True)))
        monkeypatch.setattr("models.model_manager._MN_WEIGHTS", MagicMock(exists=MagicMock(return_value=True)))
        monkeypatch.setattr("models.model_manager._XGB_WEIGHTS", MagicMock(exists=MagicMock(return_value=True)))

        mm.preload_all()

        assert fake_yolo.YOLO.called
        assert fake_models.mobilenet_v2.called
        assert fake_joblib.load.called

    def test_preload_all_propagates_first_error(self, monkeypatch):
        fake_yolo = _make_fake_yolo()
        fake_yolo.YOLO.side_effect = RuntimeError("bad weights")
        monkeypatch.setitem(sys.modules, "ultralytics", fake_yolo)

        mm = ModelManager()
        monkeypatch.setattr("models.model_manager._YOLO_WEIGHTS", MagicMock(exists=MagicMock(return_value=True)))

        with pytest.raises(RuntimeError, match="bad weights"):
            mm.preload_all()


# ---------------------------------------------------------------------------
# G. loaded_models property
# ---------------------------------------------------------------------------

class TestLoadedModelsProperty:
    """loaded_models reports which models are currently loaded."""

    def test_initial_state_all_unloaded(self):
        mm = ModelManager()
        assert mm.loaded_models == {"YOLO": False, "MobileNet": False, "XGBoost": False}

    def test_state_after_partial_load(self, monkeypatch):
        fake_yolo = _make_fake_yolo()
        monkeypatch.setitem(sys.modules, "ultralytics", fake_yolo)

        mm = ModelManager()
        monkeypatch.setattr("models.model_manager._YOLO_WEIGHTS", MagicMock(exists=MagicMock(return_value=True)))
        mm.get_detector()

        assert mm.loaded_models["YOLO"] is True
        assert mm.loaded_models["MobileNet"] is False
        assert mm.loaded_models["XGBoost"] is False

    def test_state_after_full_load(self, monkeypatch):
        fake_yolo = _make_fake_yolo()
        fake_torch = _make_fake_torch()
        fake_models = MagicMock()
        fake_models.mobilenet_v2 = MagicMock(return_value=MagicMock())
        fake_torch.models = fake_models
        fake_joblib = _make_fake_joblib()

        monkeypatch.setitem(sys.modules, "ultralytics", fake_yolo)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "torchvision", MagicMock(models=fake_models))
        monkeypatch.setitem(sys.modules, "joblib", fake_joblib)

        mm = ModelManager()
        monkeypatch.setattr("models.model_manager._YOLO_WEIGHTS", MagicMock(exists=MagicMock(return_value=True)))
        monkeypatch.setattr("models.model_manager._MN_WEIGHTS", MagicMock(exists=MagicMock(return_value=True)))
        monkeypatch.setattr("models.model_manager._XGB_WEIGHTS", MagicMock(exists=MagicMock(return_value=True)))

        mm.get_detector()
        mm.get_classifier()
        mm.get_predictor()

        assert mm.loaded_models == {"YOLO": True, "MobileNet": True, "XGBoost": True}


# ---------------------------------------------------------------------------
# H. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Repeated calls with same mocks produce equivalent results."""

    def test_repeated_get_detector_returns_same_object(self, monkeypatch):
        fake_yolo = _make_fake_yolo()
        monkeypatch.setitem(sys.modules, "ultralytics", fake_yolo)

        mm = ModelManager()
        monkeypatch.setattr("models.model_manager._YOLO_WEIGHTS", MagicMock(exists=MagicMock(return_value=True)))

        r1 = mm.get_detector()
        r2 = mm.get_detector()
        assert r1 is r2
        assert fake_yolo.YOLO.call_count == 1