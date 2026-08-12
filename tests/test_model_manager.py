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
import builtins
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from models.model_manager import ModelManager, model_manager
from utils.config import CFG
from utils.exceptions import ModelLoadError


# ---------------------------------------------------------------------------
# Lightweight stubs for heavy optional dependencies
# ---------------------------------------------------------------------------

# NOTE: Module-level stub installation removed to prevent global sys.modules
# pollution that breaks order-independence. Every test in this file now sets
# up its own mocks explicitly via monkeypatch.setitem where required.


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


def _simulate_missing_torch(monkeypatch):
    """Simulate torch/torchvision being unavailable without re-importing real torch.

    Patches builtins.__import__ first, then removes torch/torchvision from
    sys.modules. This ensures that any subsequent `import torch` attempt hits
    the patched __import__ and raises ImportError, avoiding the TORCH_LIBRARY
    re-registration issue that occurs when the real torch package is deleted
    from sys.modules and then re-imported.
    """
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name in ("torch", "torchvision"):
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    # Patch __import__ BEFORE removing modules from sys.modules
    monkeypatch.setattr(builtins, "__import__", mock_import)

    # Remove torch-related modules so the import machinery attempts to load them
    # The patched __import__ will raise ImportError instead of loading real torch
    for mod_name in ("torch", "torch.nn", "torch.nn.functional", "torchvision", "torchvision.transforms"):
        monkeypatch.delitem(sys.modules, mod_name, raising=False)


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
        _simulate_missing_torch(monkeypatch)

        mm = ModelManager()
        monkeypatch.setattr("models.model_manager._MN_WEIGHTS", MagicMock(exists=MagicMock(return_value=True)))

        with pytest.raises(ModelLoadError, match="MobileNet"):
            mm.get_classifier()

    def test_missing_torch_in_resolve_device_raises_model_load_error(self, monkeypatch):
        _simulate_missing_torch(monkeypatch)

        mm = ModelManager()

        with pytest.raises(ModelLoadError, match="torch") as exc_info:
            mm._resolve_device()

        assert "torch is not installed" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert mm._device is None

    def test_resolve_device_failure_preserves_manager_state(self, monkeypatch):
        fake_torch = _make_fake_torch()

        # Simulate missing torch using __import__ patch to avoid re-importing
        # the real installed torch package, which can trigger TORCH_LIBRARY
        # registration errors.
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        monkeypatch.delitem(sys.modules, "torch.nn", raising=False)
        monkeypatch.delitem(sys.modules, "torch.nn.functional", raising=False)

        mm = ModelManager()
        with pytest.raises(ModelLoadError, match="torch"):
            mm._resolve_device()
        assert mm._device is None

        # Undo the missing-torch patches so we can restore torch safely.
        # This restores the original __import__ and sys.modules state.
        monkeypatch.undo()

        # Now install the fake torch for the retry
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "torch.nn", fake_torch.nn)
        monkeypatch.setitem(sys.modules, "torch.nn.functional", MagicMock())

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


# ---------------------------------------------------------------------------
# I. Model path resolution
# ---------------------------------------------------------------------------

class TestModelPathResolution:
    """Model weight paths are resolved relative to the project root."""

    def test_yolo_weights_path_resolves_from_project_root(self):
        from utils.config import _PROJECT_ROOT
        from models.model_manager import _YOLO_WEIGHTS

        expected = _PROJECT_ROOT / "weights/yolo_solar.pt"
        assert _YOLO_WEIGHTS == expected

    def test_mobilenet_weights_path_resolves_from_project_root(self):
        from utils.config import _PROJECT_ROOT
        from models.model_manager import _MN_WEIGHTS

        expected = _PROJECT_ROOT / "weights/mobilenet_solar.pth"
        assert _MN_WEIGHTS == expected

    def test_xgboost_weights_path_resolves_from_project_root(self):
        from utils.config import _PROJECT_ROOT
        from models.model_manager import _XGB_WEIGHTS

        expected = _PROJECT_ROOT / "weights/xgboost_solar.joblib"
        assert _XGB_WEIGHTS == expected

    def test_config_uses_relative_weight_paths(self):
        assert CFG["models"]["yolo"]["weights"] == "weights/yolo_solar.pt"
        assert CFG["models"]["mobilenet"]["weights"] == "weights/mobilenet_solar.pth"
        assert CFG["models"]["xgboost"]["weights"] == "weights/xgboost_solar.joblib"

    def test_detector_missing_weights_error_is_actionable(self, monkeypatch, tmp_path):
        fake_yolo = _make_fake_yolo()
        monkeypatch.setitem(sys.modules, "ultralytics", fake_yolo)

        mm = ModelManager()
        missing_path = tmp_path / "nonexistent" / "yolo_solar.pt"
        monkeypatch.setattr("models.model_manager._YOLO_WEIGHTS", missing_path)

        with pytest.raises(ModelLoadError, match="YOLO") as exc_info:
            mm.get_detector()

        msg = str(exc_info.value)
        assert "yolo_solar.pt" in msg
        assert "Ensure model artifacts" in msg

    def test_classifier_missing_weights_error_is_actionable(self, monkeypatch, tmp_path):
        fake_torch = _make_fake_torch()
        fake_models = MagicMock()
        fake_models.mobilenet_v2 = MagicMock(return_value=MagicMock())
        fake_torch.models = fake_models
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "torchvision", MagicMock(models=fake_models))

        mm = ModelManager()
        missing_path = tmp_path / "nonexistent" / "mobilenet_solar.pth"
        monkeypatch.setattr("models.model_manager._MN_WEIGHTS", missing_path)

        with pytest.raises(ModelLoadError, match="MobileNet") as exc_info:
            mm.get_classifier()

        msg = str(exc_info.value)
        assert "mobilenet_solar.pth" in msg
        assert "Ensure model artifacts" in msg

    def test_predictor_missing_weights_error_is_actionable(self, monkeypatch, tmp_path):
        fake_joblib = _make_fake_joblib()
        monkeypatch.setitem(sys.modules, "joblib", fake_joblib)

        mm = ModelManager()
        missing_path = tmp_path / "nonexistent" / "xgboost_solar.joblib"
        monkeypatch.setattr("models.model_manager._XGB_WEIGHTS", missing_path)

        with pytest.raises(ModelLoadError, match="XGBoost") as exc_info:
            mm.get_predictor()

        msg = str(exc_info.value)
        assert "xgboost_solar.joblib" in msg
        assert "Ensure model artifacts" in msg