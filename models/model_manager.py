"""models/model_manager.py — Centralised AI model lifecycle manager.

Responsibility
--------------
- Load every AI model **exactly once** per process.
- Cache each loaded model internally so repeated calls to getter methods
  return the same object without re-loading from disk.
- Expose simple typed getter methods consumed by :mod:`services.pipeline`.
- Raise :class:`~utils.exceptions.ModelLoadError` on any loading failure so
  callers receive a meaningful, typed error.

Pipeline integration
--------------------
``services/pipeline.py`` obtains models exclusively through the singleton
:data:`model_manager` instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils.config import CFG
from utils.exceptions import ModelLoadError
from utils.logger import get_logger

logger = get_logger(__name__)

_YOLO_WEIGHTS: Path = Path(CFG["models"]["yolo"]["weights"])
_MN_WEIGHTS: Path = Path(CFG["models"]["mobilenet"]["weights"])
_XGB_WEIGHTS: Path = Path(CFG["models"]["xgboost"]["weights"])

_YOLOModel = object
_MobileNetModel = object
_XGBPipeline = object


class ModelManager:
    """Manages loading and caching of all three AI models.

    Models are lazy-loaded and cached. Missing artifacts are reported through
    ``ModelLoadError`` rather than silently substituted with fabricated models.
    """

    def __init__(self) -> None:
        self._detector: Optional[_YOLOModel] = None
        self._classifier: Optional[_MobileNetModel] = None
        self._predictor: Optional[_XGBPipeline] = None
        self._device: Optional[object] = None

    def _resolve_device(self) -> object:
        """Resolve and cache the torch compute device (CUDA or CPU)."""
        if self._device is None:
            try:
                import torch  # type: ignore
                self._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                logger.info("Compute device resolved: %s", self._device)
            except ImportError as exc:
                raise ModelLoadError("torch", "torch is not installed") from exc
        return self._device

    def _load_detector(self) -> None:
        """Load YOLO model from disk into ``self._detector``."""
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ModelLoadError(
                "YOLO",
                "The 'ultralytics' package is not installed. Run: pip install ultralytics",
            ) from exc
        if not _YOLO_WEIGHTS.exists():
            raise ModelLoadError(
                "YOLO",
                f"Weights not found at {_YOLO_WEIGHTS.resolve()}. Update 'models.yolo.weights' in configs/settings.yaml.",
            )
        self._detector = YOLO(str(_YOLO_WEIGHTS))
        logger.info("YOLO model loaded from %s.", _YOLO_WEIGHTS)

    def get_detector(self) -> _YOLOModel:
        """Return the cached YOLO model, loading it on first access."""
        if self._detector is None:
            self._load_detector()
        return self._detector  # type: ignore[return-value]

    def _load_classifier(self) -> None:
        """Load fine-tuned MobileNetV2 into ``self._classifier``."""
        try:
            import torch  # type: ignore
            from torchvision import models  # type: ignore
        except ImportError as exc:
            raise ModelLoadError(
                "MobileNet",
                "torch / torchvision is not installed. Run: pip install torch torchvision",
            ) from exc
        if not _MN_WEIGHTS.exists():
            raise ModelLoadError(
                "MobileNet",
                f"Weights not found at {_MN_WEIGHTS.resolve()}. Update 'models.mobilenet.weights' in configs/settings.yaml.",
            )
        device = self._resolve_device()
        num_classes: int = int(CFG["models"]["mobilenet"]["num_classes"])
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
        state_dict = torch.load(str(_MN_WEIGHTS), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        self._classifier = model
        logger.info("MobileNetV2 classifier loaded from %s (device=%s).", _MN_WEIGHTS, device)

    def get_classifier(self) -> _MobileNetModel:
        """Return the cached MobileNetV2 model, loading it on first access."""
        if self._classifier is None:
            self._load_classifier()
        return self._classifier  # type: ignore[return-value]

    def _load_predictor(self) -> None:
        """Load the XGBoost joblib pipeline into ``self._predictor``."""
        try:
            import joblib  # type: ignore
        except ImportError as exc:
            raise ModelLoadError("XGBoost", "joblib is not installed. Run: pip install joblib") from exc
        if not _XGB_WEIGHTS.exists():
            raise ModelLoadError(
                "XGBoost",
                f"Pipeline not found at {_XGB_WEIGHTS.resolve()}. Update 'models.xgboost.weights' in configs/settings.yaml.",
            )
        self._predictor = joblib.load(str(_XGB_WEIGHTS))
        logger.info("XGBoost pipeline loaded from %s.", _XGB_WEIGHTS)

    def get_predictor(self) -> _XGBPipeline:
        """Return the cached XGBoost pipeline, loading it on first access."""
        if self._predictor is None:
            self._load_predictor()
        return self._predictor  # type: ignore[return-value]

    def preload_all(self) -> None:
        """Eagerly load all three models; raises on the first unavailable artifact."""
        logger.info("ModelManager: pre-loading all models…")
        self.get_detector()
        self.get_classifier()
        self.get_predictor()
        logger.info("ModelManager: all models ready.")

    @property
    def loaded_models(self) -> dict[str, bool]:
        """Report which models are currently loaded."""
        return {
            "YOLO": self._detector is not None,
            "MobileNet": self._classifier is not None,
            "XGBoost": self._predictor is not None,
        }

    @property
    def artifact_status(self) -> dict[str, dict[str, object]]:
        """Report configured model artifact paths and whether each exists.

        This is intentionally diagnostic only: it never creates or substitutes
        model files. It allows the UI and tests to distinguish a configured,
        ready artifact from a missing externally supplied weight file.
        """
        artifacts = {
            "YOLO": _YOLO_WEIGHTS,
            "MobileNet": _MN_WEIGHTS,
            "XGBoost": _XGB_WEIGHTS,
        }
        return {
            name: {"path": str(path), "exists": path.is_file()}
            for name, path in artifacts.items()
        }


model_manager: ModelManager = ModelManager()
