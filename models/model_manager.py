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
:data:`model_manager` instance::

    from models.model_manager import model_manager

    detector   = model_manager.get_detector()
    classifier = model_manager.get_classifier()
    predictor  = model_manager.get_predictor()

Streamlit caching
-----------------
Because the ``ModelManager`` is a module-level singleton, Streamlit's
process-level reuse means models survive widget interactions without the
need for ``@st.cache_resource`` on individual model calls.  If explicit
Streamlit cache decoration is ever needed, wrap ``model_manager`` with
``@st.cache_resource`` in ``app.py`` (no changes required here).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils.config import CFG
from utils.exceptions import ModelLoadError
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pull paths from config once
# ---------------------------------------------------------------------------
_YOLO_WEIGHTS: Path = Path(CFG["models"]["yolo"]["weights"])
_MN_WEIGHTS: Path = Path(CFG["models"]["mobilenet"]["weights"])
_XGB_WEIGHTS: Path = Path(CFG["models"]["xgboost"]["weights"])


# ---------------------------------------------------------------------------
# Type aliases (avoid importing heavy libraries at module level)
# ---------------------------------------------------------------------------
_YOLOModel = object          # ultralytics.YOLO
_MobileNetModel = object     # torch.nn.Module
_XGBPipeline = object        # sklearn / joblib pipeline


class ModelManager:
    """Manages loading and caching of all three AI models.

    Models are loaded **lazily** on first access and **cached** thereafter.
    This avoids any heavyweight import at module-import time while still
    guaranteeing each model is instantiated only once per process.

    Attributes:
        _detector: Cached YOLO model (or ``None`` before first load).
        _classifier: Cached MobileNetV2 module (or ``None``).
        _predictor: Cached XGBoost pipeline (or ``None``).
    """

    def __init__(self) -> None:
        self._detector: Optional[_YOLOModel] = None
        self._classifier: Optional[_MobileNetModel] = None
        self._predictor: Optional[_XGBPipeline] = None
        self._device: Optional[object] = None  # torch.device, resolved on first use

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # YOLO
    # ------------------------------------------------------------------

    def _load_detector(self) -> None:
        """Load YOLO model from disk into ``self._detector``."""
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ModelLoadError(
                "YOLO",
                "The 'ultralytics' package is not installed. "
                "Run: pip install ultralytics",
            ) from exc

        if not _YOLO_WEIGHTS.exists():
            raise ModelLoadError(
                "YOLO",
                f"Weights not found at {_YOLO_WEIGHTS.resolve()}. "
                "Update 'models.yolo.weights' in configs/settings.yaml.",
            )

        self._detector = YOLO(str(_YOLO_WEIGHTS))
        logger.info("YOLO model loaded from %s.", _YOLO_WEIGHTS)

    def get_detector(self) -> _YOLOModel:
        """Return the cached YOLO model, loading it on the first call.

        Returns:
            Loaded ``ultralytics.YOLO`` instance.

        Raises:
            ModelLoadError: If the model file is missing or loading fails.
        """
        if self._detector is None:
            self._load_detector()
        return self._detector  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # MobileNet
    # ------------------------------------------------------------------

    def _load_classifier(self) -> None:
        """Load fine-tuned MobileNetV2 into ``self._classifier``."""
        try:
            import torch  # type: ignore
            from torchvision import models, transforms  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise ModelLoadError(
                "MobileNet",
                "torch / torchvision is not installed. "
                "Run: pip install torch torchvision",
            ) from exc

        if not _MN_WEIGHTS.exists():
            raise ModelLoadError(
                "MobileNet",
                f"Weights not found at {_MN_WEIGHTS.resolve()}. "
                "Update 'models.mobilenet.weights' in configs/settings.yaml.",
            )

        device = self._resolve_device()
        num_classes: int = int(CFG["models"]["mobilenet"]["num_classes"])

        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)

        state_dict = torch.load(
            str(_MN_WEIGHTS),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        self._classifier = model
        logger.info(
            "MobileNetV2 classifier loaded from %s (device=%s).",
            _MN_WEIGHTS,
            device,
        )

    def get_classifier(self) -> _MobileNetModel:
        """Return the cached MobileNetV2 model, loading it on first call.

        Returns:
            Loaded ``torch.nn.Module`` in eval mode on the resolved device.

        Raises:
            ModelLoadError: If the model file is missing or loading fails.
        """
        if self._classifier is None:
            self._load_classifier()
        return self._classifier  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------

    def _load_predictor(self) -> None:
        """Load the XGBoost joblib pipeline into ``self._predictor``."""
        try:
            import joblib  # type: ignore
        except ImportError as exc:
            raise ModelLoadError(
                "XGBoost",
                "joblib is not installed. Run: pip install joblib",
            ) from exc

        if not _XGB_WEIGHTS.exists():
            raise ModelLoadError(
                "XGBoost",
                f"Pipeline not found at {_XGB_WEIGHTS.resolve()}. "
                "Update 'models.xgboost.weights' in configs/settings.yaml.",
            )

        self._predictor = joblib.load(str(_XGB_WEIGHTS))
        logger.info("XGBoost pipeline loaded from %s.", _XGB_WEIGHTS)

    def get_predictor(self) -> _XGBPipeline:
        """Return the cached XGBoost pipeline, loading it on first call.

        Returns:
            Loaded joblib pipeline object with a ``.predict()`` method.

        Raises:
            ModelLoadError: If the model file is missing or loading fails.
        """
        if self._predictor is None:
            self._load_predictor()
        return self._predictor  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def preload_all(self) -> None:
        """Eagerly load all three models in sequence.

        Call this at application startup if you want to front-load the
        latency rather than paying it on the first inference request.

        Raises:
            ModelLoadError: If any model fails to load.
        """
        logger.info("ModelManager: pre-loading all models…")
        self.get_detector()
        self.get_classifier()
        self.get_predictor()
        logger.info("ModelManager: all models ready.")

    @property
    def loaded_models(self) -> dict[str, bool]:
        """Report which models are currently loaded.

        Returns:
            Dictionary mapping model name to load status.
        """
        return {
            "YOLO": self._detector is not None,
            "MobileNet": self._classifier is not None,
            "XGBoost": self._predictor is not None,
        }


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere instead of the class.
# ---------------------------------------------------------------------------
model_manager: ModelManager = ModelManager()
