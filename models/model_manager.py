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
# The eventual full six-class artifact - not yet trained, see
# configs/settings.yaml's comment on models.mobilenet.weights.
_MN_SIX_CLASS_WEIGHTS: Path = Path(CFG["models"]["mobilenet"]["weights"])
_XGB_WEIGHTS: Path = Path(CFG["models"]["xgboost"]["weights"])

# v1 (frozen, current) MobileNet artifact - see configs/settings.yaml's
# comment on models.mobilenet.v1_weights. .get() with a default keeps this
# optional so a config predating this key still loads.
_MN_V1_WEIGHTS: Optional[Path] = (
    Path(CFG["models"]["mobilenet"]["v1_weights"])
    if CFG["models"]["mobilenet"].get("v1_weights") else None
)
_MN_V1_LABELS: list[str] = list(CFG["models"]["mobilenet"].get("v1_labels") or [])
_MN_SIX_CLASS_LABELS: list[str] = list(CFG["classification"]["labels"])


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
        self._classifier_source: Optional[str] = None  # "v1" | "six_class", set on load
        self._classifier_labels: Optional[list[str]] = None

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
                "Model weights not found at configured path.\n"
                "Expected artifact: weights/yolo_solar.pt\n"
                "Ensure model artifacts are installed. "
                "Update 'models.yolo.weights' in configs/settings.yaml if using custom paths.",
            )

        try:
            self._detector = YOLO(str(_YOLO_WEIGHTS))
        except Exception as exc:  # noqa: BLE001 - ultralytics can raise many real error types for a corrupt/incompatible checkpoint
            raise ModelLoadError(
                "YOLO",
                f"Weights file exists but failed to load as a YOLO checkpoint: {type(exc).__name__}: {exc}",
            ) from exc
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
        """Load MobileNetV2 into ``self._classifier``.

        Prefers the future six-class artifact (``weights/mobilenet_solar.pth``)
        the moment it exists. Until then - which is the normal, expected
        state for the v1 release - falls back to the real, frozen v1
        artifact (``weights/mobilenet_solar_v1.pth``, covering Clean/Dusty/
        Hotspot; Bird-Drop/Electrical-Damage/Physical-Damage remain a
        documented future expansion - see
        training/classification/DATASET_SOURCES.md). Either way,
        ``self._classifier_source`` and ``self._classifier_labels`` are set
        so callers (SolarFaultClassifier.set_model(), the UI) know exactly
        which class set is actually active - never silently assumed to be
        the full six."""
        try:
            import torch  # type: ignore
            from torchvision import models  # type: ignore
        except ImportError as exc:
            raise ModelLoadError(
                "MobileNet",
                "torch / torchvision is not installed. "
                "Run: pip install torch torchvision",
            ) from exc

        if _MN_SIX_CLASS_WEIGHTS.exists():
            weights_path = _MN_SIX_CLASS_WEIGHTS
            labels = _MN_SIX_CLASS_LABELS
            source = "six_class"
        elif _MN_V1_WEIGHTS is not None and _MN_V1_WEIGHTS.exists():
            if not _MN_V1_LABELS:
                # Configured-path label only, never the resolved Path object -
                # its str() could be an absolute filesystem path (see the
                # "no absolute path in error messages" security test this
                # mirrors below).
                raise ModelLoadError(
                    "MobileNet",
                    "v1 weights exist but 'models.mobilenet.v1_labels' is empty "
                    "in configs/settings.yaml - refusing to guess the class set.",
                )
            weights_path = _MN_V1_WEIGHTS
            labels = _MN_V1_LABELS
            source = "v1"
        else:
            # Static, configured-path strings only - never interpolate the
            # resolved _MN_SIX_CLASS_WEIGHTS/_MN_V1_WEIGHTS Path objects,
            # which may resolve to an absolute local filesystem path.
            raise ModelLoadError(
                "MobileNet",
                "Model weights not found at either configured path.\n"
                "Expected future six-class artifact: weights/mobilenet_solar.pth\n"
                "Expected v1 artifact: weights/mobilenet_solar_v1.pth\n"
                "Ensure model artifacts are installed. "
                "Update 'models.mobilenet.weights'/'v1_weights' in configs/settings.yaml if using custom paths.",
            )

        device = self._resolve_device()
        num_classes = len(labels)

        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)

        try:
            state_dict = torch.load(
                str(weights_path),
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)
        except Exception as exc:
            configured_label = "weights/mobilenet_solar.pth" if source == "six_class" else "weights/mobilenet_solar_v1.pth"
            raise ModelLoadError(
                "MobileNet",
                f"Failed to load the {source} checkpoint ({configured_label}) as a "
                f"{num_classes}-class MobileNetV2: {type(exc).__name__}: {exc}",
            ) from exc
        model.to(device)
        model.eval()

        self._classifier = model
        self._classifier_source = source
        self._classifier_labels = list(labels)
        logger.info(
            "MobileNetV2 classifier loaded from %s (source=%s, classes=%s, device=%s).",
            weights_path, source, labels, device,
        )
        if source == "v1":
            logger.info(
                "MobileNet running the v1 release classifier (%s) - %s. "
                "Bird-Drop/Electrical-Damage/Physical-Damage remain a documented future "
                "expansion (see training/classification/DATASET_SOURCES.md); the future "
                "six-class artifact (weights/mobilenet_solar.pth) is not present.",
                weights_path, labels,
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

    @property
    def classifier_labels(self) -> list[str]:
        """The class label list matching the currently-loaded classifier's
        output layer, in index order - loads the classifier first if not
        already loaded. Always pass this to
        ``SolarFaultClassifier.set_model(model, labels=...)`` rather than
        assuming the full six-class label list; see ``_load_classifier()``.

        Raises:
            ModelLoadError: If no classifier artifact (v1 or six-class) is
                available.
        """
        self.get_classifier()
        assert self._classifier_labels is not None  # set by _load_classifier on success
        return self._classifier_labels

    @property
    def classifier_source(self) -> Optional[str]:
        """``"v1"`` (the frozen release classifier - the normal, expected
        value), ``"six_class"`` (the future full artifact, once trained), or
        ``None`` if the classifier has not been loaded yet. Does not trigger
        a load - check ``mobilenet_status`` for a load-free view."""
        return self._classifier_source

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
                "Pipeline not found at configured path.\n"
                "Expected artifact: weights/xgboost_solar.joblib\n"
                "Ensure model artifacts are installed. "
                "Update 'models.xgboost.weights' in configs/settings.yaml if using custom paths.",
            )

        try:
            self._predictor = joblib.load(str(_XGB_WEIGHTS))
        except Exception as exc:  # noqa: BLE001 - joblib/pickle can raise many real error types for a corrupt/incompatible artifact
            raise ModelLoadError(
                "XGBoost",
                f"Pipeline file exists but failed to load: {type(exc).__name__}: {exc}",
            ) from exc
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

    @property
    def artifact_status(self) -> dict[str, dict[str, object]]:
        """Report configured model artifact paths and whether each exists.

        This is diagnostic only. It never creates, downloads, or substitutes
        trained model artifacts.

        "MobileNet" reports ``exists=True`` when *either* the v1 artifact or
        the future six-class artifact is present - either one is a real,
        usable classifier - not only the (for v1, intentionally absent)
        six-class one. This is what scripts/check_runtime_readiness.py's
        inference-readiness verdict is built from: a v1 deployment running
        only the v1 3-class artifact must report ready, not falsely
        "missing MobileNet", since the v1 artifact fully satisfies the v1
        contract. See ``mobilenet_status`` for which artifact is actually
        active and its class set.
        """
        mobilenet_exists = _MN_SIX_CLASS_WEIGHTS.is_file() or (
            _MN_V1_WEIGHTS is not None and _MN_V1_WEIGHTS.is_file()
        )
        return {
            "YOLO": {"path": str(_YOLO_WEIGHTS), "exists": _YOLO_WEIGHTS.is_file()},
            "MobileNet": {"path": str(_MN_SIX_CLASS_WEIGHTS), "exists": mobilenet_exists},
            "XGBoost": {"path": str(_XGB_WEIGHTS), "exists": _XGB_WEIGHTS.is_file()},
        }

    @property
    def mobilenet_status(self) -> dict[str, object]:
        """Nuanced MobileNet capability report - never loads the model.

        Distinguishes "the future six-class artifact is missing" from "the
        real, frozen v1 artifact is genuinely active and usable" rather than
        collapsing both into a single ready/not_ready flag. Additive to
        ``artifact_status``, whose "MobileNet" entry reports readiness in
        the coarser sense described there.

        Returns:
            ``state``: ``"six_class"`` (the future full 6-class artifact
            exists), ``"v1"`` (the frozen v1 artifact is active - the
            normal, expected state for this release), or ``"missing"``
            (neither exists).
            ``active_labels``: the class list that would actually be used
            for inference right now, given ``state``.
            ``is_six_class_active``: whether ``active_labels`` equals the
            full 6-class future contract.
            Plus the raw path/existence of both the v1 and future six-class
            artifacts, for a status page to display in full.
        """
        six_class_exists = _MN_SIX_CLASS_WEIGHTS.is_file()
        v1_exists = _MN_V1_WEIGHTS is not None and _MN_V1_WEIGHTS.is_file()
        if six_class_exists:
            state, active_labels = "six_class", _MN_SIX_CLASS_LABELS
        elif v1_exists:
            state, active_labels = "v1", _MN_V1_LABELS
        else:
            state, active_labels = "missing", []
        return {
            "state": state,
            "active_labels": list(active_labels),
            "is_six_class_active": active_labels == _MN_SIX_CLASS_LABELS,
            "six_class_path": str(_MN_SIX_CLASS_WEIGHTS),
            "six_class_exists": six_class_exists,
            "v1_path": str(_MN_V1_WEIGHTS) if _MN_V1_WEIGHTS else None,
            "v1_exists": v1_exists,
            "six_class_labels": list(_MN_SIX_CLASS_LABELS),
            "v1_labels": list(_MN_V1_LABELS),
        }

    def verify_all(self) -> dict[str, dict[str, object]]:
        """Deep, on-demand verification: actually attempts to load every
        model (via the same cached get_*() methods the real pipeline uses -
        so if a model is already loaded this is free, not a second load),
        distinguishing a genuinely missing artifact from one that exists but
        fails to load (corrupt file, shape mismatch, incompatible torch
        build, etc.) - a distinction the cheap filesystem-only
        artifact_status/mobilenet_status properties cannot make.

        Not called automatically by any page on every render - this is
        deliberately opt-in (e.g. a "Run deep verification" button) so nothing
        pays the cost of a real model load just to display a status page.

        Returns one entry per model:
            {"state": "ready"|"v1"|"six_class"|"missing"|"error", "detail": str|None}
        "ready" is YOLO/XGBoost-only; MobileNet instead reports "v1" (the
        frozen release classifier is active - the normal, expected state)
        or "six_class" (the future full checkpoint is active).
        """
        report: dict[str, dict[str, object]] = {}

        try:
            self.get_detector()
            report["YOLO"] = {"state": "ready", "detail": None}
        except ModelLoadError as exc:
            state = "missing" if not _YOLO_WEIGHTS.is_file() else "error"
            report["YOLO"] = {"state": state, "detail": str(exc)}

        try:
            self.get_classifier()
            report["MobileNet"] = {"state": self._classifier_source or "ready", "detail": None}
        except ModelLoadError as exc:
            mn = self.mobilenet_status
            state = "missing" if mn["state"] == "missing" else "error"
            report["MobileNet"] = {"state": state, "detail": str(exc)}

        try:
            self.get_predictor()
            report["XGBoost"] = {"state": "ready", "detail": None}
        except ModelLoadError as exc:
            state = "missing" if not _XGB_WEIGHTS.is_file() else "error"
            report["XGBoost"] = {"state": state, "detail": str(exc)}

        return report


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere instead of the class.
# ---------------------------------------------------------------------------
model_manager: ModelManager = ModelManager()
