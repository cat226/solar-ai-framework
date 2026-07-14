"""models/detector.py — YOLO-based solar panel detection.

Responsibility
--------------
- Wrap a YOLO model (supplied by :mod:`models.model_manager`) and run
  inference on an input image.
- Return structured detection results (bounding boxes, confidences, classes).

Model loading
-------------
This module does **not** load the YOLO model itself.  It receives the loaded
model object from :class:`models.model_manager.ModelManager` via
:meth:`SolarPanelDetector.set_model`.  This guarantees the model is loaded
exactly once, by a single owner, per process.

This module is purely concerned with object detection.  It has no knowledge
of the UI, weather data, or downstream prediction logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PIL import Image

from utils.config import CFG
from utils.exceptions import ModelLoadError, PredictionError
from utils.image_utils import pil_to_numpy, resize_for_yolo
from utils.logger import get_logger

logger = get_logger(__name__)

# Read inference hyper-parameters from config (weights path is for ModelManager)
_YOLO_CFG: dict = CFG["models"]["yolo"]
_CONF_THRESH: float = float(_YOLO_CFG["confidence_threshold"])
_IOU_THRESH: float = float(_YOLO_CFG["iou_threshold"])
_IMG_SIZE: int = int(_YOLO_CFG["image_size"])


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Encapsulates the output of YOLO inference on a single image.

    Attributes:
        boxes: List of bounding boxes, each as [x1, y1, x2, y2] in pixels.
        confidences: Confidence score for each detected box (0.0–1.0).
        class_ids: Class index for each detected box.
        panel_count: Total number of panels detected.
        best_confidence: Highest confidence score among all detections.
                         0.0 if nothing was detected.
        detection_successful: True if at least one panel was detected.
    """

    boxes: List[List[float]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    class_ids: List[int] = field(default_factory=list)
    panel_count: int = 0
    best_confidence: float = 0.0
    detection_successful: bool = False


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class SolarPanelDetector:
    """Wraps a YOLO model to detect solar panels in images.

    The model object is injected via :meth:`set_model` (called by
    :mod:`services.pipeline` after retrieving it from
    :class:`models.model_manager.ModelManager`).

    Usage via pipeline (preferred)::

        from models.model_manager import model_manager
        detector = SolarPanelDetector()
        detector.set_model(model_manager.get_detector())
        result = detector.detect(pil_image)
    """

    def __init__(self) -> None:
        self._model: Optional[object] = None

    def set_model(self, model: object) -> None:
        """Inject the loaded YOLO model.

        Args:
            model: A loaded ``ultralytics.YOLO`` instance obtained from
                   :class:`~models.model_manager.ModelManager`.

        Raises:
            ModelLoadError: If *model* is ``None``.
        """
        if model is None:
            raise ModelLoadError("YOLO", "set_model() received None — check ModelManager.")
        self._model = model
        logger.debug("SolarPanelDetector: model injected.")

    def detect(self, image: Image.Image) -> DetectionResult:
        """Run YOLO inference on a PIL image and return a DetectionResult.

        Args:
            image: Input image (RGB PIL Image).

        Returns:
            :class:`DetectionResult` populated with boxes, confidences, and
            class IDs from YOLO inference.

        Raises:
            ModelLoadError: If the model has not been injected yet.
            PredictionError: If YOLO inference raises an unexpected error.
        """
        if self._model is None:
            raise ModelLoadError(
                "YOLO",
                "Model not set — call set_model() before detect().",
            )

        # Preprocess
        img_resized = resize_for_yolo(image)
        img_array = pil_to_numpy(img_resized)

        logger.info(
            "Running YOLO inference (conf=%.2f, iou=%.2f).",
            _CONF_THRESH,
            _IOU_THRESH,
        )

        try:
            raw = self._model(
                img_array,
                conf=_CONF_THRESH,
                iou=_IOU_THRESH,
                imgsz=_IMG_SIZE,
                verbose=False,
            )
        except Exception as exc:
            raise PredictionError("YOLO", str(exc)) from exc

        result = DetectionResult()
        for pred in raw:
            boxes_xyxy = (
                pred.boxes.xyxy.cpu().numpy()
                if pred.boxes
                else np.empty((0, 4))
            )
            confs = (
                pred.boxes.conf.cpu().numpy()
                if pred.boxes
                else np.array([])
            )
            cls_ids = (
                pred.boxes.cls.cpu().numpy().astype(int)
                if pred.boxes
                else np.array([], dtype=int)
            )

            for box, conf, cls_id in zip(boxes_xyxy, confs, cls_ids):
                result.boxes.append(box.tolist())
                result.confidences.append(float(conf))
                result.class_ids.append(int(cls_id))

        result.panel_count = len(result.boxes)
        result.best_confidence = max(result.confidences, default=0.0)
        result.detection_successful = result.panel_count > 0

        logger.info(
            "Detection complete: %d panel(s) detected (best conf=%.3f).",
            result.panel_count,
            result.best_confidence,
        )
        return result
