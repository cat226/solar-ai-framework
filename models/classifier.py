"""models/classifier.py — MobileNet-based solar panel fault classification.

Responsibility
--------------
- Wrap a MobileNetV2 model (supplied by :mod:`models.model_manager`) and run
  inference on an input image.
- Apply standard ImageNet-normalisation transforms.
- Run a forward pass and apply Softmax.
- Return the predicted fault class, confidence, and full probability distribution.

Model loading
-------------
This module does **not** load the MobileNet model itself.  It receives the
loaded ``torch.nn.Module`` from :class:`models.model_manager.ModelManager`
via :meth:`SolarFaultClassifier.set_model`.

This module is purely concerned with image classification.  It has no
knowledge of the UI, weather data, or downstream prediction logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms  # type: ignore

from utils.config import CFG
from utils.exceptions import ModelLoadError, PredictionError
from utils.image_utils import resize_for_mobilenet
from utils.logger import get_logger

logger = get_logger(__name__)

# Read classification config
_LABELS: List[str] = CFG["classification"]["labels"]

# ImageNet normalisation constants used during MobileNet pre-training
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Inference transform pipeline (resize handled by image_utils; tensor ops only here)
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Encapsulates the output of MobileNet inference on a single image.

    Attributes:
        label: Predicted fault class name (e.g. ``"Dusty"``).
        class_id: Integer index of the predicted class.
        confidence: Softmax probability for the predicted class (0.0–1.0).
        probabilities: Full softmax probability distribution over all classes,
                       keyed by class label.
        classification_successful: True if inference ran without errors.
    """

    label: str = "Unknown"
    class_id: int = -1
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    classification_successful: bool = False


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class SolarFaultClassifier:
    """Wraps a fine-tuned MobileNetV2 to classify solar panel fault types.

    The model object is injected via :meth:`set_model` (called by
    :mod:`services.pipeline` after retrieving it from
    :class:`models.model_manager.ModelManager`).

    Usage via pipeline (preferred)::

        from models.model_manager import model_manager
        classifier = SolarFaultClassifier()
        classifier.set_model(model_manager.get_classifier())
        result = classifier.classify(pil_image)
    """

    def __init__(self) -> None:
        self._model: Optional[torch.nn.Module] = None
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def set_model(self, model: torch.nn.Module) -> None:
        """Inject the loaded MobileNetV2 model.

        Args:
            model: A loaded ``torch.nn.Module`` obtained from
                   :class:`~models.model_manager.ModelManager`, already in
                   eval mode and on the correct device.

        Raises:
            ModelLoadError: If *model* is ``None``.
        """
        if model is None:
            raise ModelLoadError(
                "MobileNet", "set_model() received None — check ModelManager."
            )
        self._model = model
        # Sync device with the model's first parameter
        try:
            self._device = next(model.parameters()).device
        except StopIteration:
            pass
        logger.debug("SolarFaultClassifier: model injected (device=%s).", self._device)

    def classify(self, image: Image.Image) -> ClassificationResult:
        """Run MobileNetV2 inference on a PIL image.

        Args:
            image: Input image (RGB PIL Image).

        Returns:
            :class:`ClassificationResult` with predicted label, confidence,
            and the full probability distribution over all fault classes.

        Raises:
            ModelLoadError: If the model has not been injected yet.
            PredictionError: If inference raises an unexpected error.
        """
        if self._model is None:
            raise ModelLoadError(
                "MobileNet",
                "Model not set — call set_model() before classify().",
            )

        # Preprocess: centre-crop → tensor → normalise
        img_cropped = resize_for_mobilenet(image)
        tensor: torch.Tensor = _TRANSFORM(img_cropped)        # (3, H, W)
        tensor = tensor.unsqueeze(0).to(self._device)          # (1, 3, H, W)

        logger.info("Running MobileNet classification inference.")

        try:
            with torch.no_grad():
                logits: torch.Tensor = self._model(tensor)    # (1, num_classes)
                probs: torch.Tensor = F.softmax(logits, dim=1)
        except Exception as exc:
            raise PredictionError("MobileNet", str(exc)) from exc

        probs_np = probs.squeeze(0).cpu().numpy()

        class_id: int = int(probs_np.argmax())
        confidence: float = float(probs_np[class_id])
        label: str = _LABELS[class_id] if class_id < len(_LABELS) else "Unknown"

        prob_dict: Dict[str, float] = {
            lbl: float(p) for lbl, p in zip(_LABELS, probs_np)
        }

        logger.info(
            "Classification complete: %s (confidence=%.3f).", label, confidence
        )

        return ClassificationResult(
            label=label,
            class_id=class_id,
            confidence=confidence,
            probabilities=prob_dict,
            classification_successful=True,
        )
