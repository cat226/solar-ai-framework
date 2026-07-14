"""models/predictor.py — XGBoost regression for solar energy output prediction.

Responsibility
--------------
- Wrap a joblib-serialised XGBoost pipeline (supplied by
  :mod:`models.model_manager`) and run regression on a feature DataFrame.
- Return the predicted efficiency loss (%) and estimated output power (W).

Model loading
-------------
This module does **not** load the XGBoost model itself.  It receives the
loaded pipeline object from :class:`models.model_manager.ModelManager` via
:meth:`EnergyPredictor.set_model`.

This module is purely concerned with ML regression.  It has no knowledge of
the UI, weather fetching, or physics computations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from utils.config import CFG
from utils.exceptions import ModelLoadError, PredictionError
from utils.logger import get_logger

logger = get_logger(__name__)

# Read panel rating from config
_PANEL_RATED_POWER: float = float(CFG["physics"]["panel_rated_power_wp"])


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Encapsulates the regression output from the XGBoost model.

    Attributes:
        efficiency_loss_pct: Predicted efficiency loss as a percentage (0–100).
        estimated_output_w: Estimated actual panel output in Watts.
        prediction_successful: True if inference ran without errors.
    """

    efficiency_loss_pct: float = 0.0
    estimated_output_w: float = 0.0
    prediction_successful: bool = False


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class EnergyPredictor:
    """Wraps a joblib-serialised XGBoost pipeline for energy output regression.

    The pipeline object is injected via :meth:`set_model` (called by
    :mod:`services.pipeline` after retrieving it from
    :class:`models.model_manager.ModelManager`).

    Usage via pipeline (preferred)::

        from models.model_manager import model_manager
        predictor = EnergyPredictor()
        predictor.set_model(model_manager.get_predictor())
        result = predictor.predict(feature_dataframe)
    """

    def __init__(self) -> None:
        self._pipeline: Optional[object] = None

    def set_model(self, pipeline: object) -> None:
        """Inject the loaded XGBoost pipeline.

        Args:
            pipeline: A loaded joblib pipeline with a ``.predict()`` method,
                      obtained from :class:`~models.model_manager.ModelManager`.

        Raises:
            ModelLoadError: If *pipeline* is ``None``.
        """
        if pipeline is None:
            raise ModelLoadError(
                "XGBoost", "set_model() received None — check ModelManager."
            )
        self._pipeline = pipeline
        logger.debug("EnergyPredictor: pipeline injected.")

    def predict(self, features: pd.DataFrame) -> PredictionResult:
        """Run regression on a pre-built, validated feature DataFrame.

        Args:
            features: DataFrame produced and validated by
                      :func:`services.feature_engineering.build_feature_dataframe`.
                      Must contain the columns listed under
                      ``feature_engineering.feature_columns`` in settings.yaml.

        Returns:
            :class:`PredictionResult` with efficiency loss (%) and estimated
            output power (W).

        Raises:
            ModelLoadError: If the pipeline has not been injected yet.
            PredictionError: If ``pipeline.predict()`` raises an error.
        """
        if self._pipeline is None:
            raise ModelLoadError(
                "XGBoost",
                "Pipeline not set — call set_model() before predict().",
            )

        logger.info("Running XGBoost prediction on feature vector.")
        logger.debug("Feature vector:\n%s", features.to_string())

        try:
            raw_pred = self._pipeline.predict(features)
        except Exception as exc:
            raise PredictionError("XGBoost", str(exc)) from exc

        efficiency_loss_pct: float = float(raw_pred[0])

        # Clamp to [0, 100] — model might predict slight negatives or >100
        efficiency_loss_pct = max(0.0, min(100.0, efficiency_loss_pct))

        estimated_output_w: float = _PANEL_RATED_POWER * (
            1.0 - efficiency_loss_pct / 100.0
        )

        logger.info(
            "Prediction complete: efficiency_loss=%.2f%%, output=%.2f W.",
            efficiency_loss_pct,
            estimated_output_w,
        )

        return PredictionResult(
            efficiency_loss_pct=efficiency_loss_pct,
            estimated_output_w=estimated_output_w,
            prediction_successful=True,
        )
