"""services/pipeline.py — Application orchestrator.

Responsibility
--------------
This is the **only** public entry-point called by ``app.py``.  It:

- Retrieves all AI models from :class:`~models.model_manager.ModelManager`
  (models are loaded exactly once per process).
- Orchestrates every step of the analysis in order.
- Returns a single :class:`PipelineResult` to the UI layer.

Workflow
--------
::

    Image + inputs
      ↓ ModelManager → SolarPanelDetector      → DetectionResult
      ↓ ModelManager → SolarFaultClassifier    → ClassificationResult
      ↓ fetch_weather                           → WeatherData
      ↓ compute_physics                         → PhysicsResult
      ↓ build_feature_dataframe (+ validation) → pd.DataFrame
      ↓ ModelManager → EnergyPredictor         → PredictionResult
      ↓ generate_recommendations               → RecommendationReport
      ↓
    PipelineResult (returned to app.py)

``app.py`` must call :func:`run_pipeline` and must not import any model or
service directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time

import pandas as pd
from PIL import Image

from models.classifier import ClassificationResult, SolarFaultClassifier
from models.detector import DetectionResult, SolarPanelDetector
from models.model_manager import model_manager
from models.predictor import EnergyPredictor, PredictionResult
from services.feature_engineering import build_feature_dataframe
from services.physics import PhysicsResult, compute_physics
from services.recommendation import RecommendationReport, generate_recommendations
from services.weather import WeatherData, fetch_weather
from utils.config import CFG
from utils.exceptions import (
    FeatureValidationError,
    ImageValidationError,
    ModelLoadError,
    PredictionError,
    SolarAIError,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Consolidated results from a full analysis pipeline run.

    Every field mirrors the output of a downstream service so that ``app.py``
    only needs to import :class:`PipelineResult` and not individual service
    data classes.

    Attributes:
        detection_result: YOLO detection result.
        classification_result: MobileNet classification result.
        weather_data: Weather observation used in this run.
        physics_data: Computed physics parameters.
        feature_dataframe: Assembled features for inference.
        efficiency_prediction: XGBoost regression output.
        recommendations: Maintenance recommendation report.
        processing_time: Total time taken by the pipeline in seconds.
        status: "SUCCESS" or "ERROR".
        city: City used for the weather lookup.
        error_message: Human-readable error description on failure.
        error_type: Exception class name for structured error handling.
    """

    detection_result: DetectionResult = field(default_factory=DetectionResult)
    classification_result: ClassificationResult = field(default_factory=ClassificationResult)
    weather_data: WeatherData = field(default_factory=WeatherData)
    physics_data: PhysicsResult = field(default_factory=PhysicsResult)
    feature_dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    efficiency_prediction: PredictionResult = field(default_factory=PredictionResult)
    recommendations: RecommendationReport = field(default_factory=RecommendationReport)
    processing_time: float = 0.0
    status: str = "ERROR"
    city: str = ""
    error_message: str = ""
    error_type: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    image: Image.Image,
    city: Optional[str] = None,
    panel_age: float = 0.0,
    maintenance_count: int = 0,
    voltage: float = 0.0,
    current: float = 0.0,
    installation_type: str = "rooftop",
) -> PipelineResult:
    """Execute the full Solar AI analysis pipeline on a single image.

    Args:
        image: Uploaded solar panel image as a PIL RGB Image.
        city: City name for the OpenWeatherMap API lookup.  Falls back to
              ``weather.default_city`` in ``configs/settings.yaml``.
        panel_age: Age of the panel in years (future feature engineering use).
        maintenance_count: Number of prior maintenance events (future use).
        voltage: Measured panel voltage in Volts (future use).
        current: Measured panel current in Amperes (future use).
        installation_type: Mounting type, e.g. ``"rooftop"`` or
                           ``"ground-mount"`` (future use).

    Returns:
        :class:`PipelineResult` aggregating all service outputs.
        On any unrecoverable error, ``status="ERROR"``, ``error_message``
        describes the failure, and ``error_type`` names the exception class.
    """
    start_time = time.time()
    result = PipelineResult()
    result.city = city or CFG["weather"]["default_city"]

    # Log supplementary inputs for traceability
    logger.info(
        "Pipeline started — city='%s', panel_age=%.1f yr, "
        "maintenance_count=%d, voltage=%.2f V, current=%.2f A, "
        "installation='%s'",
        result.city, panel_age, maintenance_count,
        voltage, current, installation_type,
    )

    try:
        # ── Step 0: Image validation ────────────────────────────────────────
        if image is None:
            raise ImageValidationError("No image provided to the pipeline.")
        if image.mode != "RGB":
            image = image.convert("RGB")

        # ── Step 1: Detection ───────────────────────────────────────────────
        logger.info("Pipeline step 1/7: YOLO detection")
        detector = SolarPanelDetector()
        detector.set_model(model_manager.get_detector())
        result.detection_result = detector.detect(image)

        # ── Step 2: Classification ──────────────────────────────────────────
        logger.info("Pipeline step 2/7: MobileNet classification")
        classifier = SolarFaultClassifier()
        classifier.set_model(model_manager.get_classifier())
        result.classification_result = classifier.classify(image)

        # ── Step 3: Weather ─────────────────────────────────────────────────
        logger.info("Pipeline step 3/7: Weather fetch for '%s'", result.city)
        result.weather_data = fetch_weather(result.city)

        # ── Step 4: Physics ─────────────────────────────────────────────────
        logger.info("Pipeline step 4/7: Physics calculations")
        result.physics_data = compute_physics(
            ambient_temp_c=result.weather_data.ambient_temp_c,
            wind_speed_ms=result.weather_data.wind_speed_ms,
            cloud_cover_pct=result.weather_data.cloud_cover_pct,
            fault_label=result.classification_result.label,
        )

        # ── Step 5: Feature Engineering + Validation ─────────────────────────
        logger.info("Pipeline step 5/7: Feature engineering and validation")
        result.feature_dataframe = build_feature_dataframe(
            weather=result.weather_data,
            physics=result.physics_data,
            classification=result.classification_result,
            detection=result.detection_result,
        )

        # ── Step 6: Prediction ───────────────────────────────────────────────
        logger.info("Pipeline step 6/7: XGBoost prediction")
        predictor = EnergyPredictor()
        predictor.set_model(model_manager.get_predictor())
        result.efficiency_prediction = predictor.predict(result.feature_dataframe)

        # ── Step 7: Recommendations ──────────────────────────────────────────
        logger.info("Pipeline step 7/7: Recommendation generation")
        result.recommendations = generate_recommendations(
            classification=result.classification_result,
            physics=result.physics_data,
            prediction=result.efficiency_prediction,
        )

        result.status = "SUCCESS"
        logger.info(
            "Pipeline complete. Overall severity: %s",
            result.recommendations.overall_severity.value,
        )

    except (
        ImageValidationError,
        ModelLoadError,
        PredictionError,
        FeatureValidationError,
        SolarAIError,
    ) as exc:
        result.status = "ERROR"
        result.error_message = str(exc)
        result.error_type = type(exc).__name__
        logger.error("Pipeline failed [%s]: %s", result.error_type, exc)

    except Exception as exc:  # noqa: BLE001 — catch-all for unexpected errors
        result.status = "ERROR"
        result.error_message = f"Unexpected error: {exc}"
        result.error_type = type(exc).__name__
        logger.exception("Pipeline unexpected failure: %s", exc)

    result.processing_time = time.time() - start_time
    return result
