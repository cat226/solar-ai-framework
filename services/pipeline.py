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
from typing import List, Optional
import math
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
    InputValidationError,
    ModelLoadError,
    PredictionError,
    SolarAIError,
)
from utils.image_utils import crop_panel
from utils.logger import get_logger
from utils.security import sanitize_for_log

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class PanelResult:
    """Per-panel analysis: one detected bounding box, classified and
    (when the XGBoost artifact is available) predicted independently of
    every other panel in the same image.

    Attributes:
        panel_index: 1-based index matching the order panels were detected in
                     (and the numbering drawn on the detection overlay).
        box: [x1, y1, x2, y2] in the *original* uploaded image's pixel
             coordinates (already mapped out of YOLO's letterboxed space).
        detection_confidence: YOLO's confidence for this specific box.
        classification: MobileNet result for the cropped panel region -
                         never the whole image.
        physics: Physics computed using this panel's own classification
                  label and shared (image-level) weather conditions.
        prediction: XGBoost result for this panel.
                    ``prediction_successful=False`` (with 0.0 fields, never
                    a fabricated number) when the XGBoost artifact is not
                    available - see ``PipelineResult.xgboost_available``.
    """

    panel_index: int = 0
    box: List[float] = field(default_factory=list)
    detection_confidence: float = 0.0
    classification: ClassificationResult = field(default_factory=ClassificationResult)
    physics: PhysicsResult = field(default_factory=PhysicsResult)
    prediction: PredictionResult = field(default_factory=PredictionResult)


@dataclass
class SiteSummary:
    """Site-level aggregation across every panel in ``PipelineResult.panels``.

    Every field here is derived strictly from real per-panel results - never
    fabricated when there are zero panels (all counts are 0, all rates are
    0.0, not fabricated placeholders).
    """

    total_panels: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    clean_pct: float = 0.0
    average_efficiency_loss_pct: float = 0.0
    average_estimated_output_w: float = 0.0
    panels_with_prediction: int = 0


def _build_site_summary(panels: List["PanelResult"]) -> SiteSummary:
    """Aggregate a list of PanelResult into one SiteSummary. Pure function -
    no I/O, no model calls - so it's trivially unit-testable and never
    invents a value for an empty panel list."""
    if not panels:
        return SiteSummary()

    class_counts: dict[str, int] = {}
    for p in panels:
        class_counts[p.classification.label] = class_counts.get(p.classification.label, 0) + 1

    total = len(panels)
    clean_count = class_counts.get("Clean", 0)
    predicted = [p for p in panels if p.prediction.prediction_successful]

    return SiteSummary(
        total_panels=total,
        class_counts=class_counts,
        clean_pct=round(100.0 * clean_count / total, 2),
        average_efficiency_loss_pct=(
            round(sum(p.prediction.efficiency_loss_pct for p in predicted) / len(predicted), 2)
            if predicted else 0.0
        ),
        average_estimated_output_w=(
            round(sum(p.prediction.estimated_output_w for p in predicted) / len(predicted), 2)
            if predicted else 0.0
        ),
        panels_with_prediction=len(predicted),
    )


@dataclass
class PipelineResult:
    """Consolidated results from a full analysis pipeline run.

    Every field mirrors the output of a downstream service so that ``app.py``
    only needs to import :class:`PipelineResult` and not individual service
    data classes.

    Attributes:
        detection_result: YOLO detection result.
        classification_result: MobileNet classification result for the
            *whole uploaded image* (kept for backward compatibility with
            existing UI code) - see ``panels`` for the real per-detected-
            panel breakdown (crop → classify → physics → predict).
        weather_data: Weather observation used in this run.
        physics_data: Computed physics parameters for the whole-image result.
        feature_dataframe: Assembled features for the whole-image prediction.
        efficiency_prediction: XGBoost regression output for the whole-image
            result. ``prediction_successful=False`` when the XGBoost
            artifact is unavailable - see ``xgboost_available``.
        panels: One :class:`PanelResult` per YOLO detection box, each
            independently classified from its own cropped region (this is
            what "Panel crops → MobileNet classification" in the pipeline
            diagram refers to). Empty list on zero detections - never
            fabricated placeholder panels.
        site_summary: Aggregation of ``panels`` - see :class:`SiteSummary`.
        classifier_source: ``"production"``, ``"interim"``, or ``"unknown"``
            (set only when classification never ran, e.g. on early
            validation failure) - which MobileNet artifact actually
            produced ``classification_result``/every ``panels[i].classification``.
        xgboost_available: Whether the XGBoost artifact was present and
            loadable for this run. When ``False``, every prediction field
            above reports ``prediction_successful=False`` rather than the
            whole pipeline aborting - detection and classification results
            are still genuinely computed and returned.
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
    panels: List[PanelResult] = field(default_factory=list)
    site_summary: SiteSummary = field(default_factory=SiteSummary)
    classifier_source: str = "unknown"
    xgboost_available: bool = False
    recommendations: RecommendationReport = field(default_factory=RecommendationReport)
    processing_time: float = 0.0
    status: str = "ERROR"
    city: str = ""
    error_message: str = ""
    error_type: str = ""


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_scalar_inputs(
    panel_age: float,
    maintenance_count: int,
    voltage: float,
    current: float,
    installation_type: str,
) -> None:
    """Validate supplementary scalar inputs, failing early on bad values.

    These are the non-image parameters supplied at the public entry point.
    Each check raises :class:`InputValidationError` with an actionable
    message naming the offending parameter and its expected range.

    Args:
        panel_age: Age of the panel in years; must be finite and in [0, 100].
        maintenance_count: Prior maintenance events; must be an integer >= 0.
        voltage: Measured panel voltage in Volts; must be finite and >= 0.
        current: Measured panel current in Amperes; must be finite and >= 0.
        installation_type: Non-empty mounting type string.

    Raises:
        InputValidationError: If any parameter is out of range or the wrong type.
    """
    import math

    def _finite(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InputValidationError(
                f"'{name}' must be a number, got {type(value).__name__}."
            )
        num = float(value)
        if not math.isfinite(num):
            raise InputValidationError(f"'{name}' must be a finite number.")
        return num

    age = _finite("panel_age", panel_age)
    if not 0.0 <= age <= 100.0:
        raise InputValidationError(
            f"'panel_age' must be between 0 and 100 years, got {age:g}."
        )

    if isinstance(maintenance_count, bool) or not isinstance(maintenance_count, int):
        raise InputValidationError(
            "'maintenance_count' must be an integer, got "
            f"{type(maintenance_count).__name__}."
        )
    if maintenance_count < 0:
        raise InputValidationError(
            f"'maintenance_count' must be >= 0, got {maintenance_count}."
        )

    v = _finite("voltage", voltage)
    if v < 0.0:
        raise InputValidationError(f"'voltage' must be >= 0 V, got {v:g}.")

    c = _finite("current", current)
    if c < 0.0:
        raise InputValidationError(f"'current' must be >= 0 A, got {c:g}.")

    if not isinstance(installation_type, str) or not installation_type.strip():
        raise InputValidationError(
            "'installation_type' must be a non-empty string "
            "(e.g. 'rooftop' or 'ground-mount')."
        )


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
        sanitize_for_log(result.city),
        panel_age,
        maintenance_count,
        voltage,
        current,
        sanitize_for_log(installation_type),
    )

    try:
        # ── Step 0: Input validation ────────────────────────────────────────
        if image is None:
            raise ImageValidationError("No image provided to the pipeline.")
        if not isinstance(image, Image.Image):
            raise ImageValidationError(
                "Pipeline input must be a PIL image, got "
                f"{type(image).__name__}."
            )
        _validate_scalar_inputs(
            panel_age=panel_age,
            maintenance_count=maintenance_count,
            voltage=voltage,
            current=current,
            installation_type=installation_type,
        )
        if image.mode != "RGB":
            image = image.convert("RGB")

        # ── Step 1: Detection ───────────────────────────────────────────────
        logger.info("Pipeline step 1/7: YOLO detection")
        detector = SolarPanelDetector()
        detector.set_model(model_manager.get_detector())
        result.detection_result = detector.detect(image)

        # ── Step 2: Classification (whole image, kept for backward compat) ──
        logger.info("Pipeline step 2/7: MobileNet classification")
        classifier = SolarFaultClassifier()
        active_labels = model_manager.classifier_labels
        result.classifier_source = model_manager.classifier_source or "unknown"
        classifier.set_model(model_manager.get_classifier(), labels=active_labels)
        result.classification_result = classifier.classify(image)

        # ── Step 3: Weather ─────────────────────────────────────────────────
        logger.info("Weather fetch for '%s'", sanitize_for_log(result.city))
        result.weather_data = fetch_weather(result.city)

        # ── Step 4: Physics ─────────────────────────────────────────────────
        logger.info("Pipeline step 4/7: Physics calculations")
        result.physics_data = compute_physics(
            ambient_temp_c=result.weather_data.ambient_temp_c,
            wind_speed_ms=result.weather_data.wind_speed_ms,
            cloud_cover_pct=result.weather_data.cloud_cover_pct,
            fault_label=result.classification_result.label,
            latitude=result.weather_data.latitude,
            longitude=result.weather_data.longitude,
            observation_time=result.weather_data.timestamp,
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
        # XGBoost is treated as an optional downstream component: when its
        # artifact is genuinely absent (weights/xgboost_solar.joblib does
        # not exist), that must be reported clearly - via
        # xgboost_available=False and prediction_successful=False - rather
        # than aborting a run that otherwise has real, useful detection and
        # classification results. A model that loads but then fails during
        # prediction is a different, real error and still aborts below.
        logger.info("Pipeline step 6/7: XGBoost prediction")
        predictor = EnergyPredictor()
        try:
            predictor.set_model(model_manager.get_predictor())
            result.xgboost_available = True
        except ModelLoadError as exc:
            result.xgboost_available = False
            logger.warning(
                "XGBoost artifact unavailable - predictions will be reported as "
                "unavailable, not fabricated: %s", exc,
            )

        if result.xgboost_available:
            result.efficiency_prediction = predictor.predict(result.feature_dataframe)
            if not all(math.isfinite(v) for v in (
                result.efficiency_prediction.efficiency_loss_pct,
                result.efficiency_prediction.estimated_output_w,
            )):
                raise PredictionError(
                    "XGBoost",
                    "Prediction output contains non-finite values.",
                )

        # ── Step 6b: Per-panel analysis ──────────────────────────────────────
        # "YOLO -> Panel crops -> MobileNet classification -> ... -> per-panel
        # + site-level results" - each detected box is cropped from the real
        # original image and classified/predicted independently, rather than
        # every panel silently sharing the single whole-image classification
        # above. Reuses the exact same classifier/predictor instances (no
        # duplicated model-loading or inference logic).
        logger.info("Pipeline step 6b: per-panel analysis (%d panel(s))", result.detection_result.panel_count)
        panels: List[PanelResult] = []
        for idx, (box, det_conf) in enumerate(
            zip(result.detection_result.boxes, result.detection_result.confidences), start=1
        ):
            crop = crop_panel(image, tuple(box))
            panel_classification = classifier.classify(crop)
            panel_physics = compute_physics(
                ambient_temp_c=result.weather_data.ambient_temp_c,
                wind_speed_ms=result.weather_data.wind_speed_ms,
                cloud_cover_pct=result.weather_data.cloud_cover_pct,
                fault_label=panel_classification.label,
                latitude=result.weather_data.latitude,
                longitude=result.weather_data.longitude,
                observation_time=result.weather_data.timestamp,
            )
            panel_prediction = PredictionResult()
            if result.xgboost_available:
                panel_features = build_feature_dataframe(
                    weather=result.weather_data,
                    physics=panel_physics,
                    classification=panel_classification,
                    detection=DetectionResult(
                        boxes=[list(box)], confidences=[det_conf], class_ids=[0],
                        panel_count=1, best_confidence=det_conf, detection_successful=True,
                    ),
                )
                panel_prediction = predictor.predict(panel_features)
                if not all(math.isfinite(v) for v in (
                    panel_prediction.efficiency_loss_pct, panel_prediction.estimated_output_w,
                )):
                    raise PredictionError(
                        "XGBoost",
                        f"Panel #{idx} prediction output contains non-finite values.",
                    )
            panels.append(PanelResult(
                panel_index=idx,
                box=list(box),
                detection_confidence=float(det_conf),
                classification=panel_classification,
                physics=panel_physics,
                prediction=panel_prediction,
            ))
        result.panels = panels
        result.site_summary = _build_site_summary(panels)

        # ── Step 7: Recommendations ──────────────────────────────────────────
        # Skipped (report stays at its honest default - see RecommendationReport
        # in services/recommendation.py) when the prediction that recommendations
        # are based on isn't real - showing "No issues detected" derived from a
        # fabricated 0.0 efficiency loss would be exactly the kind of
        # misleading-because-unlabeled result this project's no-fabrication
        # policy forbids. The UI layer must check xgboost_available explicitly.
        if result.xgboost_available:
            logger.info("Pipeline step 7/7: Recommendation generation")
            result.recommendations = generate_recommendations(
                classification=result.classification_result,
                physics=result.physics_data,
                prediction=result.efficiency_prediction,
            )
        else:
            logger.info("Pipeline step 7/7: skipped (XGBoost unavailable, no prediction to base recommendations on)")

        result.status = "SUCCESS"
        logger.info(
            "Pipeline complete. Overall severity: %s",
            result.recommendations.overall_severity.value,
        )

    except (
        InputValidationError,
        ImageValidationError,
        ModelLoadError,
        PredictionError,
        FeatureValidationError,
        SolarAIError,
    ) as exc:
        # Known, typed failures carry user-safe, actionable messages already.
        result.status = "ERROR"
        result.error_message = str(exc)
        result.error_type = type(exc).__name__
        logger.error("Pipeline failed [%s]: %s", result.error_type, exc)

    except Exception as exc:  # noqa: BLE001 — catch-all for unexpected errors
        # Unexpected errors: log full details for diagnostics, but show the
        # user a generic message so raw internals are not exposed in the UI.
        result.status = "ERROR"
        result.error_message = (
            "An unexpected error occurred while analysing the image. "
            "Please try again; if the problem persists, check the logs."
        )
        result.error_type = type(exc).__name__
        logger.exception("Pipeline unexpected failure [%s]: %s",
                         type(exc).__name__, exc)

    result.processing_time = time.time() - start_time
    logger.info("Pipeline Timing: execution completed in %.2fs", result.processing_time)
    return result
