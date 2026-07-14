"""services/recommendation.py — Maintenance recommendation engine.

Responsibility
--------------
- Analyse the combined pipeline results (fault class, physics, prediction).
- Generate a prioritised list of human-readable maintenance recommendations.
- Assign a severity level: CRITICAL | WARNING | INFO | OK.

No model inference, weather fetching, or UI logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from models.classifier import ClassificationResult
from models.predictor import PredictionResult
from services.physics import PhysicsResult
from utils.config import CFG
from utils.logger import get_logger

logger = get_logger(__name__)

# Recommendation thresholds from config
_R: dict = CFG["recommendations"]
_CRIT_LOSS: float = float(_R["efficiency_loss_critical_pct"])
_WARN_LOSS: float = float(_R["efficiency_loss_warning_pct"])
_HOT_TEMP: float = float(_R["hotspot_max_temp_c"])
_HUMID_SKIP: float = float(_R["cleaning_humidity_threshold_pct"])


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Priority level for a maintenance recommendation."""
    CRITICAL = "CRITICAL"
    WARNING  = "WARNING"
    INFO     = "INFO"
    OK       = "OK"


@dataclass
class Recommendation:
    """A single maintenance recommendation.

    Attributes:
        severity: Priority level (:class:`Severity`).
        message: Human-readable recommendation text.
        action: Short imperative action (e.g. ``"Schedule cleaning"``).
    """

    severity: Severity
    message: str
    action: str

    def to_dict(self) -> dict:
        """Return a plain dictionary representation of this recommendation.

        Returns:
            Dictionary with keys ``'severity'``, ``'message'``, ``'action'``.
            Suitable for JSON serialisation or passing to the UI layer.
        """
        return {
            "severity": self.severity.value,
            "message":  self.message,
            "action":   self.action,
        }


@dataclass
class RecommendationReport:
    """Full set of recommendations generated for one analysis run.

    Attributes:
        recommendations: Ordered list from highest to lowest severity.
        overall_severity: Highest severity level present in the list.
        summary: One-line summary suitable for display in the UI.
    """

    recommendations: List[Recommendation] = field(default_factory=list)
    overall_severity: Severity = Severity.OK
    summary: str = "No issues detected."

    def to_dict(self) -> dict:
        """Return a plain dictionary representation of the full report.

        This is the structured format required by the architecture review.
        The UI layer should consume this rather than the dataclass directly
        when JSON serialisation or framework-agnostic rendering is needed.

        Returns:
            Dictionary with keys:
            - ``'status'``         — overall severity string
            - ``'summary'``        — one-line human-readable summary
            - ``'issues'``         — list of issue dicts (see below)
            - ``'recommendation'`` — highest-priority action string
            - ``'priority'``       — alias of ``'status'`` for API compat

            Each entry in ``'issues'`` has keys ``'severity'``,
            ``'message'``, and ``'action'``.
        """
        issues = [r.to_dict() for r in self.recommendations]
        top_action = (
            self.recommendations[0].action
            if self.recommendations
            else "No action required"
        )
        return {
            "status":         self.overall_severity.value,
            "summary":        self.summary,
            "issues":         issues,
            "recommendation": top_action,
            "priority":       self.overall_severity.value,
        }


# ---------------------------------------------------------------------------
# Severity ordering helper
# ---------------------------------------------------------------------------

_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.CRITICAL: 3,
    Severity.WARNING: 2,
    Severity.INFO: 1,
    Severity.OK: 0,
}


def _max_severity(a: Severity, b: Severity) -> Severity:
    """Return the higher of two severity levels."""
    return a if _SEVERITY_ORDER[a] >= _SEVERITY_ORDER[b] else b


# ---------------------------------------------------------------------------
# Rule functions — each returns a Recommendation or None
# ---------------------------------------------------------------------------

def _rule_efficiency_loss(pred: PredictionResult) -> List[Recommendation]:
    """Flag high efficiency loss."""
    recs: List[Recommendation] = []
    loss = pred.efficiency_loss_pct

    if loss >= _CRIT_LOSS:
        recs.append(Recommendation(
            severity=Severity.CRITICAL,
            message=(
                f"Predicted efficiency loss of {loss:.1f}% exceeds the critical "
                f"threshold ({_CRIT_LOSS:.0f}%). Immediate intervention required."
            ),
            action="Schedule emergency maintenance",
        ))
    elif loss >= _WARN_LOSS:
        recs.append(Recommendation(
            severity=Severity.WARNING,
            message=(
                f"Predicted efficiency loss of {loss:.1f}% is above the advisory "
                f"threshold ({_WARN_LOSS:.0f}%). Plan maintenance soon."
            ),
            action="Schedule maintenance within 7 days",
        ))

    return recs


def _rule_fault_class(clf: ClassificationResult) -> List[Recommendation]:
    """Generate fault-class-specific recommendations."""
    recs: List[Recommendation] = []
    label = clf.label

    messages: dict[str, tuple[Severity, str, str]] = {
        "Clean": (
            Severity.OK,
            "Panel surface is clean. No cleaning required.",
            "Continue regular inspection schedule",
        ),
        "Dusty": (
            Severity.WARNING,
            "Dust accumulation detected. Cleaning will restore output.",
            "Schedule surface cleaning",
        ),
        "Bird-Drop": (
            Severity.WARNING,
            "Bird-drop contamination detected. Localised soiling reduces output.",
            "Clean affected cells promptly",
        ),
        "Electrical-Damage": (
            Severity.CRITICAL,
            "Electrical damage detected. Risk of arc fault or fire. Do not operate.",
            "Disconnect panel and contact a certified electrician immediately",
        ),
        "Physical-Damage": (
            Severity.CRITICAL,
            "Physical damage (cracking or delamination) detected. "
            "Structural integrity compromised.",
            "Replace the damaged panel",
        ),
        "Hotspot": (
            Severity.CRITICAL,
            "Hotspot detected. Localised overheating can cause permanent damage "
            "and is a fire risk.",
            "Inspect bypass diodes and shading sources; replace if needed",
        ),
    }

    if label in messages:
        sev, msg, action = messages[label]
        recs.append(Recommendation(severity=sev, message=msg, action=action))
    else:
        recs.append(Recommendation(
            severity=Severity.INFO,
            message=f"Fault class '{label}' detected. Manual inspection recommended.",
            action="Perform visual inspection",
        ))

    return recs


def _rule_module_temperature(physics: PhysicsResult) -> List[Recommendation]:
    """Warn if module temperature exceeds the hotspot threshold."""
    recs: List[Recommendation] = []
    if physics.module_temp_c >= _HOT_TEMP:
        recs.append(Recommendation(
            severity=Severity.WARNING,
            message=(
                f"Module temperature ({physics.module_temp_c:.1f}°C) exceeds "
                f"safe operating limit ({_HOT_TEMP:.0f}°C)."
            ),
            action="Check for shading, soiling, or ventilation obstructions",
        ))
    return recs


def _rule_low_irradiance(physics: PhysicsResult) -> List[Recommendation]:
    """Advise if irradiance is very low (prediction may be unreliable)."""
    recs: List[Recommendation] = []
    if physics.irradiance_wm2 < 50.0:
        recs.append(Recommendation(
            severity=Severity.INFO,
            message=(
                f"Irradiance is very low ({physics.irradiance_wm2:.0f} W/m²). "
                "Predictions are less reliable under night/heavy overcast conditions."
            ),
            action="Re-run analysis during daylight hours",
        ))
    return recs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_recommendations(
    classification: ClassificationResult,
    physics: PhysicsResult,
    prediction: PredictionResult,
) -> RecommendationReport:
    """Run all recommendation rules and produce a prioritised report.

    Args:
        classification: Fault classification result.
        physics: Computed physics parameters.
        prediction: XGBoost regression result.

    Returns:
        :class:`RecommendationReport` with ordered recommendations and overall
        severity.
    """
    all_recs: List[Recommendation] = []

    all_recs.extend(_rule_fault_class(classification))
    all_recs.extend(_rule_efficiency_loss(prediction))
    all_recs.extend(_rule_module_temperature(physics))
    all_recs.extend(_rule_low_irradiance(physics))

    # Sort by severity (CRITICAL first)
    all_recs.sort(key=lambda r: _SEVERITY_ORDER[r.severity], reverse=True)

    overall = Severity.OK
    for rec in all_recs:
        overall = _max_severity(overall, rec.severity)

    if overall == Severity.CRITICAL:
        summary = "⚠️ Critical issues detected — immediate action required."
    elif overall == Severity.WARNING:
        summary = "⚡ Maintenance recommended within 7 days."
    elif overall == Severity.INFO:
        summary = "ℹ️ Minor advisory — re-inspect during next scheduled visit."
    else:
        summary = "✅ Panel is operating within normal parameters."

    report = RecommendationReport(
        recommendations=all_recs,
        overall_severity=overall,
        summary=summary,
    )

    logger.info(
        "Recommendations generated: %d item(s), overall=%s.",
        len(all_recs), overall.value,
    )
    return report
