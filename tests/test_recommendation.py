"""tests/test_recommendation.py - Deterministic unit tests for the
maintenance recommendation engine (``services/recommendation.py``).

Scope
-----
Purely a characterization / regression suite for the CURRENT implementation:

* healthy / normal condition (Clean panel, low loss)
* reduced efficiency  (warning / critical thresholds + boundaries)
* panel condition      (Clean, Dusty, Bird-Drop, damage classes, Hotspot,
                        unknown label)
* module-temperature rule (boundary)
* low-irradiance rule (boundary)
* severity / priority ordering
* recommendation list content, structure and deterministic ordering
* combined conditions (fault + loss, Clean + advisory, high loss + damage)
* report ``to_dict()`` structure
* determinism

Design rules honoured:
* deterministic, isolated, fast, model-free, weight-free, network-free, no
  Streamlit / YOLO / MobileNet / XGBoost / OpenWeatherMap.
* thresholds are read from the live configuration (``CFG["recommendations"]``)
  where the implementation does so, never hard-coded guesses.
* no production code is modified; expectations mirror current behavior.

Coverage target: 100% statement/branch of ``services/recommendation.py``.
"""

from __future__ import annotations

import pytest

from models.classifier import ClassificationResult
from models.predictor import PredictionResult
from services.physics import PhysicsResult
from services.recommendation import (
    Recommendation,
    RecommendationReport,
    Severity,
    generate_recommendations,
    _CRIT_LOSS,
    _HOT_TEMP,
    _SEVERITY_ORDER,
    _WARN_LOSS,
    _max_severity,
    _rule_efficiency_loss,
    _rule_fault_class,
    _rule_low_irradiance,
    _rule_module_temperature,
)

# ---------------------------------------------------------------------------
# Thresholds resolved from the live configuration (not guessed).
# ---------------------------------------------------------------------------

CRIT_LOSS = _CRIT_LOSS   # 20.0  - CFG["recommendations"]["efficiency_loss_critical_pct"]
WARN_LOSS = _WARN_LOSS   # 10.0  - CFG["recommendations"]["efficiency_loss_warning_pct"]
HOT_TEMP = _HOT_TEMP     # 65.0  - CFG["recommendations"]["hotspot_max_temp_c"]

# Low-irradiance threshold is hard-coded in the implementation (not config).
LOW_IRR = 50.0

SEVERITIES = ["CRITICAL", "WARNING", "INFO", "OK"]

# Fault label -> (expected severity, substring pin, substring action pin).
# Mirrored from the current `_rule_fault_class` message table.
EXPECTED_FAULT_MESSAGES = {
    "Clean": (
        Severity.OK, "clean", "Continue regular inspection schedule"),
    "Dusty": (
        Severity.WARNING, "Dust", "Schedule surface cleaning"),
    "Bird-Drop": (
        Severity.WARNING, "Bird-drop", "Clean affected cells promptly"),
    "Electrical-Damage": (
        Severity.CRITICAL, "Electrical damage",
        "Disconnect panel and contact a certified electrician immediately"),
    "Physical-Damage": (
        Severity.CRITICAL, "Physical damage", "Replace the damaged panel"),
    "Hotspot": (
        Severity.CRITICAL, "Hotspot", "Inspect bypass diodes"),
}


# ---------------------------------------------------------------------------
# Deterministic plain-data factories (no model inference, no network).
# ---------------------------------------------------------------------------


def make_classification(label="Clean", confidence=0.95, successful=True):
    """Build a ``ClassificationResult`` with the given label."""
    return ClassificationResult(
        label=label,
        class_id=0,
        confidence=confidence,
        probabilities={label: confidence},
        classification_successful=successful,
    )


def make_prediction(loss=2.0, output_w=392.0, successful=True):
    """Build a ``PredictionResult`` with the given efficiency-loss percentage."""
    return PredictionResult(
        efficiency_loss_pct=loss,
        estimated_output_w=output_w,
        prediction_successful=successful,
    )


def make_physics(module_temp=49.75, irradiance=888.0):
    """Build a ``PhysicsResult``; only the two fields the rules read matter."""
    return PhysicsResult(
        irradiance_wm2=irradiance,
        module_temp_c=module_temp,
        soiling_ratio=1.0,
        temp_loss_pct=9.9,
        effective_efficiency=0.901,
        cloud_factor=0.925,
        wind_cooling_factor=3.0,
    )


# ---------------------------------------------------------------------------
# 1. Efficiency-loss rule (reduced efficiency + boundaries)
# ---------------------------------------------------------------------------


class TestEfficiencyLossRule:
    """_rule_efficiency_loss: OK / WARNING / CRITICAL with inclusive bounds."""

    def test_none_below_warning_threshold(self):
        assert _rule_efficiency_loss(make_prediction(loss=2.0)) == []

    def test_none_at_zero_loss(self):
        assert _rule_efficiency_loss(make_prediction(loss=0.0)) == []

    def test_none_just_below_warning(self):
        assert _rule_efficiency_loss(make_prediction(loss=WARN_LOSS - 0.01)) == []

    def test_warning_exactly_at_warning_threshold(self):
        recs = _rule_efficiency_loss(make_prediction(loss=WARN_LOSS))
        assert len(recs) == 1
        assert recs[0].severity == Severity.WARNING
        assert recs[0].action == "Schedule maintenance within 7 days"
        assert f"{WARN_LOSS:.0f}%" in recs[0].message

    def test_warning_just_above_warning_threshold(self):
        recs = _rule_efficiency_loss(make_prediction(loss=WARN_LOSS + 0.01))
        assert len(recs) == 1
        assert recs[0].severity == Severity.WARNING

    def test_warning_mid_band(self):
        recs = _rule_efficiency_loss(make_prediction(loss=15.0))
        assert len(recs) == 1
        assert recs[0].severity == Severity.WARNING

    def test_warning_just_below_critical_threshold(self):
        recs = _rule_efficiency_loss(make_prediction(loss=CRIT_LOSS - 0.01))
        assert len(recs) == 1
        assert recs[0].severity == Severity.WARNING

    def test_critical_exactly_at_critical_threshold(self):
        recs = _rule_efficiency_loss(make_prediction(loss=CRIT_LOSS))
        assert len(recs) == 1
        assert recs[0].severity == Severity.CRITICAL
        assert recs[0].action == "Schedule emergency maintenance"
        assert f"{CRIT_LOSS:.0f}%" in recs[0].message

    def test_critical_just_above_critical_threshold(self):
        recs = _rule_efficiency_loss(make_prediction(loss=CRIT_LOSS + 0.01))
        assert len(recs) == 1
        assert recs[0].severity == Severity.CRITICAL

    def test_critical_at_max_loss(self):
        recs = _rule_efficiency_loss(make_prediction(loss=100.0))
        assert len(recs) == 1
        assert recs[0].severity == Severity.CRITICAL

    def test_warning_and_critical_messages_contain_embedded_loss(self):
        msg_w = _rule_efficiency_loss(make_prediction(loss=12.34))[0].message
        assert "12.3%" in msg_w
        msg_c = _rule_efficiency_loss(make_prediction(loss=25.67))[0].message
        assert "25.7%" in msg_c

    # via the public entry point
    def test_warning_via_generate(self):
        rep = generate_recommendations(
            make_classification(), make_physics(), make_prediction(loss=15.0))
        assert rep.overall_severity == Severity.WARNING
        assert "Maintenance recommended within 7 days." in rep.summary

    def test_critical_via_generate(self):
        rep = generate_recommendations(
            make_classification(), make_physics(), make_prediction(loss=25.0))
        assert rep.overall_severity == Severity.CRITICAL


# ---------------------------------------------------------------------------
# 2. Fault-class rule (panel condition)
# ---------------------------------------------------------------------------


class TestFaultClassRule:
    """_rule_fault_class: every label handled by the implementation."""

    @pytest.mark.parametrize("label", list(EXPECTED_FAULT_MESSAGES.keys()))
    def test_known_label_maps_to_expected_severity(self, label):
        (exp_sev, msg_pin, action_pin) = EXPECTED_FAULT_MESSAGES[label]
        recs = _rule_fault_class(make_classification(label=label))
        assert len(recs) == 1
        assert recs[0].severity == exp_sev
        assert msg_pin in recs[0].message
        assert action_pin in recs[0].action

    @pytest.mark.parametrize("label", list(EXPECTED_FAULT_MESSAGES.keys()))
    def test_known_label_mapping_via_public_rules(self, label):
        # Through generate_recommendations, the isolated fault-class rule
        # reproduces the same severity (Clean physics/efficiency, no other recs).
        rep = generate_recommendations(
            make_classification(label=label),
            make_physics(),
            make_prediction(loss=2.0),
        )
        assert rep.overall_severity == EXPECTED_FAULT_MESSAGES[label][0]

    def test_unknown_label_falls_back_to_info(self):
        recs = _rule_fault_class(make_classification(label="Overheating"))
        assert len(recs) == 1
        assert recs[0].severity == Severity.INFO
        assert "Overheating" in recs[0].message
        assert recs[0].action == "Perform visual inspection"

    def test_unknown_variant_unknown_string(self):
        recs = _rule_fault_class(make_classification(label="Some-Random-Fault"))
        assert len(recs) == 1
        assert recs[0].severity == Severity.INFO
        assert recs[0].message == (
            "Fault class 'Some-Random-Fault' detected. "
            "Manual inspection recommended."
        )
        assert recs[0].action == "Perform visual inspection"


# ---------------------------------------------------------------------------
# 3. Module-temperature rule (boundary)
# ---------------------------------------------------------------------------


class TestModuleTempRule:
    """_rule_module_temperature: WARNING when module_temp >= HOT_TEMP."""

    def test_none_below_threshold(self):
        assert _rule_module_temperature(make_physics(module_temp=50.0)) == []

    def test_none_just_below(self):
        assert _rule_module_temperature(make_physics(module_temp=HOT_TEMP - 0.01)) == []

    def test_warning_exactly_at_threshold(self):
        recs = _rule_module_temperature(make_physics(module_temp=HOT_TEMP))
        assert len(recs) == 1
        assert recs[0].severity == Severity.WARNING
        assert recs[0].action == (
            "Check for shading, soiling, or ventilation obstructions")
        assert f"{HOT_TEMP:.0f}" in recs[0].message

    def test_warning_just_above(self):
        recs = _rule_module_temperature(make_physics(module_temp=HOT_TEMP + 0.01))
        assert len(recs) == 1
        assert recs[0].severity == Severity.WARNING

    def test_warning_well_above(self):
        recs = _rule_module_temperature(make_physics(module_temp=80.0))
        assert len(recs) == 1
        assert recs[0].severity == Severity.WARNING
        assert "80.0" in recs[0].message

    def test_via_generate(self):
        rep = generate_recommendations(
            make_classification(),
            make_physics(module_temp=HOT_TEMP + 1.0),
            make_prediction(),
        )
        assert rep.overall_severity == Severity.WARNING


# ---------------------------------------------------------------------------
# 4. Low-irradiance rule (boundary)
# ---------------------------------------------------------------------------


class TestLowIrradianceRule:
    """_rule_low_irradiance: INFO when irradiance < 50.0 (hard-coded)."""

    def test_none_above_threshold(self):
        assert _rule_low_irradiance(make_physics(irradiance=100.0)) == []

    def test_none_just_above(self):
        assert _rule_low_irradiance(make_physics(irradiance=LOW_IRR + 0.01)) == []

    def test_none_exactly_at_threshold(self):
        # '< 50.0' is strictly exclusive, so exactly 50.0 -> no recommendation.
        assert _rule_low_irradiance(make_physics(irradiance=LOW_IRR)) == []

    def test_info_just_below(self):
        recs = _rule_low_irradiance(make_physics(irradiance=LOW_IRR - 0.01))
        assert len(recs) == 1
        assert recs[0].severity == Severity.INFO
        assert recs[0].action == "Re-run analysis during daylight hours"

    def test_info_at_night(self):
        recs = _rule_low_irradiance(make_physics(irradiance=0.0))
        assert len(recs) == 1
        assert recs[0].severity == Severity.INFO

    def test_via_generate(self):
        rep = generate_recommendations(
            make_classification(),
            make_physics(irradiance=10.0),
            make_prediction(),
        )
        assert rep.overall_severity == Severity.INFO
        assert "advisory" in rep.summary.lower()


# ---------------------------------------------------------------------------
# 5. Severity ordering helper
# ---------------------------------------------------------------------------


class TestSeverityHelpers:
    """_SEVERITY_ORDER and _max_severity."""

    def test_severity_order_monotonic(self):
        assert _SEVERITY_ORDER[Severity.CRITICAL] == 3
        assert _SEVERITY_ORDER[Severity.WARNING] == 2
        assert _SEVERITY_ORDER[Severity.INFO] == 1
        assert _SEVERITY_ORDER[Severity.OK] == 0

    def test_max_severity_symmetry_and_ties(self):
        assert _max_severity(Severity.OK, Severity.INFO) == Severity.INFO
        assert _max_severity(Severity.INFO, Severity.OK) == Severity.INFO
        assert _max_severity(Severity.WARNING, Severity.INFO) == Severity.WARNING
        assert _max_severity(Severity.INFO, Severity.WARNING) == Severity.WARNING
        assert _max_severity(Severity.CRITICAL, Severity.WARNING) == Severity.CRITICAL
        assert _max_severity(Severity.WARNING, Severity.CRITICAL) == Severity.CRITICAL
        assert _max_severity(Severity.OK, Severity.OK) == Severity.OK
        assert _max_severity(Severity.WARNING, Severity.WARNING) == Severity.WARNING


# ---------------------------------------------------------------------------
# 6. Normal / healthy condition
# ---------------------------------------------------------------------------


class TestHealthyCondition:
    """Clean panel + low loss + comfortable temperature + good irradiance."""

    def test_single_ok_recommendation(self):
        rep = generate_recommendations(
            make_classification(label="Clean"),
            make_physics(),
            make_prediction(loss=2.0),
        )
        assert rep.overall_severity == Severity.OK
        assert len(rep.recommendations) == 1
        assert rep.recommendations[0].severity == Severity.OK
        assert "operating within normal parameters" in rep.summary

    def test_ok_summary_exact(self):
        rep = generate_recommendations(
            make_classification(), make_physics(), make_prediction())
        assert "\"[checkmark]\"" not in rep.summary  # not a literal marker
        assert "normal parameters" in rep.summary

# ---------------------------------------------------------------------------
# 7. Combined conditions (multiple rules firing at once)
# ---------------------------------------------------------------------------


class TestCombinedConditions:
    """Multiple rules firing simultaneously: ordering and overall severity."""

    def test_dusty_plus_warning_loss(self):
        # Dusty (WARNING) + 15% loss (WARNING): the stable sort keeps rule
        # order for equal severities, so the fault-class rec comes first.
        rep = generate_recommendations(
            make_classification(label="Dusty"),
            make_physics(),
            make_prediction(loss=15.0),
        )
        assert rep.overall_severity == Severity.WARNING
        assert len(rep.recommendations) == 2
        assert [r.action for r in rep.recommendations] == [
            "Schedule surface cleaning",
            "Schedule maintenance within 7 days",
        ]

    def test_dusty_plus_critical_loss(self):
        # CRITICAL (loss) outranks WARNING (Dusty) -> loss rec first.
        rep = generate_recommendations(
            make_classification(label="Dusty"),
            make_physics(),
            make_prediction(loss=25.0),
        )
        assert rep.overall_severity == Severity.CRITICAL
        assert len(rep.recommendations) == 2
        assert rep.recommendations[0].severity == Severity.CRITICAL
        assert rep.recommendations[0].action == "Schedule emergency maintenance"
        assert rep.summary == "⚠️ Critical issues detected — immediate action required."

    def test_damage_plus_critical_loss_both_critical(self):
        # Two CRITICAL recs: the stable sort keeps rule order (fault first).
        rep = generate_recommendations(
            make_classification(label="Physical-Damage"),
            make_physics(),
            make_prediction(loss=30.0),
        )
        assert rep.overall_severity == Severity.CRITICAL
        assert [r.action for r in rep.recommendations] == [
            "Replace the damaged panel",
            "Schedule emergency maintenance",
        ]

    def test_hotspot_plus_hot_temperature(self):
        rep = generate_recommendations(
            make_classification(label="Hotspot"),
            make_physics(module_temp=HOT_TEMP + 1.0),
            make_prediction(),
        )
        assert rep.overall_severity == Severity.CRITICAL
        assert [r.severity for r in rep.recommendations] == [
            Severity.CRITICAL, Severity.WARNING,
        ]

    def test_clean_plus_low_irradiance_advisory(self):
        # INFO (low irradiance) outranks the OK (Clean) recommendation.
        rep = generate_recommendations(
            make_classification(label="Clean"),
            make_physics(irradiance=10.0),
            make_prediction(),
        )
        assert rep.overall_severity == Severity.INFO
        assert len(rep.recommendations) == 2
        assert rep.recommendations[0].severity == Severity.INFO
        assert rep.recommendations[1].severity == Severity.OK
        assert rep.recommendations[0].action == "Re-run analysis during daylight hours"
        assert rep.summary == "ℹ️ Minor advisory — re-inspect during next scheduled visit."

    def test_dusty_plus_hot_and_low_irradiance(self):
        # WARNING + WARNING + INFO: overall WARNING, rule order within tiers.
        rep = generate_recommendations(
            make_classification(label="Dusty"),
            make_physics(module_temp=HOT_TEMP + 1.0, irradiance=10.0),
            make_prediction(),
        )
        assert rep.overall_severity == Severity.WARNING
        assert [r.action for r in rep.recommendations] == [
            "Schedule surface cleaning",
            "Check for shading, soiling, or ventilation obstructions",
            "Re-run analysis during daylight hours",
        ]
        assert rep.recommendations[-1].severity == Severity.INFO


# ---------------------------------------------------------------------------
# 8. Recommendation / report serialisation (to_dict)
# ---------------------------------------------------------------------------


class TestRecommendationDict:
    """Recommendation.to_dict(): plain dict with a string severity value."""

    def test_recommendation_to_dict(self):
        rec = Recommendation(
            severity=Severity.WARNING,
            message="Dust accumulation detected. Cleaning will restore output.",
            action="Schedule surface cleaning",
        )
        assert rec.to_dict() == {
            "severity": "WARNING",
            "message": "Dust accumulation detected. Cleaning will restore output.",
            "action": "Schedule surface cleaning",
        }

    def test_severity_str_enum_value(self):
        assert Severity.CRITICAL == "CRITICAL"  # str,Enum equality
        assert Severity.WARNING.value == "WARNING"
        assert Severity.INFO.value == "INFO"
        assert Severity.OK.value == "OK"
        assert list(Severity) == [
            Severity.CRITICAL, Severity.WARNING, Severity.INFO, Severity.OK,
        ]


class TestReportDict:
    """RecommendationReport.to_dict(): status / summary / issues /
    recommendation / priority."""

    def test_report_to_dict_healthy(self):
        rep = generate_recommendations(
            make_classification(), make_physics(), make_prediction())
        d = rep.to_dict()
        assert d["status"] == "OK"
        assert d["status"] in SEVERITIES
        assert d["summary"] == "✅ Panel is operating within normal parameters."
        assert len(d["issues"]) == 1
        assert set(d["issues"][0]) == {"severity", "message", "action"}
        assert d["recommendation"] == "Continue regular inspection schedule"
        assert d["priority"] == d["status"] == "OK"

    def test_report_to_dict_warning(self):
        rep = generate_recommendations(
            make_classification(label="Dusty"),
            make_physics(),
            make_prediction(loss=15.0),
        )
        d = rep.to_dict()
        assert d["status"] == "WARNING"
        assert d["priority"] == "WARNING"
        assert len(d["issues"]) == 2
        # Both recs are WARNING; the stable sort keeps the fault-class rec first.
        assert d["issues"][0]["action"] == "Schedule surface cleaning"
        assert d["recommendation"] == "Schedule surface cleaning"

    def test_report_to_dict_critical_top_action_is_fault_rec(self):
        # Both recs are CRITICAL; the stable sort keeps the fault-class rec
        # first, so the report-level "recommendation" is its action.
        rep = generate_recommendations(
            make_classification(label="Electrical-Damage"),
            make_physics(),
            make_prediction(loss=25.0),
        )
        d = rep.to_dict()
        assert d["status"] == "CRITICAL"
        assert d["priority"] == "CRITICAL"
        assert d["issues"][0]["action"] == (
            "Disconnect panel and contact a certified electrician immediately")
        assert d["recommendation"] == (
            "Disconnect panel and contact a certified electrician immediately")

    def test_empty_report_defaults(self):
        rep = RecommendationReport()
        assert rep.recommendations == []
        assert rep.overall_severity == Severity.OK
        assert rep.summary == "No issues detected."
        assert rep.to_dict() == {
            "status": "OK",
            "summary": "No issues detected.",
            "issues": [],
            "recommendation": "No action required",
            "priority": "OK",
        }


# ---------------------------------------------------------------------------
# 9. Priority ordering and determinism
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    """Recommendations are ordered CRITICAL > WARNING > INFO > OK."""

    def test_all_tiers_descending(self):
        rep = generate_recommendations(
            make_classification(label="Hotspot"),      # CRITICAL
            make_physics(module_temp=HOT_TEMP + 1.0),  # WARNING
            make_prediction(loss=30.0),                # CRITICAL
        )
        order = [_SEVERITY_ORDER[r.severity] for r in rep.recommendations]
        assert order == sorted(order, reverse=True)

    def test_priority_aliases_status_in_dict(self):
        scenarios = [
            ("Clean", 2.0, make_physics()),
            ("Dusty", 15.0, make_physics()),
            ("Physical-Damage", 30.0, make_physics()),
        ]
        for label, loss, physics in scenarios:
            d = generate_recommendations(
                make_classification(label=label), physics,
                make_prediction(loss=loss)).to_dict()
            assert d["priority"] == d["status"]

    def test_report_summary_matches_severity(self):
        scenarios = {
            Severity.CRITICAL: (
                make_classification(label="Electrical-Damage"),
                make_physics(), make_prediction(loss=25.0),
                "Critical issues detected"),
            Severity.WARNING: (
                make_classification(label="Dusty"),
                make_physics(), make_prediction(loss=15.0),
                "Maintenance recommended within 7 days"),
            Severity.INFO: (
                make_classification(label="Clean"),
                make_physics(irradiance=10.0), make_prediction(),
                "Minor advisory"),
            Severity.OK: (
                make_classification(label="Clean"),
                make_physics(), make_prediction(loss=2.0),
                "normal parameters"),
        }
        for sev, (clf, phys, pred, pin) in scenarios.items():
            rep = generate_recommendations(clf, phys, pred)
            assert rep.overall_severity == sev
            assert pin in rep.summary


class TestDeterminism:
    """Repeated runs with identical inputs produce identical reports."""

    def test_repeated_generation_is_identical(self):
        args = (
            make_classification(label="Dusty"),
            make_physics(module_temp=HOT_TEMP + 1.0),
            make_prediction(loss=15.0),
        )
        first = generate_recommendations(*args).to_dict()
        for _ in range(5):
            assert generate_recommendations(*args).to_dict() == first

    def test_stable_sort_preserves_rule_order_for_equal_severity(self):
        # Clean + warning loss + hot module: the two WARNING recs keep rule
        # order (efficiency before module temperature) under the stable sort;
        # the OK (Clean) rec sorts last.
        rep = generate_recommendations(
            make_classification(label="Clean"),
            make_physics(module_temp=HOT_TEMP + 1.0),
            make_prediction(loss=15.0),
        )
        assert [r.action for r in rep.recommendations] == [
            "Schedule maintenance within 7 days",
            "Check for shading, soiling, or ventilation obstructions",
            "Continue regular inspection schedule",
        ]


