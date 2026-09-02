"""utils/ui_helpers.py — Streamlit UI helper functions.

Responsibility
--------------
- Custom theme / CSS injection (visual only — no data or business logic)
- Result rendering
- Status cards
- Recommendation formatting
- Progress display
- Streamlit helper components
"""

import streamlit as st
from services.pipeline import PipelineResult

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

# Severity -> (accent colour, icon, display label). Mirrors the CRITICAL /
# WARNING / INFO / OK vocabulary already produced by
# services.recommendation.Severity — used purely to colour real values that
# the pipeline already computed, never to invent new judgements.
_SEVERITY_STYLE: dict[str, tuple[str, str, str]] = {
    "CRITICAL": ("#EF4444", "🔴", "Critical"),
    "WARNING":  ("#F59E0B", "🟠", "Warning"),
    "INFO":     ("#38BDF8", "🔵", "Info"),
    "OK":       ("#22C55E", "🟢", "Normal"),
}

# Fault label -> severity bucket, for the classification badge only. This is
# a display-only convenience that mirrors the same categorisation already
# encoded in services.recommendation._rule_fault_class; it does not change
# what label/confidence is shown, only how it is colour-coded.
_FAULT_SEVERITY: dict[str, str] = {
    "Clean": "OK",
    "Dusty": "WARNING",
    "Bird-Drop": "WARNING",
    "Electrical-Damage": "CRITICAL",
    "Physical-Damage": "CRITICAL",
    "Hotspot": "CRITICAL",
}

_CUSTOM_CSS = """
/* Subtle photovoltaic grid pattern behind the main canvas */
[data-testid="stAppViewContainer"] > .main {
    background-image:
        linear-gradient(rgba(245, 166, 35, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(245, 166, 35, 0.035) 1px, transparent 1px);
    background-size: 42px 42px;
}

/* Hero banner */
.solar-hero {
    background: linear-gradient(135deg, #1c2b3a 0%, #101826 65%, #0b1119 100%);
    border: 1px solid rgba(245, 166, 35, 0.35);
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.solar-hero::before {
    content: "";
    position: absolute;
    top: -60%;
    right: -8%;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, rgba(245,166,35,0.35) 0%, rgba(245,166,35,0) 70%);
    pointer-events: none;
}
.solar-hero h1 {
    margin: 0 0 0.3rem 0;
    font-size: 1.9rem;
    color: #F5A623;
}
.solar-hero p {
    margin: 0;
    color: #B9C4D0;
    font-size: 0.98rem;
    position: relative;
}

/* Sun-ray accent on section subheaders */
[data-testid="stMarkdownContainer"] h3 {
    border-left: 4px solid #F5A623;
    padding-left: 0.6rem;
    margin-top: 0.1rem;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(245, 166, 35, 0.18);
    border-top: 3px solid #F5A623;
    border-radius: 10px;
    padding: 0.85rem 1rem 0.55rem 1rem;
}
div[data-testid="stMetricLabel"] { opacity: 0.75; }

/* Bordered section containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
}

/* Fault-severity / status badges */
.solar-badge {
    display: inline-block;
    padding: 0.15rem 0.65rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* Overall status banner */
.status-banner {
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 1rem;
    border-left: 5px solid var(--sb-color, #F5A623);
    background: rgba(255, 255, 255, 0.03);
    font-size: 1.02rem;
}

/* Sidebar accent */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(245, 166, 35, 0.15);
}
"""


def apply_custom_theme() -> None:
    """Inject the solar-panel visual theme (CSS only; no data/logic changes).

    Complements the Streamlit ``[theme]`` section in ``.streamlit/config.toml``
    with accents (grid backdrop, metric cards, section framing, badges) that
    Streamlit's native theming cannot express on its own.
    """
    st.markdown(f"<style>{_CUSTOM_CSS}</style>", unsafe_allow_html=True)


def _badge_html(severity: str) -> str:
    """Return an inline-styled pill badge for a CRITICAL/WARNING/INFO/OK value."""
    color, icon, label = _SEVERITY_STYLE.get(severity, ("#94A3B8", "⚪", severity))
    return (
        f'<span class="solar-badge" '
        f'style="background:{color}26;color:{color};border:1px solid {color};">'
        f"{icon} {label}</span>"
    )


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

def display_results(result: PipelineResult) -> None:
    """Render all pipeline results to the Streamlit UI."""
    _display_status_banner(result)
    _display_detection(result)
    _display_classification(result)
    _display_weather_physics(result)
    _display_prediction(result)
    _display_recommendations(result)


def _display_status_banner(result: PipelineResult) -> None:
    """Show an at-a-glance severity banner ahead of the detailed sections."""
    report_dict: dict = result.recommendations.to_dict()
    color, _icon, _label = _SEVERITY_STYLE.get(
        report_dict["status"], ("#94A3B8", "⚪", report_dict["status"])
    )
    st.markdown(
        f'<div class="status-banner" style="--sb-color:{color};">'
        f'{_badge_html(report_dict["status"])} &nbsp; {report_dict["summary"]}'
        f"</div>",
        unsafe_allow_html=True,
    )


def _display_detection(result: PipelineResult) -> None:
    """Show YOLO detection metrics."""
    det = result.detection_result
    with st.container(border=True):
        st.subheader("🔍 Panel Detection")
        col1, col2 = st.columns(2)
        col1.metric("Panels Detected", det.panel_count)
        col2.metric("Best Confidence", f"{det.best_confidence:.1%}")
        if not det.detection_successful:
            st.warning("No solar panels detected in the uploaded image.")


def _display_classification(result: PipelineResult) -> None:
    """Show MobileNet fault classification."""
    clf = result.classification_result
    with st.container(border=True):
        st.subheader("🏷️ Fault Classification")
        col1, col2 = st.columns(2)
        col1.metric("Fault Type", clf.label)
        col2.metric("Confidence", f"{clf.confidence:.1%}")
        st.markdown(
            _badge_html(_FAULT_SEVERITY.get(clf.label, "INFO")),
            unsafe_allow_html=True,
        )
        if clf.probabilities:
            st.caption("Class probability distribution")
            st.bar_chart(clf.probabilities)
        if not clf.classification_successful:
            st.warning("Classification did not complete successfully.")


def _display_weather_physics(result: PipelineResult) -> None:
    """Show weather conditions and physics analysis."""
    wth, phy = result.weather_data, result.physics_data
    with st.container(border=True):
        st.subheader("🌤️ Environmental Conditions")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temperature", f"{wth.ambient_temp_c:.1f} °C")
        c2.metric("Humidity", f"{wth.humidity_pct:.0f} %")
        c3.metric("Wind Speed", f"{wth.wind_speed_ms:.1f} m/s")
        c4.metric("Cloud Cover", f"{wth.cloud_cover_pct:.0f} %")

        st.divider()

        st.subheader("⚡ Physics Analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Irradiance", f"{phy.irradiance_wm2:.0f} W/m²")
        c2.metric("Module Temp", f"{phy.module_temp_c:.1f} °C")
        c3.metric("Soiling Ratio", f"{phy.soiling_ratio:.2f}")
        c4.metric("Temp Loss", f"{phy.temp_loss_pct:.1f} %")

        if not wth.fetch_successful:
            st.info("ℹ️ Weather API unavailable — physics computed with default values.")


def _display_prediction(result: PipelineResult) -> None:
    """Show XGBoost energy prediction."""
    pred = result.efficiency_prediction
    with st.container(border=True):
        st.subheader("📈 Energy Output Prediction")
        col1, col2 = st.columns(2)
        col1.metric("Efficiency Loss", f"{pred.efficiency_loss_pct:.1f} %")
        col2.metric("Estimated Output", f"{pred.estimated_output_w:.0f} W")


def _display_recommendations(result: PipelineResult) -> None:
    """Render recommendations from the structured report dict."""
    report_dict: dict = result.recommendations.to_dict()
    with st.container(border=True):
        st.subheader("🔧 Maintenance Recommendations")
        st.markdown(f"**{report_dict['summary']}**")

        sev_colour = {
            "CRITICAL": "error",
            "WARNING":  "warning",
            "INFO":     "info",
            "OK":       "success",
        }
        for issue in report_dict["issues"]:
            fn = getattr(st, sev_colour.get(issue["severity"], "info"))
            fn(
                f"**[{issue['severity']}]** {issue['message']}  \n"
                f"*Action: {issue['action']}*"
            )
