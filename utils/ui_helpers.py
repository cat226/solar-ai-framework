"""utils/ui_helpers.py — Streamlit UI helper functions.

Responsibility
--------------
- Result rendering
- Status cards
- Recommendation formatting
- Progress display
- Streamlit helper components
"""

import streamlit as st
from services.pipeline import PipelineResult

def display_results(result: PipelineResult) -> None:
    """Render all pipeline results to the Streamlit UI."""
    _display_detection(result)
    _display_classification(result)
    _display_weather_physics(result)
    _display_prediction(result)
    _display_recommendations(result)

def _display_detection(result: PipelineResult) -> None:
    """Show YOLO detection metrics."""
    det = result.detection_result
    st.subheader("🔍 Panel Detection")
    col1, col2 = st.columns(2)
    col1.metric("Panels Detected", det.panel_count)
    col2.metric("Best Confidence", f"{det.best_confidence:.1%}")
    if not det.detection_successful:
        st.warning("No solar panels detected in the uploaded image.")

def _display_classification(result: PipelineResult) -> None:
    """Show MobileNet fault classification."""
    clf = result.classification_result
    st.subheader("🏷️ Fault Classification")
    col1, col2 = st.columns(2)
    col1.metric("Fault Type", clf.label)
    col2.metric("Confidence", f"{clf.confidence:.1%}")
    if clf.probabilities:
        st.bar_chart(clf.probabilities)

def _display_weather_physics(result: PipelineResult) -> None:
    """Show weather conditions and physics analysis."""
    wth, phy = result.weather_data, result.physics_data
    st.subheader("🌤️ Environmental Conditions")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature", f"{wth.ambient_temp_c:.1f} °C")
    c2.metric("Humidity", f"{wth.humidity_pct:.0f} %")
    c3.metric("Wind Speed", f"{wth.wind_speed_ms:.1f} m/s")
    c4.metric("Cloud Cover", f"{wth.cloud_cover_pct:.0f} %")

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
    st.subheader("📈 Energy Output Prediction")
    col1, col2 = st.columns(2)
    col1.metric("Efficiency Loss", f"{pred.efficiency_loss_pct:.1f} %")
    col2.metric("Estimated Output", f"{pred.estimated_output_w:.0f} W")

def _display_recommendations(result: PipelineResult) -> None:
    """Render recommendations from the structured report dict."""
    report_dict: dict = result.recommendations.to_dict()
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
