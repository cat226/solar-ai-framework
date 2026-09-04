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
from PIL import Image, ImageDraw

from services.pipeline import PipelineResult
from utils.image_utils import crop_panel, unletterbox_box
from utils.ui_theme import empty_state

def display_results(result: PipelineResult, source_image: Image.Image | None = None) -> None:
    """Render all pipeline results to the Streamlit UI.

    Args:
        result: Completed pipeline result.
        source_image: The original uploaded image, used to draw a real
                      bounding-box overlay for detected panels. When omitted,
                      the detection section falls back to metrics only.
    """
    _display_capability_notice(result)
    _display_detection(result, source_image)
    _display_panel_selector(result, source_image)
    _display_classification(result)
    _display_weather_physics(result)
    _display_prediction(result)
    _display_recommendations(result)


def _display_capability_notice(result: PipelineResult) -> None:
    """Honest, up-front disclosure of what this specific run could and
    couldn't do - shown before any results, not buried at the bottom."""
    if result.classifier_source == "interim":
        st.info(
            "ℹ️ **Classifier coverage: Clean / Dusty / Hotspot only.** This inspection "
            "used the interim MobileNet checkpoint - Bird-Drop, Electrical-Damage, and "
            "Physical-Damage are not yet classifiable (see the **Limitations** page). "
            "The final six-class production model is planned once those datasets are acquired."
        )
    elif result.classifier_source == "missing":
        st.warning(
            "⚠️ No MobileNet checkpoint (production or interim) is available - "
            "classification results below are not real."
        )
    if not result.xgboost_available:
        st.warning(
            "⚠️ **Efficiency/output predictions unavailable.** The XGBoost artifact "
            "(`weights/xgboost_solar.joblib`) is not present, so no efficiency-loss or "
            "power-output estimate was computed for this inspection - detection and "
            "classification results below are real and unaffected."
        )

def _draw_detection_overlay(image: Image.Image, det) -> Image.Image:
    """Draw real detection boxes (mapped out of letterboxed coordinates back
    onto the original image) with per-box confidence labels."""
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for i, (box, conf) in enumerate(zip(det.boxes, det.confidences), start=1):
        x1, y1, x2, y2 = unletterbox_box(tuple(box), image.size)
        draw.rectangle([x1, y1, x2, y2], outline=(217, 123, 41), width=3)
        label = f"#{i} {conf:.0%}"
        draw.rectangle([x1, max(0, y1 - 18), x1 + 8 * len(label), y1], fill=(217, 123, 41))
        draw.text((x1 + 2, max(0, y1 - 17)), label, fill=(255, 255, 255))
    return overlay

def _display_detection(result: PipelineResult, source_image: Image.Image | None) -> None:
    """Show YOLO detection metrics and, when available, a real bounding-box overlay."""
    det = result.detection_result
    st.subheader("🔍 Panel Detection")
    col1, col2 = st.columns(2)
    col1.metric("Panels Detected", det.panel_count)
    col2.metric("Best Confidence", f"{det.best_confidence:.1%}")
    if not det.detection_successful:
        st.warning("No solar panels detected in the uploaded image.")
    elif source_image is not None and det.boxes:
        overlay = _draw_detection_overlay(source_image, det)
        st.image(overlay, caption="Detected panels", use_container_width=True)

def _display_panel_selector(result: PipelineResult, source_image: Image.Image | None) -> None:
    """Let the user pick one detected panel and see its own crop,
    classification, and (when available) efficiency estimate - the real
    per-panel breakdown, not the whole-image result repeated per box.
    A full sortable table of every panel lives on the **Panel Results** page
    (st.session_state carries this same result there)."""
    if not result.panels:
        return
    st.subheader(f"🔎 Panel-by-Panel Detail ({len(result.panels)} panel(s))")
    options = [f"Panel #{p.panel_index} — {p.classification.label} ({p.detection_confidence:.0%} confidence)" for p in result.panels]
    idx = st.selectbox("Select a detected panel", range(len(options)), format_func=lambda i: options[i])
    panel = result.panels[idx]

    col_img, col_data = st.columns([1, 2])
    with col_img:
        if source_image is not None:
            st.image(crop_panel(source_image, tuple(panel.box)), caption=f"Panel #{panel.panel_index} crop", use_container_width=True)
    with col_data:
        c1, c2 = st.columns(2)
        c1.metric("Classification", panel.classification.label, f"{panel.classification.confidence:.0%} confidence")
        c2.metric("Detection confidence", f"{panel.detection_confidence:.0%}")
        if panel.prediction.prediction_successful:
            c3, c4 = st.columns(2)
            c3.metric("Efficiency loss (estimate)", f"{panel.prediction.efficiency_loss_pct:.1f}%")
            c4.metric("Estimated output (estimate)", f"{panel.prediction.estimated_output_w:.0f} W")
        else:
            st.caption("Efficiency/output estimate unavailable for this panel (XGBoost artifact not present).")


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
    """Show XGBoost energy prediction - or an honest unavailable state
    rather than a fabricated 0.0% loss / 0 W when the artifact is missing."""
    pred = result.efficiency_prediction
    st.subheader("📈 Energy Output Prediction (estimate)")
    if not result.xgboost_available:
        empty_state(
            "Unavailable — the XGBoost artifact is not present. This is not a real "
            "0% loss measurement; no prediction was computed.",
            icon="🚫",
        )
        return
    col1, col2 = st.columns(2)
    col1.metric("Efficiency Loss", f"{pred.efficiency_loss_pct:.1f} %")
    col2.metric("Estimated Output", f"{pred.estimated_output_w:.0f} W")
    st.caption("Model estimate, not a measured sensor reading.")

def _display_recommendations(result: PipelineResult) -> None:
    """Render recommendations from the structured report dict - or an
    honest unavailable state when there's no real prediction to base a
    recommendation on."""
    st.subheader("🔧 Maintenance Recommendations")
    if not result.xgboost_available:
        empty_state(
            "Unavailable — recommendations are derived from the efficiency prediction "
            "above, which did not run.",
            icon="🚫",
        )
        return

    report_dict: dict = result.recommendations.to_dict()
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
