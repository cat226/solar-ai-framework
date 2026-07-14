"""app.py — Solar AI Framework: Streamlit user interface.

Responsibilities (UI only)
--------------------------
- Render the Streamlit page layout and inputs.
- Accept a user-uploaded solar panel image plus supplementary panel inputs.
- Call :func:`services.pipeline.run_pipeline`.
- Display results via :mod:`utils.ui_helpers`.

This file must remain under 120-150 lines.
"""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

from services.pipeline import PipelineResult, run_pipeline
from utils.config import CFG
from utils.logger import get_logger
from utils.ui_helpers import display_results

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Solar AI Framework",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
def _render_sidebar() -> tuple:
    """Render sidebar inputs and return all pipeline parameters."""
    st.sidebar.title("☀️ Solar AI Framework")
    st.sidebar.markdown("Upload a solar panel image to begin analysis.")

    uploaded_file = st.sidebar.file_uploader(
        "Solar Panel Image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear photo of the solar panel surface.",
    )

    city = st.sidebar.text_input(
        "Location (City)",
        value=CFG["weather"]["default_city"],
        help="Used to fetch live weather data from OpenWeatherMap.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Panel Details")

    panel_age = st.sidebar.number_input(
        "Panel Age (years)", min_value=0.0, max_value=40.0,
        value=0.0, step=0.5,
    )
    maintenance_count = st.sidebar.number_input(
        "Prior Maintenance Events", min_value=0, max_value=50,
        value=0, step=1,
    )
    voltage = st.sidebar.number_input(
        "Measured Voltage (V)", min_value=0.0, max_value=1000.0,
        value=0.0, step=0.1,
    )
    current = st.sidebar.number_input(
        "Measured Current (A)", min_value=0.0, max_value=100.0,
        value=0.0, step=0.1,
    )
    installation_type = st.sidebar.selectbox(
        "Installation Type",
        options=["rooftop", "ground-mount", "carport", "floating"],
    )

    pil_image: Image.Image | None = None
    if uploaded_file is not None:
        pil_image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

    return pil_image, city, float(panel_age), int(maintenance_count), \
        float(voltage), float(current), str(installation_type)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Main Streamlit entry point."""
    st.title("☀️ Solar AI Framework")
    st.caption("Automated solar panel fault detection and energy output prediction.")

    pil_image, city, panel_age, maintenance_count, \
        voltage, current, installation_type = _render_sidebar()

    if pil_image is None:
        st.info("Upload a solar panel image in the sidebar to start analysis.")
        return

    st.image(pil_image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running analysis pipeline…"):
        result: PipelineResult = run_pipeline(
            image=pil_image,
            city=city,
            panel_age=panel_age,
            maintenance_count=maintenance_count,
            voltage=voltage,
            current=current,
            installation_type=installation_type,
        )
        logger.info(
            "Pipeline returned status=%s for city='%s'.", result.status, city
        )

    if result.status == "ERROR":
        st.error(f"Pipeline error [{result.error_type}]: {result.error_message}")
        return

    st.success(f"Pipeline completed in {result.processing_time:.2f}s")
    display_results(result)


if __name__ == "__main__":
    main()
