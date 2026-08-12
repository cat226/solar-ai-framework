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
import re

import streamlit as st
from PIL import Image, UnidentifiedImageError

from services.pipeline import PipelineResult, run_pipeline
from utils.config import CFG
from utils.logger import get_logger
from utils.security import sanitize_for_log
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


def _sanitize_city(value: str) -> str:
    """Normalize user-supplied city text before API calls and log messages.

    Limits input length and removes control characters so untrusted text cannot
    create oversized requests or inject misleading multiline log records.
    """
    cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", str(value))
    return " ".join(cleaned.split())[:100]


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
def _render_sidebar() -> tuple[Image.Image | None, str, float, int, float, float, str]:
    """Render sidebar inputs and return all pipeline parameters."""
    st.sidebar.title("☀️ Solar AI Framework")
    st.sidebar.markdown("Upload a solar panel image to begin analysis.")

    uploaded_file = st.sidebar.file_uploader(
        "Solar Panel Image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear photo of the solar panel surface (max 10 MB).",
    )

    city = st.sidebar.text_input(
        "Location (City)",
        value=CFG["weather"]["default_city"],
        max_chars=100,
        help="Used to fetch live weather data from OpenWeatherMap.",
    )
    city = _sanitize_city(city)

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
        try:
            raw_bytes = uploaded_file.read()
            with Image.open(io.BytesIO(raw_bytes)) as image:
                image.verify()
            with Image.open(io.BytesIO(raw_bytes)) as image:
                pil_image = image.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            logger.warning("Rejected invalid uploaded image: %s", exc)
            st.sidebar.error(
                "The uploaded file is not a valid or safe image. "
                "Please upload a JPG, JPEG, PNG, or WebP image under 10 MB."
            )

    return pil_image, city, panel_age, maintenance_count, voltage, current, installation_type


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
        try:
            result: PipelineResult = run_pipeline(
                image=pil_image,
                city=city,
                panel_age=panel_age,
                maintenance_count=maintenance_count,
                voltage=voltage,
                current=current,
                installation_type=installation_type,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled pipeline exception in UI layer")
            st.error(
                "An unexpected error occurred while analysing the image. "
                "Please try again; if the problem persists, check the logs."
            )
            return
        logger.info(
            "Pipeline returned status=%s for city='%s'.",
            result.status,
            sanitize_for_log(city),
        )

    if result.status == "ERROR":
        st.error(f"Pipeline error [{result.error_type}]: {result.error_message}")
        return

    st.success(f"Pipeline completed in {result.processing_time:.2f}s")
    display_results(result)


if __name__ == "__main__":
    main()
