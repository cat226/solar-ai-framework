"""pages/04_🌦️_Environment.py — Environmental inputs used by the predictor,
for the most recent live inspection.

Shows real weather-service output (or its documented fallback) and the
physics constants actually configured in configs/settings.yaml — never
invents an environmental value.
"""
from __future__ import annotations

import streamlit as st

from utils.auth import require_access
from utils.config import CFG
from utils.ui_theme import apply_page_chrome, empty_state

apply_page_chrome("Environment")
require_access()

st.title("🌦️ Environment / Physics")

result = st.session_state.get("last_result")
if result is None:
    empty_state("No environmental data yet. Run an inspection on the main **Inspect** page.")
else:
    wth, phy = result.weather_data, result.physics_data

    st.subheader("Environmental inputs")
    source = "API (OpenWeatherMap)" if wth.fetch_successful else "default (API unavailable)"
    st.markdown(f"**Source:** {source}")
    if not wth.fetch_successful:
        st.warning(
            "⚠️ The weather API was unavailable for this inspection — the values below "
            "are the configured defaults (`weather.defaults` in configs/settings.yaml), "
            "not a live observation."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Irradiance", f"{phy.irradiance_wm2:.0f} W/m²")
    c2.metric("Module temperature", f"{phy.module_temp_c:.1f} °C")
    c3.metric("Ambient temperature", f"{wth.ambient_temp_c:.1f} °C")
    c4.metric("Humidity", f"{wth.humidity_pct:.0f} %")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Wind speed", f"{wth.wind_speed_ms:.1f} m/s")
    c6.metric("Cloud cover", f"{wth.cloud_cover_pct:.0f} %")
    c7.metric("Soiling ratio", f"{phy.soiling_ratio:.2f}", help="Derived from the classified fault label, not a sensor reading.")
    c8.metric("Cloud factor", f"{phy.cloud_factor:.2f}")

st.divider()

# ---------------------------------------------------------------------------
# Configured physics constants — the actual values in use, not aspirational.
# ---------------------------------------------------------------------------
st.subheader("Configured physics constants")
st.caption("Read live from configs/settings.yaml — not hardcoded here.")
phys_cfg = CFG["physics"]
st.code(
    f"max_irradiance_wm2       = {phys_cfg['max_irradiance_wm2']}\n"
    f"irradiance_cloud_factor  = {phys_cfg['irradiance_cloud_factor']}  # legacy compatibility field\n"
    f"noct_celsius             = {phys_cfg['noct_celsius']}\n"
    f"noct_irradiance_ref      = {phys_cfg['noct_irradiance_ref']}\n"
    f"wind_cooling_coefficient = {phys_cfg['wind_cooling_coefficient']}\n"
    f"temp_coefficient_pmax    = {phys_cfg['temp_coefficient_pmax']}\n"
    f"panel_rated_power_wp     = {phys_cfg['panel_rated_power_wp']}",
    language="text",
)
st.subheader("Soiling ratio per class")
st.json(phys_cfg["soiling_ratios"])
