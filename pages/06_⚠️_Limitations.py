"""pages/06_⚠️_Limitations.py — Honest, explicit capability disclosure.

Static content cross-checked against the live model status below (the
class list is read from models.model_manager, not hardcoded twice) so this
page can never silently drift out of sync with what's actually running.
"""
from __future__ import annotations

import streamlit as st

from models.model_manager import model_manager
from utils.auth import require_access
from utils.ui_theme import apply_page_chrome

apply_page_chrome("Limitations")
require_access()

st.title("⚠️ Limitations")
st.caption("What this deployment can and cannot do right now — kept visible, not buried.")

mn_status = model_manager.mobilenet_status
active = set(mn_status["active_labels"])
production = mn_status["production_labels"]
unavailable = [c for c in production if c not in active]

st.subheader("Classifier coverage")
if mn_status["state"] == "production":
    st.success("✅ Full six-class production classifier is active. No classes are unavailable.")
elif mn_status["state"] == "interim":
    st.warning(
        f"⚠️ **Currently classifiable:** {', '.join(active)}.\n\n"
        f"❌ **Not currently available:** {', '.join(unavailable)}."
    )
    for cls in unavailable:
        st.markdown(f"- **{cls}** — unavailable")
    st.markdown(
        "The final six-class production model (`weights/mobilenet_solar.pth`) remains "
        "**planned**, not abandoned — see `training/classification/DATASET_SOURCES.md` "
        "for the exact provenance status of every class, including why the classes "
        "above are currently blocked (missing licensed/accessible datasets, or access "
        "pending owner approval)."
    )
else:
    st.error("❌ No classifier is active at all — classification cannot run.")

st.divider()

st.subheader("Other known limitations")
st.markdown(
    "- **Efficiency/output estimates depend on environmental inputs and model "
    "assumptions.** They are model predictions, not measured sensor readings — see "
    "the **Environment** page for the exact inputs used in your last inspection.\n"
    "- **Weather data falls back to configured defaults** when the OpenWeatherMap API "
    "is unavailable, rather than blocking the analysis — this is always disclosed "
    "on the Environment page when it happens.\n"
    "- **No sensor/telemetry integration.** All inputs come from the uploaded image "
    "plus a weather API lookup or manual entry — there is no live IoT/SCADA "
    "connection to a real installation.\n"
    "- **Single-image analysis.** Each inspection analyzes one uploaded photo; there "
    "is no temporal tracking of a specific physical panel across multiple "
    "inspections beyond what the History page's timestamps show.\n"
    "- **Efficiency-loss prediction requires the XGBoost artifact.** When it's "
    "absent (see **Model Status**), detection and classification still run, but "
    "efficiency/output figures are reported as unavailable, never estimated as zero."
)
