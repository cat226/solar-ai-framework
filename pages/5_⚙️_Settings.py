"""pages/5_⚙️_Settings.py — Real system/model information and app preferences.

Shows the actual configured inference thresholds, artifact paths/status, and
data-handling facts about this deployment. Nothing here is aspirational —
every value is read live from configs/settings.yaml or models.model_manager.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from models.model_manager import model_manager
from utils.auth import require_access
from utils.config import CFG, get_secret
from utils.ui_theme import apply_page_chrome

apply_page_chrome("Settings")
require_access()

st.title("⚙️ Settings")

st.subheader("Model artifacts")
status = model_manager.artifact_status
rows = [
    {"Model": name, "Path": entry["path"], "Status": "✅ present" if entry["exists"] else "❌ missing"}
    for name, entry in status.items()
]
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.subheader("Inference configuration")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**YOLO detector**")
    st.code(
        f"confidence_threshold = {CFG['models']['yolo']['confidence_threshold']}\n"
        f"iou_threshold        = {CFG['models']['yolo']['iou_threshold']}\n"
        f"image_size            = {CFG['models']['yolo']['image_size']}",
        language="text",
    )
with c2:
    st.markdown("**MobileNet classifier**")
    st.code(
        f"num_classes = {CFG['models']['mobilenet']['num_classes']}\n"
        f"input_size  = {CFG['models']['mobilenet']['input_size']}\n"
        f"classes     = {', '.join(CFG['classification']['labels'])}",
        language="text",
    )

st.subheader("Access control")
if get_secret("APP_ACCESS_PASSWORD"):
    st.success(
        "A shared access password is configured for this deployment. "
        "This is a single shared password, not per-user accounts — see `utils/auth.py`."
    )
else:
    st.warning(
        "No access password is configured (`APP_ACCESS_PASSWORD` unset) — this "
        "deployment is open to anyone who can reach it. Set the secret to enable "
        "the access gate."
    )

st.subheader("Data & privacy")
st.markdown(
    "- Each completed inspection's **structured results** (detected fault, confidence, "
    "efficiency estimate, environmental readings) are saved locally to `data/inspections.db` "
    "so the Dashboard/History/Analytics/Alerts pages have real data to show.\n"
    "- The **uploaded image itself is never stored** — only a SHA-256 hash of it, for "
    "identity purposes.\n"
    "- This database is local to the deployment (SQLite file), not sent to any external "
    "service.\n"
    "- Weather lookups are sent to OpenWeatherMap using the city you enter; no other "
    "personal data leaves this application."
)

st.subheader("About")
st.caption(
    "Solar AI Framework — YOLO panel detection → MobileNetV2 fault classification → "
    "XGBoost efficiency-loss prediction. See the README for architecture, training "
    "pipelines, and known limitations."
)
