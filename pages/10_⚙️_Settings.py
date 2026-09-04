"""pages/10_⚙️_Settings.py — Access control, data/privacy facts, and inference
configuration for this deployment.

Detailed per-model artifact status/hashes live on the **Model Status** page;
this page covers access control, privacy, and the raw configured thresholds.
Nothing here is aspirational — every value is read live from
configs/settings.yaml or models.model_manager.
"""
from __future__ import annotations

import streamlit as st

from utils.auth import require_access
from utils.config import CFG, get_secret
from utils.ui_theme import apply_page_chrome

apply_page_chrome("Settings")
require_access()

st.title("⚙️ Settings")
st.caption("Per-model artifact status, hashes, and class coverage: see the **Model Status** page.")

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
        f"num_classes (production) = {CFG['models']['mobilenet']['num_classes']}\n"
        f"input_size                = {CFG['models']['mobilenet']['input_size']}\n"
        f"production classes        = {', '.join(CFG['classification']['labels'])}\n"
        f"interim classes           = {', '.join(CFG['models']['mobilenet'].get('interim_labels') or [])}",
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
    "- Each completed inspection's **structured results** (detected faults per panel, "
    "confidences, efficiency estimates, environmental readings) are saved locally to "
    "`data/inspections.db` so the Overview/History/Analytics/Alerts pages have real "
    "data to show.\n"
    "- The **uploaded image itself is never stored** — only a SHA-256 hash of it, for "
    "identity purposes.\n"
    "- This database is local to the deployment (SQLite file), not sent to any external "
    "service.\n"
    "- Weather lookups are sent to OpenWeatherMap using the city you enter; no other "
    "personal data leaves this application."
)

st.subheader("Storage policy")
st.caption(
    "Large local Solar AI data (training datasets, model checkpoints, Kaggle staging) "
    "lives under `E:\\Solar AI Training Images\\` on the development machine — see "
    "`training/cloud/base/storage_paths.py` and `training/cloud/README.md`. This has no "
    "effect on the deployed application itself, which only reads the small artifacts "
    "already placed under `weights/`."
)

st.subheader("About")
st.caption(
    "Solar AI Framework — YOLO panel detection → MobileNetV2 fault classification → "
    "XGBoost efficiency-loss prediction. See the README for architecture, training "
    "pipelines, and known limitations (also summarized on the **Limitations** page)."
)
