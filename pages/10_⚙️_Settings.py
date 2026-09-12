"""pages/10_⚙️_Settings.py — Access control, data/privacy facts, and inference
configuration for this deployment.

Detailed per-model artifact status/hashes live on the **Model Status** page;
this page covers access control, privacy, and the raw configured thresholds.
Nothing here is aspirational — every value is read live or matches the
literal defaults in app.py's model-loading/inference code.
"""
from __future__ import annotations

import streamlit as st

from utils.auth import require_access
from utils.config import get_secret
from utils.ui_theme import apply_page_chrome

apply_page_chrome("Settings")
require_access()

st.title("⚙️ Settings")
st.caption("Per-model artifact status, hashes, and class coverage: see the **Model Status** page.")

st.subheader("Inference configuration")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Ground panel detector**")
    st.code(
        "confidence_threshold = 0.25 (adjustable via the sensitivity slider, 0.10-0.60)\n"
        "iou_threshold        = 0.50\n"
        "image_size            = 640",
        language="text",
    )
with c2:
    st.markdown("**Condition engines**")
    st.code(
        "classifier input_size        = 224\n"
        "classifier predicted threshold = 0.45\n"
        "detector detection floor      = 0.25\n"
        "detector predicted threshold  = 0.45\n"
        "active taxonomy               = CLEAN, DUSTY, BIRD_DROP, PHYSICAL_DAMAGE",
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
    "- Each completed Ground Inspection run's **structured results** (per-panel condition "
    "reads from both engines, confidences, detection geometry) are saved locally to "
    "`data/inspections.db` (`ground_inspections` table) so the Overview/History/Analytics/"
    "Alerts pages have real data to show.\n"
    "- The **uploaded image itself is never stored**.\n"
    "- This database is local to the deployment (SQLite file), not sent to any external "
    "service.\n"
    "- No other personal data leaves this application — Ground Inspection has no weather/"
    "location lookup."
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
    "Solar AI — ground-level, dynamic multi-panel inspection. Dynamic panel detector → "
    "per-panel crop → condition classifier (primary) + condition detector (secondary, "
    "localized cross-check). See the README for architecture, training pipelines, and "
    "known limitations (also summarized on the **Limitations** page)."
)
