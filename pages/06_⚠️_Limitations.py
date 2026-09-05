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
six_class_taxonomy = mn_status["six_class_labels"]
future_only = [c for c in six_class_taxonomy if c not in active]

st.subheader("Classifier coverage")
if mn_status["state"] == "six_class":
    st.success("✅ Full six-class classifier is active. No classes are unavailable.")
elif mn_status["state"] == "v1":
    st.success(
        f"**Supported now (v1):** solar-panel detection, {', '.join(sorted(active))}.\n\n"
        f"**Not supported in v1:** {', '.join(future_only)}."
    )
    for cls in future_only:
        st.markdown(f"- **{cls}** — not supported in v1")
    st.markdown(
        "This is v1's intentionally frozen scope, not an incomplete or broken system. "
        "The full six-class classifier (`weights/mobilenet_solar.pth`) remains a "
        "**documented future roadmap item** — see `training/classification/DATASET_SOURCES.md` "
        "for the exact provenance status of every class, including why the three classes "
        "above are not part of this release (no licensed/accessible dataset yet, or "
        "access pending the dataset owner's approval)."
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
    "efficiency/output figures are reported as unavailable, never estimated as zero.\n"
    "- **No XGBoost artifact currently exists, and none is planned until a genuine "
    "dataset is found.** Investigated on 2026-09-04: no dataset was found that pairs "
    "a real fault classification with paired environmental telemetry and a genuinely "
    "measured efficiency-loss value — see `training/prediction/DATASET_SOURCES.md`. "
    "This is a real, current limitation, not a placeholder pending routine training."
)
