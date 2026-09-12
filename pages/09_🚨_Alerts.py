"""pages/09_🚨_Alerts.py — Real alerts derived from recorded Ground
Inspection history, plus live system-availability alerts derived from
actual model readiness.

No alert on this page is synthetic — each is either a stored Ground
Inspection run with at least one confidently-flagged non-CLEAN panel, or a
live model-missing condition checked directly against the weight files.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from services import storage
from utils.auth import require_access
from utils.ui_theme import apply_page_chrome, empty_state, severity_pill

apply_page_chrome("Alerts")
require_access()

st.title("🚨 Alerts")

# ---------------------------------------------------------------------------
# System-availability alerts (live, not stored — reflects current readiness)
# ---------------------------------------------------------------------------
st.subheader("System availability")

_GROUND_WEIGHTS = {
    "ground panel detector": Path("weights/yolo_solar_ground_research_v1.pt"),
    "condition classifier": Path("weights/mobilenet_condition_4class_research_v1.pth"),
    "condition detector": Path("weights/yolo_solar_condition_4class_research_v2.pt"),
}
ground_missing = [name for name, p in _GROUND_WEIGHTS.items() if not p.is_file()]

if ground_missing:
    st.error(
        f"**Ground Inspection unavailable**: {', '.join(ground_missing)} not present. "
        "The Ground Inspection page cannot run until genuine trained weights are supplied "
        "in `weights/`."
    )
else:
    st.success("Ground Inspection is fully ready — panel detector + condition classifier + condition detector all present.")

st.divider()

# ---------------------------------------------------------------------------
# Ground Inspection alerts (real stored history)
# ---------------------------------------------------------------------------
st.subheader("Ground Inspection alerts")
galerts = storage.get_ground_alerts(limit=100)
if not galerts:
    empty_state(
        "No Ground Inspection run has flagged a confident non-CLEAN panel yet. This is a "
        "real absence of alerts, not an unmonitored gap — every completed run is evaluated."
    )
else:
    for row in galerts:
        severity = "CRITICAL" if row["worst_label"] == "PHYSICAL_DAMAGE" else "WARNING"
        st.markdown(
            f"{severity_pill(severity)} &nbsp; **{row['worst_label']}** *(classifier, primary signal)* "
            f"— {row['created_at']}  \n"
            f"Panels: {row['panel_count']} · Flagged: {row['flagged_count']} · "
            f"Unread: {row['unread_count']} · [View in History →](/📜_History)",
            unsafe_allow_html=True,
        )
        st.divider()
