"""pages/09_🚨_Alerts.py — Real alerts derived from recorded inspection history,
plus live system-availability alerts derived from actual model readiness.

No alert on this page is synthetic — each is either a stored inspection
that crossed a real CRITICAL/WARNING severity threshold (computed by
services.recommendation from actual model output), or a live model-missing
condition reported by models.model_manager.
"""
from __future__ import annotations

from models.model_manager import model_manager
from services import storage
from utils.auth import require_access
from utils.ui_theme import apply_page_chrome, empty_state, severity_pill
import streamlit as st

apply_page_chrome("Alerts")
require_access()

st.title("🚨 Alerts")

# ---------------------------------------------------------------------------
# System-availability alerts (live, not stored — reflects current readiness)
# ---------------------------------------------------------------------------
st.subheader("System availability")
status = model_manager.artifact_status
mn_status = model_manager.mobilenet_status
missing = [name for name in ("YOLO", "XGBoost") if not status[name]["exists"]]
if mn_status["state"] == "missing":
    missing.append("MobileNet")

if missing:
    st.error(
        f"**Model unavailable**: {', '.join(missing)} artifact(s) are not present. "
        "Analysis requiring these models cannot run until genuine trained weights "
        "are supplied in `weights/`."
    )
elif mn_status["state"] == "interim":
    st.warning(
        "**MobileNet running on the interim checkpoint** — Bird-Drop, "
        "Electrical-Damage, and Physical-Damage are not currently classifiable. "
        "See the **Limitations** page."
    )
else:
    st.success("All model artifacts are present and inference is fully ready.")

st.divider()

# ---------------------------------------------------------------------------
# Inspection-derived alerts (real stored history)
# ---------------------------------------------------------------------------
st.subheader("Inspection alerts")
alerts = storage.get_alerts(limit=100)
if not alerts:
    empty_state(
        "No CRITICAL or WARNING inspections recorded. This is a real absence of "
        "alerts, not an unmonitored gap — every completed inspection is evaluated."
    )
else:
    for row in alerts:
        st.markdown(
            f"{severity_pill(row['severity'])} &nbsp; **{row['fault_label']}** "
            f"— {row['city']} — {row['created_at']}  \n"
            f"Panels: {row['panel_count']} · Efficiency loss: {row['efficiency_loss_pct']:.1f}% "
            f"· [View in History →](/📜_History)",
            unsafe_allow_html=True,
        )
        st.divider()
