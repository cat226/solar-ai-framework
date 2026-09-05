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

# Only YOLO/MobileNet being absent is a genuine, actionable fault - install
# real weights and it's resolved. XGBoost's absence is v1's documented,
# permanent capability boundary (no legitimate training dataset exists - see
# Limitations), not an operational problem to alert on the same way.
missing = [name for name in ("YOLO",) if not status[name]["exists"]]
if mn_status["state"] == "missing":
    missing.append("MobileNet")

if missing:
    st.error(
        f"**Model unavailable**: {', '.join(missing)} artifact(s) are not present. "
        "Analysis requiring these models cannot run until genuine trained weights "
        "are supplied in `weights/`."
    )
else:
    st.success(
        "Detection and classification are fully ready — v1 supports Clean, Dusty, "
        "and Hotspot. Bird-Drop, Electrical-Damage, and Physical-Damage remain a "
        "documented future expansion; see the **Limitations** page."
    )

if not status["XGBoost"]["exists"]:
    st.info(
        "ℹ️ Efficiency-loss/output-power prediction is unavailable in v1 — no "
        "legitimate training dataset was found (see **Limitations**). This is an "
        "expected v1 boundary, not a fault."
    )

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
