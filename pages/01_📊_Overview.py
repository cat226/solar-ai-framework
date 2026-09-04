"""pages/01_📊_Overview.py — Top-level KPIs and system status.

All figures come from services.storage (real recorded inspections) or
models.model_manager (real artifact readiness) — never fabricated
placeholder data. When no inspections have been recorded yet, the page
shows an explicit empty state instead of invented numbers.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from models.model_manager import model_manager
from services import storage
from utils.auth import require_access
from utils.ui_theme import apply_page_chrome, empty_state

apply_page_chrome("Overview")
require_access()

st.title("📊 Overview")
st.caption("Live overview, computed from this deployment's own recorded inspection history.")

# ---------------------------------------------------------------------------
# System status — real readiness, never shown as healthy if it isn't.
# ---------------------------------------------------------------------------
st.subheader("System status")
status = model_manager.artifact_status
mn_status = model_manager.mobilenet_status
cols = st.columns(4)
with cols[0]:
    st.success("**YOLO**\n\nready") if status["YOLO"]["exists"] else st.error("**YOLO**\n\nmissing")
with cols[1]:
    if mn_status["state"] == "six_class":
        st.success("**MobileNet**\n\nready (6-class)")
    elif mn_status["state"] == "v1":
        st.success("**MobileNet**\n\nready (v1, 3-class)")
    else:
        st.error("**MobileNet**\n\nmissing")
with cols[2]:
    st.success("**XGBoost**\n\nready") if status["XGBoost"]["exists"] else st.error("**XGBoost**\n\nmissing")
with cols[3]:
    core_ready = status["YOLO"]["exists"] and mn_status["state"] != "missing"
    if core_ready and status["XGBoost"]["exists"]:
        st.success("**Inference**\n\nfully ready")
    elif core_ready:
        st.success("**Inference**\n\nv1 ready (no efficiency prediction)")
    else:
        st.error("**Inference**\n\nnot ready")
st.caption("Full breakdown: see the **Model Status** page.")

st.divider()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
stats = storage.get_summary_stats()

if stats["total_inspections"] == 0:
    empty_state(
        "No inspections recorded yet. KPIs and charts will appear here once you "
        "run analyses from the **Inspect** page (the app's main entry point).",
    )
else:
    st.subheader("Inspection summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inspections recorded", stats["total_inspections"])
    c2.metric("Total panels analyzed", stats.get("total_panels_analyzed", 0))
    c3.metric("Clean panels", stats.get("clean_count", 0))
    c4.metric("Dusty panels", stats.get("dusty_count", 0))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Hotspot detections", stats.get("hotspot_count", 0))
    c6.metric("Predictions unavailable", stats.get("xgboost_unavailable_count", 0),
              help="Inspections where the XGBoost artifact was missing, so no efficiency/output estimate exists for them.")
    avg_eff = stats["avg_efficiency_loss_pct"]
    c7.metric(
        "Avg. estimated efficiency loss",
        f"{avg_eff:.1f}%" if avg_eff is not None else "N/A",
        help=(
            "Model estimate, averaged only over inspections where a prediction was actually "
            "computed. N/A means no stored inspection has one yet — never shown as 0%, which "
            "would look like a measured zero loss."
        ),
    )
    c8.metric("Critical / Warning", f"{stats['critical_count']} / {stats['warning_count']}")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Fault distribution")
        if stats["fault_distribution"]:
            df = pd.DataFrame({"count": stats["fault_distribution"]}).sort_values("count", ascending=False)
            st.bar_chart(df)
        else:
            empty_state("No classified faults yet.")

    with col_b:
        st.subheader("Recent inspections")
        recent = storage.get_recent_inspections(limit=8)
        if recent:
            df = pd.DataFrame(recent)[
                ["created_at", "city", "fault_label", "panel_count", "efficiency_loss_pct", "severity"]
            ]
            df.columns = ["When", "City", "Fault", "Panels", "Eff. loss %", "Severity"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            empty_state("No inspections yet.")

st.caption(
    "Per-inspection panel detail: **Panel Results**. Site-level rollups: **Site Health**. "
    "Full history and filters: **History**. Real-time analysis: the app's main **Inspect** page."
)
