"""pages/01_📊_Overview.py — Top-level KPIs and system status.

Ground Inspection (app.py, the landing page) is now the primary flow, so
its recorded history (services.storage.get_ground_summary_stats) is the
primary content here. The legacy Aerial Inspect history is kept, not
deleted, in a collapsed section below — still real, still queryable, just
no longer the headline.

All figures come from services.storage (real recorded runs) or
models.model_manager / direct weight-file checks (real artifact readiness)
— never fabricated placeholder data. When no runs have been recorded yet,
the page shows an explicit empty state instead of invented numbers.
"""
from __future__ import annotations

from pathlib import Path

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

_GROUND_WEIGHTS = {
    "Panel detector": Path("weights/yolo_solar_ground_research_v1.pt"),
    "Condition classifier": Path("weights/mobilenet_condition_4class_research_v1.pth"),
    "Condition detector": Path("weights/yolo_solar_condition_4class_research_v2.pt"),
}
ground_missing = [name for name, p in _GROUND_WEIGHTS.items() if not p.is_file()]

status = model_manager.artifact_status
mn_status = model_manager.mobilenet_status

cols = st.columns(2)
with cols[0]:
    if ground_missing:
        st.error(f"**Ground Inspection**\n\nmissing: {', '.join(ground_missing)}")
    else:
        st.success("**Ground Inspection**\n\nready (panel detector + classifier + detector)")
with cols[1]:
    aerial_core_ready = status["YOLO"]["exists"] and mn_status["state"] != "missing"
    if aerial_core_ready and status["XGBoost"]["exists"]:
        st.success("**Aerial Inspect**\n\nfully ready")
    elif aerial_core_ready:
        st.success("**Aerial Inspect**\n\nv1 ready (no efficiency prediction)")
    else:
        st.error("**Aerial Inspect**\n\nnot ready")
st.caption("Full breakdown: see the **Model Status** page.")

st.divider()

# ---------------------------------------------------------------------------
# Ground Inspection summary — PRIMARY content (landing-page flow)
# ---------------------------------------------------------------------------
gstats = storage.get_ground_summary_stats()

if gstats["total_runs"] == 0:
    empty_state(
        "No Ground Inspection runs recorded yet. KPIs and charts will appear here once you "
        "run an inspection from the **Ground Inspection** page (sidebar top).",
    )
else:
    st.subheader("Ground Inspection summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs recorded", gstats["total_runs"])
    c2.metric("Total panels analyzed", gstats["total_panels_analyzed"])
    c3.metric("Clean panels", gstats["total_clean_panels"])
    c4.metric("Flagged panels", gstats["total_flagged_panels"],
              help="Panels the classifier confidently (>=45%) read as a non-CLEAN condition.")

    c5, c6, c7 = st.columns(3)
    c5.metric("Runs needing attention", gstats["runs_with_flagged_panels"])
    c6.metric("Unread panels", gstats["total_unread_panels"],
              help="Panels whose classifier read was uncertain — never shown as Clean.")
    c7.metric("Avg. panels per run", f"{gstats['avg_panels_per_run']:.1f}")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Condition distribution (classifier read)")
        dist = {k: v for k, v in gstats["condition_distribution"].items() if v > 0}
        if dist:
            df = pd.DataFrame({"count": dist}).sort_values("count", ascending=False)
            st.bar_chart(df)
        else:
            empty_state("No confidently-read conditions yet.")

    with col_b:
        st.subheader("Recent Ground Inspection runs")
        recent = storage.get_recent_ground_inspections(limit=8)
        if recent:
            df = pd.DataFrame(recent)[
                ["created_at", "capture_count", "panel_count", "flagged_count", "worst_label"]
            ]
            df.columns = ["When", "Captures", "Panels", "Flagged", "Worst condition"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            empty_state("No runs yet.")

st.caption(
    "Per-panel detail: **Panel Results**. Site-level rollup: **Site Health**. "
    "Full history and filters: **History**. Alerts: **Alerts**. "
    "Run a new inspection: the **Ground Inspection** page (sidebar top)."
)

st.divider()

# ---------------------------------------------------------------------------
# Aerial Inspect history — legacy production pipeline, kept queryable but
# no longer the headline (see docs/PHASE_16_ARCHITECTURE_REVIEW.md).
# ---------------------------------------------------------------------------
with st.expander("Aerial Inspect history (legacy production pipeline)"):
    stats = storage.get_summary_stats()
    if stats["total_inspections"] == 0:
        empty_state(
            "No inspections recorded yet. Run an analysis from the **Aerial Inspect** page "
            "(sidebar) to see history here.",
        )
    else:
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
        )
        c8.metric("Critical / Warning", f"{stats['critical_count']} / {stats['warning_count']}")

        st.subheader("Fault distribution (whole-image classification)")
        if stats["fault_distribution"]:
            df = pd.DataFrame({"count": stats["fault_distribution"]}).sort_values("count", ascending=False)
            st.bar_chart(df)
        else:
            empty_state("No classified faults yet.")

        st.subheader("Recent inspections")
        recent = storage.get_recent_inspections(limit=8)
        if recent:
            df = pd.DataFrame(recent)[
                ["created_at", "city", "fault_label", "panel_count", "efficiency_loss_pct", "severity"]
            ]
            df.columns = ["When", "City", "Fault (whole image)", "Panels", "Eff. loss %", "Severity"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            empty_state("No inspections yet.")
