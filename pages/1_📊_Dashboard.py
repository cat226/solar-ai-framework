"""pages/1_📊_Dashboard.py — Overview KPIs and system status.

All figures on this page come from services.storage (real recorded
inspections) or models.model_manager (real artifact readiness) — never
fabricated placeholder data. When no inspections have been recorded yet,
the page shows an explicit empty state instead of invented numbers.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from models.model_manager import model_manager
from services import storage
from utils.auth import require_access
from utils.ui_theme import apply_page_chrome, empty_state

apply_page_chrome("Dashboard")
require_access()

st.title("📊 Dashboard")
st.caption("Live overview, computed from this deployment's own recorded inspection history.")

# ---------------------------------------------------------------------------
# System status — real readiness, never shown as healthy if it isn't.
# ---------------------------------------------------------------------------
st.subheader("System status")
status = model_manager.artifact_status
cols = st.columns(len(status) + 1)
for col, (name, entry) in zip(cols, status.items()):
    with col:
        if entry["exists"]:
            st.success(f"**{name}**\n\nready")
        else:
            st.error(f"**{name}**\n\nmissing")
missing = [name for name, entry in status.items() if not entry["exists"]]
with cols[-1]:
    if missing:
        st.warning(f"**Inference**\n\nnot ready")
    else:
        st.success(f"**Inference**\n\nready")

st.divider()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
stats = storage.get_summary_stats()

if stats["total_inspections"] == 0:
    empty_state(
        "No inspections recorded yet. KPIs and charts will appear here once you "
        "run analyses from the **Inspect** page.",
    )
else:
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inspections recorded", stats["total_inspections"])
    c2.metric("Avg. panels detected", f"{stats['avg_panel_count']:.1f}")
    c3.metric("Avg. efficiency loss", f"{stats['avg_efficiency_loss_pct']:.1f}%")
    c4.metric("Critical / Warning", f"{stats['critical_count']} / {stats['warning_count']}")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Fault distribution")
        if stats["fault_distribution"]:
            df = pd.DataFrame(
                {"count": stats["fault_distribution"]}
            ).sort_values("count", ascending=False)
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

st.caption("Full history and filters: see the **History** page. Real-time analysis: see **Inspect**.")
