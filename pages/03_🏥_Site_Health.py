"""pages/03_🏥_Site_Health.py — Site-level rollup for the most recent live
inspection, plus real aggregate history from services.storage.

Every number here is either directly aggregated from
st.session_state["last_result"].site_summary (computed by
services.pipeline._build_site_summary from real per-panel results) or from
services.storage's recorded history — never fabricated.
"""
from __future__ import annotations

import streamlit as st

from services import storage
from utils.auth import require_access
from utils.ui_theme import apply_page_chrome, empty_state

apply_page_chrome("Site Health")
require_access()

st.title("🏥 Site Health")

result = st.session_state.get("last_result")

st.subheader("This inspection")
if result is None or not getattr(result, "panels", None):
    empty_state(
        "No site-level summary yet. Run an inspection on the main **Inspect** page "
        "with at least one detected panel."
    )
else:
    summary = result.site_summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total detected panels", summary.total_panels)
    c2.metric("Clean %", f"{summary.clean_pct:.1f}%")
    c3.metric("Hotspot count", summary.class_counts.get("Hotspot", 0))
    c4.metric("Dusty %", f"{100.0 * summary.class_counts.get('Dusty', 0) / summary.total_panels:.1f}%" if summary.total_panels else "—")

    st.markdown("**Measured / observed vs. model-predicted vs. estimated**")
    st.caption(
        "Panel counts and classifications above are **model-predicted** from the "
        "uploaded image. Efficiency/output figures below are **model estimates**, "
        "not measured sensor readings."
    )
    if summary.panels_with_prediction > 0:
        c5, c6 = st.columns(2)
        c5.metric("Avg. estimated efficiency loss", f"{summary.average_efficiency_loss_pct:.1f}%")
        c6.metric("Avg. estimated output", f"{summary.average_estimated_output_w:.0f} W")
    else:
        st.info("🚫 No efficiency/output estimates available for this inspection (XGBoost artifact unavailable).")

    if summary.class_counts:
        st.subheader("Class breakdown (this inspection)")
        st.bar_chart(summary.class_counts)

    st.caption(f"Inspection timestamp source: recorded at analysis time. Weather/environment conditions used: see the **Environment** page.")

st.divider()

# ---------------------------------------------------------------------------
# Aggregate site health across all recorded history
# ---------------------------------------------------------------------------
st.subheader("Aggregate (all recorded inspections)")
stats = storage.get_summary_stats()
if stats["total_inspections"] == 0:
    empty_state("No recorded history yet.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total inspections", stats["total_inspections"])
    c2.metric("Total panels analyzed", stats["total_panels_analyzed"])
    c3.metric("Predictions unavailable", stats["xgboost_unavailable_count"])
