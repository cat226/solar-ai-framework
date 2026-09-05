"""pages/02_🔎_Panel_Results.py — Panel-level table for the most recent
live inspection.

Reads st.session_state["last_result"], set by app.py (the Inspect page)
after a completed run — this is the only mechanism for one Streamlit page
to hand data to another in a multi-page app. Shows an honest empty state
when no inspection has been run yet in this browser session, rather than
any placeholder rows.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.auth import require_access
from utils.ui_theme import apply_page_chrome, empty_state

apply_page_chrome("Panel Results")
require_access()

st.title("🔎 Panel Results")

result = st.session_state.get("last_result")
if result is None or not getattr(result, "panels", None):
    empty_state(
        "No panels to show yet. Run an inspection on the main **Inspect** page - "
        "each detected panel's own classification and estimate will appear here."
    )
    st.stop()

st.caption(
    f"From the most recent inspection in this session — {len(result.panels)} panel(s) detected. "
    "Full historical detail: see the **History** page's per-inspection JSON."
)

rows = []
for p in result.panels:
    rows.append({
        "Panel ID": p.panel_index,
        "Detection confidence": p.detection_confidence,
        "Fault / class": p.classification.label,
        "Classification confidence": p.classification.confidence,
        "Efficiency loss (%)": p.prediction.efficiency_loss_pct if p.prediction.prediction_successful else None,
        "Estimated output (W)": p.prediction.estimated_output_w if p.prediction.prediction_successful else None,
        "Status": "estimate available" if p.prediction.prediction_successful else "Estimate unavailable",
    })
df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Filter / sort
# ---------------------------------------------------------------------------
with st.expander("Filter", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        classes = sorted(df["Fault / class"].unique().tolist())
        class_filter = st.multiselect("Fault / class", classes)
    with c2:
        sort_col = st.selectbox("Sort by", df.columns.tolist(), index=0)

filtered = df.copy()
if class_filter:
    filtered = filtered[filtered["Fault / class"].isin(class_filter)]
filtered = filtered.sort_values(sort_col)

st.dataframe(
    filtered.style.format({
        "Detection confidence": "{:.0%}",
        "Classification confidence": "{:.0%}",
        "Efficiency loss (%)": lambda v: f"{v:.1f}" if pd.notna(v) else "—",
        "Estimated output (W)": lambda v: f"{v:.0f}" if pd.notna(v) else "—",
    }),
    use_container_width=True, hide_index=True,
)

if not result.xgboost_available:
    st.caption(
        "⚠️ Efficiency/output columns show '—' because the XGBoost artifact was "
        "unavailable for this inspection — not because the panels had zero loss."
    )
