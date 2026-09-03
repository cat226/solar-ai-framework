"""pages/2_📜_History.py — Searchable inspection history.

All rows come from services.storage (real recorded inspections). Filters
operate on that real data; an empty result set shows an explicit "no
matches" state rather than any placeholder rows.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from services import storage
from utils.auth import require_access
from utils.ui_theme import apply_page_chrome, empty_state, severity_pill

apply_page_chrome("History")
require_access()

st.title("📜 Inspection History")

all_rows = storage.get_recent_inspections(limit=500)

if not all_rows:
    empty_state(
        "No inspections recorded yet. Run an analysis from the **Inspect** page "
        "and it will appear here."
    )
    st.stop()

df = pd.DataFrame(all_rows)

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
with st.expander("Filters", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        fault_options = sorted(df["fault_label"].unique().tolist())
        fault_filter = st.multiselect("Fault type", fault_options)
    with c2:
        severity_options = sorted(df["severity"].unique().tolist())
        severity_filter = st.multiselect("Severity", severity_options)
    with c3:
        city_filter = st.text_input("City contains")

filtered = df.copy()
if fault_filter:
    filtered = filtered[filtered["fault_label"].isin(fault_filter)]
if severity_filter:
    filtered = filtered[filtered["severity"].isin(severity_filter)]
if city_filter:
    filtered = filtered[filtered["city"].str.contains(city_filter, case=False, na=False)]

st.caption(f"Showing {len(filtered)} of {len(df)} recorded inspections.")

if filtered.empty:
    empty_state("No inspections match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
display_df = filtered[
    ["id", "created_at", "city", "fault_label", "fault_confidence", "panel_count",
     "efficiency_loss_pct", "severity"]
].rename(columns={
    "id": "ID", "created_at": "When", "city": "City", "fault_label": "Fault",
    "fault_confidence": "Confidence", "panel_count": "Panels",
    "efficiency_loss_pct": "Eff. loss %", "severity": "Severity",
})
st.dataframe(display_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Detail view
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Inspection detail")
selected_id = st.selectbox("Select an inspection ID to inspect", filtered["id"].tolist())
if selected_id:
    row = storage.get_inspection(int(selected_id))
    if row:
        st.markdown(
            f"**Inspection #{row['id']}** — {row['created_at']} — {row['city']} "
            f"{severity_pill(row['severity'])}",
            unsafe_allow_html=True,
        )
        result = json.loads(row["result_json"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Fault", row["fault_label"], f"{row['fault_confidence']:.1%} confidence")
        c2.metric("Panels detected", row["panel_count"])
        c3.metric("Efficiency loss", f"{row['efficiency_loss_pct']:.1f}%")
        with st.expander("Full recorded result (JSON)"):
            st.json(result)
