"""pages/07_📜_History.py — Searchable Ground Inspection run history.

All rows come from services.storage (real recorded Ground Inspection runs).
Filters operate on that real data; an empty result set shows an explicit
"no matches" state rather than any placeholder rows.
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

st.title("📜 Ground Inspection History")

all_rows = storage.get_recent_ground_inspections(limit=500)

if not all_rows:
    empty_state(
        "No Ground Inspection runs recorded yet. Run an analysis from the "
        "**Ground Inspection** page (sidebar top) and it will appear here."
    )
else:
    df = pd.DataFrame(all_rows)

    with st.expander("Filters", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            worst_options = sorted(v for v in df["worst_label"].unique().tolist() if v)
            worst_filter = st.multiselect("Worst condition", worst_options)
        with c2:
            only_flagged = st.checkbox("Only runs needing attention (flagged_count > 0)")

    filtered = df.copy()
    if worst_filter:
        filtered = filtered[filtered["worst_label"].isin(worst_filter)]
    if only_flagged:
        filtered = filtered[filtered["flagged_count"] > 0]

    st.caption(f"Showing {len(filtered)} of {len(df)} recorded Ground Inspection runs.")

    if filtered.empty:
        empty_state("No runs match the current filters.")
    else:
        display_cols = ["id", "created_at", "capture_count", "panel_count", "clean_count",
                         "flagged_count", "unread_count", "worst_label"]
        rename_map = {
            "id": "ID", "created_at": "When", "capture_count": "Captures", "panel_count": "Panels",
            "clean_count": "Clean", "flagged_count": "Flagged", "unread_count": "Unread",
            "worst_label": "Worst condition",
        }
        display_df = filtered[display_cols].rename(columns=rename_map)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Run detail")
        selected_id = st.selectbox("Select a run ID to inspect", filtered["id"].tolist())
        if selected_id:
            row = storage.get_ground_inspection(int(selected_id))
            if row:
                worst_pill = severity_pill(
                    "CRITICAL" if row["worst_label"] == "PHYSICAL_DAMAGE"
                    else "WARNING" if row["flagged_count"] > 0
                    else "INFO" if row["unread_count"] > 0 and row["flagged_count"] == 0
                    else "OK"
                )
                st.markdown(
                    f"**Run #{row['id']}** — {row['created_at']} {worst_pill}",
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Captures", row["capture_count"])
                c2.metric("Panels", row["panel_count"])
                c3.metric("Worst condition", row["worst_label"] or "—")
                result = json.loads(row["result_json"])
                with st.expander("Full recorded result (JSON)"):
                    st.json(result)
