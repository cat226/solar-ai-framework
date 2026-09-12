"""pages/07_📜_History.py — Searchable Ground Inspection run history.

All rows come from services.storage (real recorded Ground Inspection runs).
Filters operate on that real data; an empty result set shows an explicit
"no matches" state rather than any placeholder rows. Legacy Aerial Inspect
history is kept, unchanged, in a collapsed section below.
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

st.divider()

with st.expander("Aerial Inspect history (legacy production pipeline)"):
    aerial_rows = storage.get_recent_inspections(limit=500)
    if not aerial_rows:
        empty_state(
            "No inspections recorded yet. Run an analysis from the **Aerial Inspect** page "
            "and it will appear here."
        )
    else:
        adf = pd.DataFrame(aerial_rows)

        _CLASSIFIER_SOURCE_DISPLAY = {
            "production": "six_class (legacy label)",
            "interim": "v1 (legacy label)",
        }
        if "classifier_source" in adf.columns:
            adf["classifier_source"] = adf["classifier_source"].map(
                lambda v: _CLASSIFIER_SOURCE_DISPLAY.get(v, v)
            )

        ac1, ac2, ac3, ac4 = st.columns(4)
        with ac1:
            fault_options = sorted(adf["fault_label"].unique().tolist())
            fault_filter = st.multiselect("Fault type", fault_options)
        with ac2:
            severity_options = sorted(adf["severity"].unique().tolist())
            severity_filter = st.multiselect("Severity", severity_options)
        with ac3:
            city_filter = st.text_input("City contains")
        with ac4:
            source_options = sorted(adf["classifier_source"].unique().tolist()) if "classifier_source" in adf.columns else []
            source_filter = st.multiselect("Classifier used", source_options)

        afiltered = adf.copy()
        if fault_filter:
            afiltered = afiltered[afiltered["fault_label"].isin(fault_filter)]
        if severity_filter:
            afiltered = afiltered[afiltered["severity"].isin(severity_filter)]
        if city_filter:
            afiltered = afiltered[afiltered["city"].str.contains(city_filter, case=False, na=False, regex=False)]
        if source_filter and "classifier_source" in afiltered.columns:
            afiltered = afiltered[afiltered["classifier_source"].isin(source_filter)]

        st.caption(f"Showing {len(afiltered)} of {len(adf)} recorded inspections.")

        if afiltered.empty:
            empty_state("No inspections match the current filters.")
        else:
            adisplay_cols = ["id", "created_at", "city", "fault_label", "fault_confidence", "panel_count",
                              "efficiency_loss_pct", "severity"]
            arename_map = {
                "id": "ID", "created_at": "When", "city": "City", "fault_label": "Fault (whole image)",
                "fault_confidence": "Confidence", "panel_count": "Panels",
                "efficiency_loss_pct": "Eff. loss %", "severity": "Severity",
            }
            if "classifier_source" in afiltered.columns:
                adisplay_cols.append("classifier_source")
                arename_map["classifier_source"] = "Classifier"
            adisplay_df = afiltered[adisplay_cols].rename(columns=arename_map)
            st.dataframe(adisplay_df, use_container_width=True, hide_index=True)

            st.markdown("**Inspection detail**")
            aselected_id = st.selectbox("Select an inspection ID to inspect", afiltered["id"].tolist(), key="aerial_history_select")
            if aselected_id:
                arow = storage.get_inspection(int(aselected_id))
                if arow:
                    st.markdown(
                        f"**Inspection #{arow['id']}** — {arow['created_at']} — {arow['city']} "
                        f"{severity_pill(arow['severity'])}",
                        unsafe_allow_html=True,
                    )
                    aresult = json.loads(arow["result_json"])
                    with st.expander("Full recorded result (JSON)"):
                        st.json(aresult)
