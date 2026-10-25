"""pages/08_📈_Analytics.py — Trend charts over real recorded Ground
Inspection history.

Every chart here is computed from services.storage's recorded ground
inspection runs. With fewer than 2 data points, trend charts are not
meaningful — the page shows an explicit note rather than a single-point
"trend."
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services import storage
from utils.auth import require_access
from utils.ui_theme import apply_page_chrome, empty_state

apply_page_chrome("Analytics")
require_access()

st.title("📈 Analytics")

rows = storage.get_recent_ground_inspections(limit=1000)
if not rows:
    empty_state("No Ground Inspection runs recorded yet — analytics need real history to chart.")
else:
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.sort_values("created_at")

    if len(df) < 2:
        st.info(
            f"Only {len(df)} Ground Inspection run recorded so far — trend charts need at least "
            "a couple of data points. The single result is available on the History page."
        )
    else:
        st.subheader("Flagged panels over time")
        st.line_chart(df.set_index("created_at")["flagged_count"])

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Panels per run")
            st.bar_chart(df.set_index("created_at")["panel_count"])
        with col_b:
            st.subheader("Worst condition frequency")
            worst_counts = df[df["worst_label"] != ""]["worst_label"].value_counts()
            if len(worst_counts) > 0:
                st.bar_chart(worst_counts)
            else:
                empty_state("No runs with a flagged/uncertain/clean read yet.")

        gstats = storage.get_ground_summary_stats()
        st.subheader("Condition distribution (all recorded panels)")
        dist = {k: v for k, v in gstats["condition_distribution"].items() if v > 0}
        if dist:
            st.bar_chart(pd.DataFrame({"count": dist}).sort_values("count", ascending=False))
        else:
            empty_state("No confidently-read conditions yet.")
