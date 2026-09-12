"""pages/08_📈_Analytics.py — Trend charts over real recorded Ground
Inspection history.

Every chart here is computed from services.storage's recorded ground
inspection runs. With fewer than 2 data points, trend charts are not
meaningful — the page shows an explicit note rather than a single-point
"trend." Legacy Aerial Inspect analytics are kept, unchanged, in a
collapsed section below.
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

st.divider()

with st.expander("Aerial Inspect analytics (legacy production pipeline)"):
    arows = storage.get_recent_inspections(limit=1000)
    if not arows:
        empty_state("No inspections recorded yet — analytics need real history to chart.")
    else:
        adf = pd.DataFrame(arows)
        adf["created_at"] = pd.to_datetime(adf["created_at"])
        adf = adf.sort_values("created_at")

        if len(adf) < 2:
            st.info(
                f"Only {len(adf)} inspection recorded so far — trend charts need at least a "
                "couple of data points. The single result is available on the History page."
            )
        else:
            st.markdown("**Efficiency loss over time**")
            if "xgboost_available" in adf.columns and (adf["xgboost_available"] == 0).any():
                st.caption("⚠️ Points from inspections where prediction was unavailable are excluded from this chart.")
                eff_df = adf[adf["xgboost_available"] == 1]
            else:
                eff_df = adf
            if not eff_df.empty:
                st.line_chart(eff_df.set_index("created_at")["efficiency_loss_pct"])
            else:
                empty_state("No inspections with an available prediction yet.")

            ac1, ac2 = st.columns(2)
            with ac1:
                st.markdown("**Panel health distribution**")
                severity_counts = adf["severity"].value_counts()
                st.bar_chart(severity_counts)
            with ac2:
                st.markdown("**Fault frequency**")
                fault_counts = adf["fault_label"].value_counts()
                st.bar_chart(fault_counts)

            st.markdown("**Inspections by city**")
            city_counts = adf["city"].value_counts()
            if len(city_counts) > 0:
                st.bar_chart(city_counts)
            else:
                empty_state("No city data recorded.")
