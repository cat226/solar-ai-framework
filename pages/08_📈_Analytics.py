"""pages/08_📈_Analytics.py — Trend charts over real recorded history.

Every chart here is computed from services.storage's recorded inspections.
With fewer than 2 data points, trend charts are not meaningful — the page
shows an explicit note rather than a single-point "trend."
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

rows = storage.get_recent_inspections(limit=1000)
if not rows:
    empty_state("No inspections recorded yet — analytics need real history to chart.")
    st.stop()

df = pd.DataFrame(rows)
df["created_at"] = pd.to_datetime(df["created_at"])
df = df.sort_values("created_at")

if len(df) < 2:
    st.info(
        f"Only {len(df)} inspection recorded so far — trend charts need at least a "
        "couple of data points. The single result is available on the History page."
    )
    st.stop()

st.subheader("Efficiency loss over time")
if "xgboost_available" in df.columns and (df["xgboost_available"] == 0).any():
    st.caption("⚠️ Points from inspections where prediction was unavailable are excluded from this chart.")
    eff_df = df[df["xgboost_available"] == 1]
else:
    eff_df = df
if not eff_df.empty:
    st.line_chart(eff_df.set_index("created_at")["efficiency_loss_pct"])
else:
    empty_state("No inspections with an available prediction yet.")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Panel health distribution")
    severity_counts = df["severity"].value_counts()
    st.bar_chart(severity_counts)

with col_b:
    st.subheader("Fault frequency")
    fault_counts = df["fault_label"].value_counts()
    st.bar_chart(fault_counts)

st.subheader("Inspections by city")
city_counts = df["city"].value_counts()
if len(city_counts) > 0:
    st.bar_chart(city_counts)
else:
    empty_state("No city data recorded.")

st.caption(
    "Correlation with environmental conditions (irradiance, temperature) is available "
    "per-inspection on the History page's detail view — aggregate environmental "
    "correlation charts require more recorded history to be statistically meaningful."
)
