"""pages/02_🔎_Panel_Results.py — Panel-level table for the most recent
live Ground Inspection run, with the legacy Aerial Inspect table kept as a
secondary, collapsed section.

Reads st.session_state["ground_inspection_captures"], set by app.py (the
Ground Inspection landing page) after a completed run — this is the only
mechanism for one Streamlit page to hand data to another in a multi-page
app. Shows an honest empty state when no run has happened yet in this
browser session, rather than any placeholder rows.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from utils.auth import require_access
from utils.ui_theme import apply_page_chrome, empty_state

apply_page_chrome("Panel Results")
require_access()

st.title("🔎 Panel Results")

captures: list[dict[str, Any]] | None = st.session_state.get("ground_inspection_captures")

if not captures:
    empty_state(
        "No panels to show yet. Run an inspection on the **Ground Inspection** page (sidebar "
        "top) - each detected panel's classifier + detector reads will appear here."
    )
else:
    multi = len(captures) > 1
    rows = []
    for idx, cap in enumerate(captures, start=1):
        report = cap["report"]
        for p in report.get("panels", []):
            display_id = f"C{idx}-{p['panel_id']}" if multi else p["panel_id"]
            cls_cond, det_cond = p["classifier_condition"], p["condition"]
            rows.append({
                "Panel ID": display_id,
                "Capture": cap["label"],
                "Detection confidence": p["detection"]["confidence"],
                "Condition (classifier, primary)": cls_cond["label"],
                "Classifier confidence": cls_cond["confidence"],
                "Condition (detector, secondary)": det_cond["label"] or "no prediction",
                "Detector confidence": det_cond["confidence"],
            })

    if not rows:
        empty_state("The most recent run detected 0 panels in every capture. Nothing to show.")
    else:
        df = pd.DataFrame(rows)
        st.caption(
            f"From the most recent Ground Inspection run in this session — {len(rows)} panel(s) "
            "across {} capture(s). Full historical detail: see the **History** page.".format(len(captures))
        )

        with st.expander("Filter", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                classes = sorted(df["Condition (classifier, primary)"].unique().tolist())
                class_filter = st.multiselect("Condition (classifier)", classes)
            with c2:
                sort_col = st.selectbox("Sort by", df.columns.tolist(), index=0)

        filtered = df.copy()
        if class_filter:
            filtered = filtered[filtered["Condition (classifier, primary)"].isin(class_filter)]
        filtered = filtered.sort_values(sort_col)

        st.dataframe(
            filtered.style.format({
                "Detection confidence": "{:.0%}",
                "Classifier confidence": "{:.0%}",
                "Detector confidence": "{:.0%}",
            }),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "The classifier is the primary condition signal (measured better: 0.752 vs. 0.727 "
            "image-level test accuracy — see `docs/CONDITION_MODEL_4CLASS_CLASSIFIER_V1.md`). "
            "The detector's read is kept as a secondary, localized cross-check."
        )

st.divider()

with st.expander("Aerial Inspect panel results (legacy production pipeline)"):
    result = st.session_state.get("last_result")
    if result is None or not getattr(result, "panels", None):
        empty_state(
            "No panels to show yet. Run an inspection on the **Aerial Inspect** page - "
            "each detected panel's own classification and estimate will appear here."
        )
    else:
        aerial_rows = []
        for p in result.panels:
            aerial_rows.append({
                "Panel ID": p.panel_index,
                "Detection confidence": p.detection_confidence,
                "Fault / class": p.classification.label,
                "Classification confidence": p.classification.confidence,
                "Efficiency loss (%)": p.prediction.efficiency_loss_pct if p.prediction.prediction_successful else None,
                "Estimated output (W)": p.prediction.estimated_output_w if p.prediction.prediction_successful else None,
                "Status": "estimate available" if p.prediction.prediction_successful else "Estimate unavailable",
            })
        adf = pd.DataFrame(aerial_rows)
        st.dataframe(
            adf.style.format({
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
