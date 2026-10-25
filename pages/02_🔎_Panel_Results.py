"""pages/02_🔎_Panel_Results.py — Panel-level table for the most recent
live Ground Inspection run.

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
            "The classifier is the primary condition signal (rooftop-adapted Phase 19-B retrain; "
            "its non-CLEAN accuracy on this rooftop is thinly measured — see "
            "`docs/ROOFTOP_CONDITION_MIX_PHASE19B.md`). "
            "The detector's read is kept as a secondary, localized cross-check."
        )
