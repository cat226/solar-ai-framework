"""pages/03_🏥_Site_Health.py — Ground Inspection array health rollup.

Redefined from its original aerial-pipeline meaning ("efficiency loss %,
estimated output W" — XGBoost-derived concepts Ground Inspection has no
equivalent of, since it involves no weather/physics/XGBoost at all). The
honest ground-inspection equivalent of "site health" is: what fraction of
panels read Clean, and which panel is worst. See
docs/PHASE_16_ARCHITECTURE_REVIEW.md for why Ground Inspection has no
efficiency/output concept.

Every number here is either aggregated from
st.session_state["ground_inspection_captures"] (the most recent live run)
or from services.storage's recorded ground-inspection history — never
fabricated. The original aerial "Site Health" (efficiency loss / output)
is kept, unchanged, in a collapsed legacy section.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from services import storage
from utils.auth import require_access
from utils.ui_theme import apply_page_chrome, empty_state

apply_page_chrome("Site Health")
require_access()

st.title("🏥 Site Health")

captures: list[dict[str, Any]] | None = st.session_state.get("ground_inspection_captures")

st.subheader("This run")
if not captures:
    empty_state(
        "No array health summary yet. Run an inspection on the **Ground Inspection** page "
        "(sidebar top) with at least one detected panel."
    )
else:
    panel_count = flagged_count = unread_count = clean_count = 0
    condition_counts: dict[str, int] = {}
    worst_panel: dict[str, Any] | None = None
    _severity = {"PHYSICAL_DAMAGE": 3, "BIRD_DROP": 2, "DUSTY": 1}

    for cap in captures:
        for p in cap["report"].get("panels", []):
            panel_count += 1
            cond = p["classifier_condition"]
            if cond["status"] == "predicted" and cond["label"] == "CLEAN":
                clean_count += 1
            elif cond["status"] == "predicted":
                flagged_count += 1
                condition_counts[cond["label"]] = condition_counts.get(cond["label"], 0) + 1
                if worst_panel is None or _severity.get(cond["label"], 0) > _severity.get(worst_panel["classifier_condition"]["label"], 0):
                    worst_panel = p
            else:
                unread_count += 1

    if panel_count == 0:
        empty_state("The most recent run detected 0 panels. Nothing to summarize.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total panels", panel_count)
        c2.metric("Clean %", f"{100.0 * clean_count / panel_count:.1f}%")
        c3.metric("Flagged (confident issue)", flagged_count)
        c4.metric("Unread (uncertain)", unread_count)

        st.caption(
            "Panel counts and conditions above are **model-predicted** (classifier, primary "
            "signal) from the uploaded photo(s) — not measured sensor readings. No efficiency "
            "or output-power estimate exists for Ground Inspection (no XGBoost/weather/physics "
            "involved) — see `docs/PHASE_16_ARCHITECTURE_REVIEW.md`."
        )

        if condition_counts:
            st.subheader("Flagged condition breakdown (this run)")
            st.bar_chart(condition_counts)

        if worst_panel is not None:
            wc = worst_panel["classifier_condition"]
            st.warning(
                f"⚠️ **Worst panel this run: {wc['label']}** — confidence {wc['confidence']:.0%}. "
                "See **Panel Results** for the full per-panel breakdown."
            )

st.divider()

# ---------------------------------------------------------------------------
# Aggregate ground health across all recorded history
# ---------------------------------------------------------------------------
st.subheader("Aggregate (all recorded Ground Inspection runs)")
gstats = storage.get_ground_summary_stats()
if gstats["total_runs"] == 0:
    empty_state("No recorded Ground Inspection history yet.")
else:
    total_panels = gstats["total_panels_analyzed"]
    clean_pct = (100.0 * gstats["total_clean_panels"] / total_panels) if total_panels else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Total runs", gstats["total_runs"])
    c2.metric("Total panels analyzed", total_panels)
    c3.metric("Overall clean %", f"{clean_pct:.1f}%")

st.divider()

with st.expander("Aerial Inspect site health (legacy production pipeline)"):
    result = st.session_state.get("last_result")
    if result is None or not getattr(result, "panels", None):
        empty_state(
            "No site-level summary yet. Run an inspection on the **Aerial Inspect** page "
            "with at least one detected panel."
        )
    else:
        summary = result.site_summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total detected panels", summary.total_panels)
        c2.metric("Clean %", f"{summary.clean_pct:.1f}%")
        c3.metric("Hotspot count", summary.class_counts.get("Hotspot", 0))
        c4.metric("Dusty %", f"{100.0 * summary.class_counts.get('Dusty', 0) / summary.total_panels:.1f}%" if summary.total_panels else "—")

        if summary.panels_with_prediction > 0:
            c5, c6 = st.columns(2)
            c5.metric("Avg. estimated efficiency loss", f"{summary.average_efficiency_loss_pct:.1f}%")
            c6.metric("Avg. estimated output", f"{summary.average_estimated_output_w:.0f} W")
        else:
            st.info("🚫 No efficiency/output estimates available for this inspection (XGBoost artifact unavailable).")

        if summary.class_counts:
            st.subheader("Class breakdown (this inspection)")
            st.bar_chart(summary.class_counts)

    st.markdown("**Aggregate (all recorded Aerial Inspect history)**")
    stats = storage.get_summary_stats()
    if stats["total_inspections"] == 0:
        empty_state("No recorded history yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total inspections", stats["total_inspections"])
        c2.metric("Total panels analyzed", stats["total_panels_analyzed"])
        c3.metric("Predictions unavailable", stats["xgboost_unavailable_count"])
