"""pages/11_🕰️_Panel_History.py — Persistent panel identity across runs.

The headline feature from the project vision: "Panel P2: CLEAN -> DUSTY,
first observed capture 3." Built on
training/detection/ground_pv/panel_identity.py's geometry matching (see
that module's docstring and docs/PANEL_IDENTITY_TRACKING.md) — every claim
here is a BEST-EFFORT ESTIMATE, never a verified identity.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services import storage
from utils.auth import require_access
from utils.ui_theme import apply_page_chrome, empty_state

apply_page_chrome("Panel History")
require_access()

st.title("🕰️ Panel History")
st.warning(
    "**Estimated identity, not verified.** Panels are matched across runs by comparing "
    "their normalized position in frame to the previous run — this assumes a genuinely "
    "fixed camera and the same capture order each time. If the camera moves, lighting "
    "changes framing, or you photograph the array in a different order, matching quality "
    "degrades and panels may be (correctly, honestly) reported as new rather than matched "
    "wrongly. Not yet validated against real multi-day captures — only tested with "
    "synthetic/same-session data. See `docs/PANEL_IDENTITY_TRACKING.md`."
)

st.divider()

# ---------------------------------------------------------------------------
# Recent condition changes — the actual point of this feature
# ---------------------------------------------------------------------------
st.subheader("Recent condition changes")
changes = storage.get_recent_condition_changes(limit=50)
if not changes:
    empty_state(
        "No condition changes detected yet. This requires at least two runs where the "
        "same panel was matched and confidently read both times — run Ground Inspection "
        "again on the same array to start building history."
    )
else:
    for c in changes:
        st.markdown(
            f"🔔 **{c['persistent_id']}** — {c['created_at']}  \n"
            f"**{c['previous_label']}** ({c['previous_confidence']:.0%}) → "
            f"**{c['new_label']}** ({c['new_confidence']:.0%})"
        )
        st.divider()

# ---------------------------------------------------------------------------
# All tracked panels
# ---------------------------------------------------------------------------
st.subheader("All tracked panels")
summary = storage.get_panel_identities_summary()
if not summary:
    empty_state("No panels tracked yet. Run Ground Inspection to start tracking panels across captures.")
else:
    df = pd.DataFrame(summary)[
        ["persistent_id", "capture_slot", "last_condition_label", "last_condition_confidence",
         "observation_count", "first_seen_run_id", "last_seen_run_id"]
    ]
    df.columns = ["Panel ID", "Capture slot", "Latest condition", "Confidence", "Times seen", "First seen run", "Last seen run"]
    st.dataframe(
        df.style.format({"Confidence": "{:.0%}"}),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.subheader("Panel detail")
    selected = st.selectbox("Select a panel to see its full history", [s["persistent_id"] for s in summary])
    if selected:
        history = storage.get_panel_observation_history(selected)
        if history:
            hdf = pd.DataFrame(history)[["run_id", "created_at", "condition_status", "condition_label", "condition_confidence", "match_distance"]]
            hdf.columns = ["Run", "When", "Status", "Condition", "Confidence", "Match distance"]
            st.dataframe(
                hdf.style.format({
                    "Confidence": lambda v: f"{v:.0%}" if pd.notna(v) else "—",
                    "Match distance": lambda v: f"{v:.3f}" if pd.notna(v) else "new panel (no match)",
                }),
                use_container_width=True, hide_index=True,
            )
        else:
            empty_state(f"No history found for {selected}.")

st.divider()
st.caption(
    "Pipeline: after each Ground Inspection run, "
    "`training/detection/ground_pv/panel_identity.py` matches the run's panels against the "
    "most recently known position of every tracked panel, per capture slot (no cross-slot "
    "matching, no appearance/pixel matching — position only). A condition change is only "
    "ever recorded between two CONFIDENT (\"predicted\") reads — an uncertain read is never "
    "compared as if it were a confirmed condition."
)
