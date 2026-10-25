"""pages/12_📷_Cameras.py — Admin overview of every registered camera
(Phase 18). Deliberately minimal per the confirmed brief: one row per
camera, its number, whether it's currently reporting a problem, and when it
was last seen. No configuration controls here — that's
pages/13_🛠️_Camera_Configuration.py, kept separate so this page stays a
clean "what's the state of my array right now" read.
"""
from __future__ import annotations

import html

import streamlit as st

from services import storage
from utils.auth import require_access
from utils.ui_theme import apply_page_chrome, empty_state, severity_pill

apply_page_chrome("Cameras", page_icon="📷")
require_access()

st.title("📷 Cameras")
st.caption(
    "Every registered Pi camera device, its latest reported result, and whether it currently "
    "has a flagged issue. Add or remove cameras on the **Camera Configuration** page (sidebar)."
)

cameras = storage.get_cameras_with_latest_status()

if not cameras:
    empty_state(
        "No cameras registered yet. Go to **Camera Configuration** (sidebar) to add one and "
        "get its API token for the Pi."
    )
    st.stop()

problem_cameras = [c for c in cameras if c["latest_run"] and c["latest_run"]["flagged_count"] > 0]
if problem_cameras:
    numbers = ", ".join(f"#{c['camera_number']}" for c in problem_cameras)
    st.error(f"🚨 **{len(problem_cameras)} camera(s) reporting a problem:** {numbers}")
else:
    st.success("✅ No connected camera is currently reporting a flagged issue.")

st.divider()

for cam in cameras:
    latest = cam["latest_run"]
    with st.container(border=True):
        cols = st.columns([1, 2, 2, 2])

        cols[0].markdown(f"### #{cam['camera_number']}")
        status_pill = "OK" if cam["status"] == "active" else "WARNING"
        cols[0].markdown(severity_pill(status_pill if cam["status"] == "active" else "WARNING"), unsafe_allow_html=True)

        cols[1].markdown(f"**{cam['name']}**")
        if cam["location"]:
            cols[1].caption(cam["location"])
        cols[1].caption(f"Status: {cam['status']}")

        if cam["last_seen_at"]:
            cols[2].markdown(f"**Last seen**  \n{cam['last_seen_at']}")
        else:
            cols[2].markdown("**Last seen**  \nNever reported in")

        if latest is None:
            cols[3].markdown("**Latest result**  \n_No runs yet_")
        elif latest["flagged_count"] > 0:
            # worst_label is stored data (see the same note in
            # pages/09_🚨_Alerts.py) — escaped before going into this
            # unsafe_allow_html=True block as defense in depth.
            worst_label = html.escape(str(latest["worst_label"]))
            cols[3].markdown(
                f"{severity_pill('CRITICAL' if latest['worst_label'] == 'PHYSICAL_DAMAGE' else 'WARNING')} "
                f"&nbsp; **{worst_label}**", unsafe_allow_html=True,
            )
            cols[3].caption(f"{latest['panel_count']} panels · {latest['flagged_count']} flagged · {latest['created_at']}")
        else:
            cols[3].markdown(f"{severity_pill('OK')} &nbsp; **Clean**", unsafe_allow_html=True)
            cols[3].caption(f"{latest['panel_count']} panels · {latest['created_at']}")

st.divider()
st.caption(
    "Data comes from `services/storage.py::get_cameras_with_latest_status()` — the same "
    "history table Overview/History/Analytics/Alerts read, filtered per camera. A camera "
    "'reports in' by POSTing an already-analyzed capture batch to the ingestion API "
    "(`api/main.py`) — see `docs/RASPBERRY_PI_DEPLOYMENT.md`."
)
