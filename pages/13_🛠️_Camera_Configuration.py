"""pages/13_🛠️_Camera_Configuration.py — add/remove/disable registered
Pi camera devices (Phase 18). Kept separate from pages/12_📷_Cameras.py so
that page stays a clean read-only status view and this one holds every
mutating control.

Registration issues a fresh API token via `storage.register_camera()` —
shown here ONCE, immediately after creation, matching how e.g. GitHub only
displays a new access token at creation time. It is never re-displayed by
any read path in `services/storage.py` (list_cameras/get_camera/
get_cameras_with_latest_status all redact it).
"""
from __future__ import annotations

import streamlit as st

from services import storage
from utils.auth import require_access
from utils.ui_theme import apply_page_chrome, empty_state

apply_page_chrome("Camera Configuration", page_icon="🛠️")
require_access()

st.title("🛠️ Camera Configuration")
st.caption(
    "Add a new Pi camera or remove/disable an existing one. See the overview of live camera "
    "status on the **Cameras** page (sidebar)."
)

# ---------------------------------------------------------------------------
# Add a camera
# ---------------------------------------------------------------------------
st.subheader("Add a camera")

with st.form("add_camera_form", clear_on_submit=True):
    name = st.text_input("Camera name", placeholder="e.g. Rooftop North")
    location = st.text_input("Location (optional)", placeholder="e.g. Building A, north-facing array")
    submitted = st.form_submit_button("➕ Register camera", type="primary")

if submitted:
    if not name.strip():
        st.error("Camera name is required.")
    else:
        new_cam = storage.register_camera(name, location)
        st.session_state["_just_registered_camera"] = new_cam

just_registered = st.session_state.get("_just_registered_camera")
if just_registered:
    st.success(f"✅ Registered **camera #{just_registered['camera_number']}** — {just_registered['name']}")
    st.warning(
        "⚠️ **Copy this API token now — it will never be shown again.** Put it in the Pi "
        "agent's config (see `docs/RASPBERRY_PI_DEPLOYMENT.md`)."
    )
    st.code(just_registered["api_token"], language=None)
    st.caption(
        f"Ingestion endpoint: `POST /v1/ingest` with header "
        f"`Authorization: Bearer {just_registered['api_token'][:8]}...` — the Pi agent "
        "reads this from its own local config file, never hardcoded in this app."
    )
    if st.button("Done — dismiss this token"):
        del st.session_state["_just_registered_camera"]
        st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Existing cameras
# ---------------------------------------------------------------------------
st.subheader("Existing cameras")

cameras = storage.list_cameras()
if not cameras:
    empty_state("No cameras registered yet — add one above.")
    st.stop()

for cam in cameras:
    with st.container(border=True):
        cols = st.columns([3, 2, 2, 2])
        cols[0].markdown(f"**#{cam['camera_number']} — {cam['name']}**")
        if cam["location"]:
            cols[0].caption(cam["location"])
        cols[1].markdown(f"Status: **{cam['status']}**")
        cols[1].caption(f"Registered {cam['created_at']}")
        cols[2].markdown(f"Last seen: {cam['last_seen_at'] or 'Never'}")

        action_cols = cols[3].columns(2)
        if cam["status"] == "active":
            if action_cols[0].button("Disable", key=f"disable_{cam['id']}"):
                storage.set_camera_status(cam["id"], "disabled")
                st.rerun()
        else:
            if action_cols[0].button("Enable", key=f"enable_{cam['id']}"):
                storage.set_camera_status(cam["id"], "active")
                st.rerun()

        confirm_key = f"confirm_delete_{cam['id']}"
        if st.session_state.get(confirm_key):
            if action_cols[1].button("⚠️ Confirm delete", key=f"confirm_{cam['id']}", type="primary"):
                storage.delete_camera(cam["id"])
                del st.session_state[confirm_key]
                st.rerun()
        else:
            if action_cols[1].button("Remove", key=f"remove_{cam['id']}"):
                st.session_state[confirm_key] = True
                st.rerun()

st.divider()
st.caption(
    "**Disable** immediately stops that camera's token from authenticating with the "
    "ingestion API, without deleting its recorded history (reversible — enable it again "
    "later). **Remove** permanently deletes the camera registration; its historical runs "
    "and panels stay in the database (they keep the deleted camera's id as an orphaned "
    "reference, so real recorded history is never silently destroyed)."
)
