"""utils/auth.py — Single shared-password access gate.

Responsibility
--------------
- Block access to the application until a shared password is entered.

What this deliberately is NOT
------------------------------
This is a single shared password for the whole deployment, checked
client-session-side by Streamlit — it is not a multi-user account system,
has no per-user identity, no password reset flow, and no SSO. It exists to
keep a casual visitor from opening an unprotected deployment, not to
withstand a targeted attacker. If genuine multi-user authentication is
required, this module is the seam to replace, not a starting point to
"add more" to — that would misrepresent this simple gate's actual guarantees.

Configuration: set the ``APP_ACCESS_PASSWORD`` secret (via
``.streamlit/secrets.toml`` or the ``APP_ACCESS_PASSWORD`` environment
variable). When unset, the gate is a no-op (open access) — this matches
local development, where requiring a password by default would be
surprising. Production deployments should set it explicitly.
"""

from __future__ import annotations

import hmac

import streamlit as st

from utils.config import get_secret


def require_access() -> None:
    """Block the rest of the page until the correct password is entered.

    Call this as the first statement on every page (app.py and each file
    under pages/). Returns normally (falls through) once access is granted;
    calls st.stop() otherwise, so nothing below it executes.
    """
    configured = get_secret("APP_ACCESS_PASSWORD")
    if not configured:
        # No password configured — explicit local/dev mode, not a silent gap.
        return

    if st.session_state.get("solarai_authenticated") is True:
        return

    st.title("☀️ Solar AI")
    st.caption(
        "This deployment is protected by a single shared access password "
        "(not a multi-user account system)."
    )
    entered = st.text_input("Access password", type="password")
    if st.button("Enter", type="primary"):
        if hmac.compare_digest(entered, configured):
            st.session_state["solarai_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()
