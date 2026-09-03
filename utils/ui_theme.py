"""utils/ui_theme.py — Shared visual chrome for every Solar AI page.

Responsibility
--------------
- Apply consistent page config, header branding, and light CSS polish across
  app.py and every page under pages/, so the multi-page app reads as one
  coherent product rather than a pile of default Streamlit pages.
- Small, presentation-only helpers (status pills, KPI cards) used by
  multiple pages, to avoid duplicating markup.

This module contains no business logic and no model/pipeline knowledge.
"""

from __future__ import annotations

import streamlit as st

_BRAND_CSS = """
<style>
/* Tighter, more deliberate spacing than Streamlit's defaults, and a
   consistent card treatment for metric/status blocks across pages. */
div[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 10px;
    padding: 0.9rem 1rem 0.6rem 1rem;
}
.solarai-pill {
    display: inline-block;
    padding: 0.15rem 0.65rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}
.solarai-pill-critical { background: #FEE2E2; color: #991B1B; }
.solarai-pill-warning  { background: #FEF3C7; color: #92400E; }
.solarai-pill-info     { background: #DBEAFE; color: #1E40AF; }
.solarai-pill-ok       { background: #DCFCE7; color: #166534; }
.solarai-caption {
    color: #6B7280;
    font-size: 0.85rem;
}
</style>
"""

_SEVERITY_PILL_CLASS = {
    "CRITICAL": "solarai-pill-critical",
    "WARNING": "solarai-pill-warning",
    "INFO": "solarai-pill-info",
    "OK": "solarai-pill-ok",
}


def apply_page_chrome(page_title: str, *, page_icon: str = "☀️") -> None:
    """Set page config and inject shared branding CSS. Call once per page,
    before any other Streamlit output."""
    st.set_page_config(
        page_title=f"Solar AI — {page_title}",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_BRAND_CSS, unsafe_allow_html=True)


def severity_pill(severity: str) -> str:
    """Return an HTML span rendering `severity` as a colored pill.
    Pass through st.markdown(..., unsafe_allow_html=True)."""
    css_class = _SEVERITY_PILL_CLASS.get(severity.upper(), "solarai-pill-info")
    return f'<span class="solarai-pill {css_class}">{severity.upper()}</span>'


def empty_state(message: str, *, icon: str = "📭") -> None:
    """Render a consistent, honest empty state instead of fabricated data."""
    st.info(f"{icon} {message}")
