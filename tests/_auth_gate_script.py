"""Test-only Streamlit script driving utils.auth.require_access() under
Streamlit's AppTest framework. Not a real app page - lives outside pages/
so it never appears in the actual application's navigation.
"""
from utils.auth import require_access

require_access()
import streamlit as st
st.text("GATE_PASSED_MARKER")
