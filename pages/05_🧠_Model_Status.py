"""pages/05_🧠_Model_Status.py — Transparent model/system status.

Every value here is read live from models.model_manager (real artifact
paths/existence/hashes) — nothing aspirational, nothing fabricated. This is
the one place the app explicitly states its current classification
coverage and what remains pending.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from models.model_manager import model_manager
from utils.auth import require_access
from utils.ui_theme import apply_page_chrome

apply_page_chrome("Model Status")
require_access()

st.title("🧠 Model Status")
st.caption("Dataset/model provenance detail, kept separate from the main Inspect workflow so it doesn't clutter it.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


status = model_manager.artifact_status
mn_status = model_manager.mobilenet_status

# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------
st.subheader("🎯 YOLO — panel detection")
yolo = status["YOLO"]
if yolo["exists"]:
    st.success("Ready")
    yolo_path = Path(yolo["path"])
    with st.spinner("Computing checkpoint hash…"):
        st.code(f"path   = {yolo['path']}\nsha256 = {_sha256(yolo_path)}", language="text")
else:
    st.error(f"Missing — expected at `{yolo['path']}`")

st.divider()

# ---------------------------------------------------------------------------
# MobileNet
# ---------------------------------------------------------------------------
st.subheader("🏷️ MobileNet — fault classification")
if mn_status["state"] == "six_class":
    st.success(f"Ready — **future six-class artifact, {len(mn_status['active_labels'])}-class**")
elif mn_status["state"] == "v1":
    st.success(f"Ready — **v1, {len(mn_status['active_labels'])}-class**")
    st.markdown(
        "This is the real, frozen v1 release classifier — not a placeholder. It covers "
        "the three classes with genuinely licensed, accessible data. Bird-Drop, "
        "Electrical-Damage, and Physical-Damage remain a documented future expansion; "
        "see the **Limitations** page."
    )
else:
    st.error("Missing — no v1 or future six-class checkpoint present")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Currently active classes (v1)**")
    st.write(mn_status["active_labels"] or "—")
with c2:
    st.markdown("**Full future six-class order (roadmap)**")
    st.write(mn_status["six_class_labels"])

if mn_status["v1_exists"] or mn_status["six_class_exists"]:
    active_path = Path(mn_status["six_class_path"]) if mn_status["state"] == "six_class" else Path(mn_status["v1_path"])
    with st.spinner("Computing checkpoint hash…"):
        st.code(f"path   = {active_path}\nsha256 = {_sha256(active_path)}", language="text")

st.divider()

# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
st.subheader("📈 XGBoost — efficiency-loss prediction")
xgb = status["XGBoost"]
if xgb["exists"]:
    st.success("Ready")
    with st.spinner("Computing artifact hash…"):
        st.code(f"path   = {xgb['path']}\nsha256 = {_sha256(Path(xgb['path']))}", language="text")
else:
    st.error(f"Missing — expected at `{xgb['path']}`")
    st.caption(
        "No XGBoost artifact has been trained/promoted for this deployment. Inspections "
        "still run detection and classification; efficiency/output estimates are "
        "reported as unavailable rather than fabricated."
    )
    with st.expander("Why is there no XGBoost artifact?"):
        st.markdown(
            "Investigated on 2026-09-04: no dataset was found that legitimately pairs a "
            "real `fault_class_id` (this project's own six-class taxonomy) with real "
            "environmental telemetry *and* a genuinely measured efficiency-loss target. "
            "Rather than train on invented labels, or on this app's own physics heuristic "
            "restated as if it were ground truth, the artifact is left unavailable. "
            "See `training/prediction/DATASET_SOURCES.md` for the full investigation and "
            "what would need to exist to revisit this."
        )

st.divider()

# ---------------------------------------------------------------------------
# Deep verification (opt-in — actually attempts to load every model)
# ---------------------------------------------------------------------------
st.subheader("Deep verification")
st.caption(
    "The status above only checks whether artifact *files* exist. This actually attempts "
    "to load each one, catching a present-but-corrupt or incompatible checkpoint that file "
    "existence alone can't detect. Opt-in (not run automatically) since it can trigger a "
    "real model load - free if the model is already loaded this session, otherwise pays "
    "the real load cost once."
)
if st.button("Run deep verification"):
    with st.spinner("Loading each model…"):
        report = model_manager.verify_all()
    state_display = {
        "ready": ("✅", "ready"), "v1": ("✅", "v1"), "six_class": ("✅", "six_class"),
        "missing": ("❌", "missing"), "error": ("🚫", "present but failed to load"),
    }
    for name, entry in report.items():
        icon, label = state_display.get(entry["state"], ("❓", entry["state"]))
        st.markdown(f"{icon} **{name}**: {label}")
        if entry["detail"]:
            st.caption(entry["detail"])

st.divider()
st.subheader("Overall")
if yolo["exists"] and mn_status["state"] != "missing" and xgb["exists"]:
    st.success("✅ Fully operational — detection, classification, and efficiency prediction all ready.")
elif yolo["exists"] and mn_status["state"] != "missing":
    st.success(
        "✅ **Solar AI v1 is fully operational.** Detection and classification (Clean, "
        "Dusty, Hotspot) work; efficiency/output prediction is unavailable — a documented "
        "v1 boundary, not a fault (see **Limitations**)."
    )
else:
    st.error("❌ Not ready — a required artifact (YOLO or MobileNet) is missing.")
