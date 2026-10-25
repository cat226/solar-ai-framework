"""pages/05_🧠_Model_Status.py — Transparent model/system status.

Every value here is read live from the weight files on disk (existence,
SHA-256) — nothing aspirational, nothing fabricated. Ground Inspection is
the only pipeline now — the legacy Aerial Inspect pipeline was fully
removed (see docs/PHASE_16_ARCHITECTURE_REVIEW.md).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from utils.auth import require_access
from utils.ui_theme import apply_page_chrome

apply_page_chrome("Model Status")
require_access()

st.title("🧠 Model Status")
st.caption("Dataset/model provenance detail, kept separate from the Ground Inspection workflow so it doesn't clutter it.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_card(title: str, path: Path, note: str) -> None:
    st.subheader(title)
    if path.is_file():
        st.success(f"Ready — `{path}`")
        with st.spinner("Computing checkpoint hash…"):
            st.code(f"path   = {path}\nsha256 = {_sha256(path)}", language="text")
    else:
        st.error(f"Missing — expected at `{path}`")
    st.markdown(note)
    st.divider()


_artifact_card(
    "🎯 Ground panel detector",
    Path("weights/yolo_solar_rooftop_v1.pt"),
    "Phase 19 retrain on 54 first-party rooftop photos (labels are AI-reviewed SAM "
    "proposals, **not human-verified** — see `docs/ROOFTOP_PANEL_DETECTOR_PHASE19.md`). "
    "On a held-out, scene-disjoint slice of that same rooftop, recall went from 6/48 "
    "panels (predecessor checkpoint) to 39/48 (independently re-verified locally, "
    "conf≥0.25), with no regression on the original Susan test set. ⚠️ Trained on ONE "
    "rooftop — the held-out scenes are a different capture session on the same site, "
    "not a different installation, so generalization to a different roof is unverified. "
    "See **Limitations** for the full picture.",
)

_artifact_card(
    "🏷️ Condition classifier (primary signal)",
    Path("weights/mobilenet_condition_4class_rooftop_mix_oversample1_v1.pth"),
    "MobileNetV2 (Phase 16-B architecture), Phase 19-B domain-adapted retrain: the "
    "original Phase 16-B checkpoint (0.752 on its own Susan-derived test set) scored only "
    "**0.271 (13/48)** on THIS rooftop's own held-out photos — systematically misreading "
    "CLEAN panels as DUSTY/PHYSICAL_DAMAGE at high confidence. Retrained blended with this "
    "rooftop's own labelled crops: scored 0.938 (45/48) on 48 AI-labelled held-out crops "
    "(46 CLEAN + 2 DUSTY, no defect examples). **That figure is not evidence of defect "
    "detection:** a later direct check against the owner's own defect marks found it called all "
    "5 marked bird-drop panels CLEAN. Disclosed cost: 0.731 (down from 0.752) on the original "
    "Susan-derived benchmark. A web-trained ensemble is available as an opt-in alternative on the "
    "Ground Inspection page (macro-F1 0.878 on web photos only; also unvalidated on this rooftop). "
    "See `docs/ROOFTOP_CONDITION_MIX_PHASE19B.md` for the full comparison and confusion "
    "matrices.",
)

_artifact_card(
    "📦 Condition detector (secondary, localized)",
    Path("weights/yolo_solar_condition_4class_research_v2.pt"),
    "Box-detector condition model. Measured **0.444 mAP50 / 0.441 box-level recall** — "
    "weakness is dominated by localization noise, not misrecognition (image-level "
    "agreement 0.727 on the same checkpoint). See "
    "`docs/CONDITION_MODEL_IMAGE_LEVEL_AGREEMENT.md`. Kept for its localization value on "
    "BIRD_DROP/PHYSICAL_DAMAGE, where the classifier gives up 'where on the panel.'",
)

st.subheader("Overall")
_all_present = all(
    Path(p).is_file()
    for p in (
        "weights/yolo_solar_rooftop_v1.pt",
        "weights/mobilenet_condition_4class_rooftop_mix_oversample1_v1.pth",
        "weights/yolo_solar_condition_4class_research_v2.pt",
    )
)
if _all_present:
    st.success(
        "✅ Ground Inspection pipeline ready — panel detection and both condition engines "
        "execute without error. This describes pipeline *readiness*, not *accuracy* — see "
        "the notes above and **Limitations**."
    )
else:
    st.error("❌ Not ready — at least one required artifact is missing.")
