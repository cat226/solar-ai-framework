"""pages/06_⚠️_Limitations.py — Honest, explicit capability disclosure.

Ground Inspection is the only pipeline now — the legacy Aerial Inspect
pipeline was fully removed (see docs/PHASE_16_ARCHITECTURE_REVIEW.md).
"""
from __future__ import annotations

import streamlit as st

from utils.auth import require_access
from utils.ui_theme import apply_page_chrome

apply_page_chrome("Limitations")
require_access()

st.title("⚠️ Limitations")
st.caption("What this deployment can and cannot do right now — kept visible, not buried.")

st.subheader("Ground Inspection — research prototype, not production-validated")
st.error(
    "**Not a validated diagnosis.** Every result is a raw model output on a research "
    "checkpoint, evaluated only on a held-out test set — never on the live photo you upload."
)
st.markdown(
    "- **Condition classifier (primary signal):** MobileNetV2, measured 0.752 image-level "
    "test accuracy overall (DUSTY 0.794, BIRD_DROP 0.676, CLEAN 0.764, PHYSICAL_DAMAGE 0.778 "
    "on only 9 test images — high variance). See `docs/CONDITION_MODEL_4CLASS_CLASSIFIER_V1.md`.\n"
    "- **Condition detector (secondary, localized):** box-detector, measured 0.444 mAP50 / "
    "0.441 recall at the box level (localization-limited, not misrecognition — see "
    "`docs/CONDITION_MODEL_IMAGE_LEVEL_AGREEMENT.md`). Kept visible for localization value "
    "on BIRD_DROP/PHYSICAL_DAMAGE, where the classifier gives up 'where on the panel.'\n"
    "- **Panel detector:** can miss a single large foreground panel, or draw one 'group' box "
    "around several adjacent panels instead of separating them — documented in "
    "`datasets/ground_pv_research/reports/real_world_holdout.md`. Tiling was tested as a fix "
    "and produced confident FALSE positives on sky/brick instead, so it is NOT used.\n"
    "- **One distant photo of a whole 6+ panel array does not work reliably** — the "
    "detector's real working range is close-up shots of ~1-3 panels. Cover a larger array "
    "with multiple closer photos instead (built into the page).\n"
    "- **No persistent panel identity across captures.** Panel IDs (`P1`, `P2`, ...) are "
    "only spatial identity within one capture — the same physical panel is not matched "
    "across different photos or over time yet.\n"
    "- **No thermal/hotspot capability.** RGB imagery cannot validate a thermal anomaly; "
    "HOTSPOT is intentionally excluded from the active taxonomy.\n"
    "- **Provenance grade C, unverified.** The underlying training images' rights are "
    "personal/non-commercial research use only — see `docs/PHASE_16_ARCHITECTURE_REVIEW.md` "
    "§52 for the full provenance discipline.\n"
    "- **No efficiency/output prediction, no weather/physics/telemetry integration.** Ground "
    "Inspection is detection + condition analysis only.\n"
    "- **Single-run analysis, no live sensor/IoT/SCADA connection.** Each run analyzes the "
    "photo(s) you provide; there is no automatic capture pipeline yet."
)
