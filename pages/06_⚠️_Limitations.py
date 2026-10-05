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
    "- **Condition classifier (primary signal):** MobileNetV2, Phase 19-B rooftop-adapted "
    "retrain. Measured 0.938 (45/48) on this rooftop's held-out crops, but that set is 46 CLEAN "
    "+ 2 DUSTY, so it mostly shows correct CLEAN reads: there are **no held-out BIRD_DROP or "
    "PHYSICAL_DAMAGE examples and only 2 DUSTY**, so defect detection on this rooftop is "
    "essentially unmeasured. Its labels are AI-reviewed, not owner-verified. On the original "
    "Susan-derived benchmark it scores 0.731 (the previous checkpoint: 0.752, but only 0.271 on "
    "this rooftop). See `docs/ROOFTOP_CONDITION_MIX_PHASE19B.md`.\n"
    "- **Condition detector (secondary, localized):** box-detector, measured 0.444 mAP50 / "
    "0.441 recall at the box level (localization-limited, not misrecognition — see "
    "`docs/CONDITION_MODEL_IMAGE_LEVEL_AGREEMENT.md`). Kept visible for localization value "
    "on BIRD_DROP/PHYSICAL_DAMAGE, where the classifier gives up 'where on the panel.'\n"
    "- **Panel detector (Phase 19, trained on this one rooftop):** finds 39/48 held-out panels "
    "at the default sensitivity (46/48 at the most permissive setting, with more false boxes); "
    "it never reaches 100%. It still misses panels in dense, steep-angle shots with 10+ panels "
    "crowded together. Generalization to a different rooftop is unverified. Tiling was tested "
    "earlier and produced confident FALSE positives, so it is NOT used. See "
    "`docs/ROOFTOP_PANEL_DETECTOR_PHASE19.md`.\n"
    "- **Whole-array photos now work much better, but not perfectly** — on this rooftop a "
    "6-panel frame detects 6 panels; very wide, crowded shots lose some. Closer photos remain "
    "the most reliable input.\n"
    "- **Panel identity across captures is best-effort only.** Within one run, IDs (`P1`, "
    "`P2`, ...) are spatial only; across separate runs a fixed-camera position match is "
    "estimated (Phase 17), not verified identity, and it has not been validated on real "
    "multi-visit data.\n"
    "- **No thermal/hotspot capability.** RGB imagery cannot validate a thermal anomaly; "
    "HOTSPOT is intentionally excluded from the active taxonomy.\n"
    "- **Provenance grade C, unverified.** The underlying training images' rights are "
    "personal/non-commercial research use only — see `docs/PHASE_16_ARCHITECTURE_REVIEW.md` "
    "§52 for the full provenance discipline.\n"
    "- **No efficiency/output prediction, no weather/physics/telemetry integration.** Ground "
    "Inspection is detection + condition analysis only.\n"
    "- **No live sensor/IoT/SCADA connection.** The page analyzes the photo(s) you provide. A "
    "Raspberry Pi capture agent and ingestion API exist (Phase 18) but have not been tested on "
    "real Pi hardware."
)
