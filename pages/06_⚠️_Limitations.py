"""pages/06_⚠️_Limitations.py — Honest, explicit capability disclosure.

Covers BOTH pipelines now that Ground Inspection exists. Static content
cross-checked against the live model status below (the class list is read
from models.model_manager, not hardcoded twice) so this page can never
silently drift out of sync with what's actually running.
"""
from __future__ import annotations

import streamlit as st

from models.model_manager import model_manager
from utils.auth import require_access
from utils.ui_theme import apply_page_chrome

apply_page_chrome("Limitations")
require_access()

st.title("⚠️ Limitations")
st.caption("What this deployment can and cannot do right now — kept visible, not buried.")

st.info(
    "This page covers **both** pipelines. **Ground Inspection** (the landing page) is for "
    "close-up/ground-level photos — its limitations are below. **Aerial Inspect** (sidebar) "
    "is the legacy production pipeline for aerial/satellite imagery — its limitations, "
    "including the close-up detection weakness that Ground Inspection exists to address, "
    "are in the collapsed section further down."
)

st.divider()

# ---------------------------------------------------------------------------
# Ground Inspection limitations — PRIMARY pipeline
# ---------------------------------------------------------------------------
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
    "- **No efficiency/output prediction.** Ground Inspection involves no weather, physics, "
    "or XGBoost — that concept only exists on the legacy Aerial Inspect pipeline."
)

st.divider()

with st.expander("Aerial Inspect limitations (legacy production pipeline)"):
    mn_status = model_manager.mobilenet_status
    active = set(mn_status["active_labels"])
    six_class_taxonomy = mn_status["six_class_labels"]
    future_only = [c for c in six_class_taxonomy if c not in active]

    st.markdown("**Classifier coverage**")
    if mn_status["state"] == "six_class":
        st.success("✅ Full six-class classifier is active. No classes are unavailable.")
    elif mn_status["state"] == "v1":
        st.success(
            f"**Supported now (v1):** solar-panel detection, {', '.join(sorted(active))}.\n\n"
            f"**Not supported in v1:** {', '.join(future_only)}."
        )
        for cls in future_only:
            st.markdown(f"- **{cls}** — not supported in v1")
        st.markdown(
            "This is v1's intentionally frozen scope, not an incomplete or broken system. "
            "The full six-class classifier (`weights/mobilenet_solar.pth`) remains a "
            "**documented future roadmap item** — see `training/classification/DATASET_SOURCES.md` "
            "for the exact provenance status of every class, including why the three classes "
            "above are not part of this release (no licensed/accessible dataset yet, or "
            "access pending the dataset owner's approval)."
        )
    else:
        st.error("❌ No classifier is active at all — classification cannot run.")

    st.markdown("**Detection coverage**")
    st.error(
        "❌ **Panel-level detection is not reliable on close-up/ground-level photos** — the "
        "production aerial detector (`weights/yolo_solar.pt`) was trained and validated on "
        "aerial/satellite imagery (BDAPPV). Independent measurements: ~2.6% detection rate on "
        "a 156-image close-up sample, and 0/16 on a separate 16-image real-photo validation "
        "set. **This is exactly why the Ground Inspection page exists** — it uses a "
        "separate detector trained specifically for this domain."
    )
    st.markdown(
        "**What this means in practice (Aerial Inspect only):**\n"
        "- When YOLO detects a panel, the app shows a real bounding-box overlay and runs "
        "genuine panel-crop classification, clearly labeled as panel-level.\n"
        "- When YOLO detects nothing (the common case for a close-up upload), **no crop is "
        "manufactured and no whole-image result is presented as if it were panel-level.** "
        "The app still runs whole-image classification independently and labels it "
        "explicitly: *\"Whole-image classification — not panel-level diagnosis.\"*\n"
        "- An **experimental** ground-domain-adapted checkpoint "
        "(`weights/yolo_solar_domain_v1.pt`) was fine-tuned and evaluated on this pipeline. "
        "It shows a real, measured improvement on a held-out ground-level benchmark "
        "(mAP50 0.000 → 0.445) and does not meaningfully regress aerial performance, but an "
        "end-to-end real-photo validation found it still produces **zero** detections at the "
        "deployed confidence threshold on the same 16-image sample as the production model. "
        "It is **not used in this deployment**. Full evidence: `docs/RELEASE_v1.0.0.md`."
    )

    st.markdown("**Other known limitations (Aerial Inspect only)**")
    st.markdown(
        "- **Efficiency/output estimates depend on environmental inputs and model "
        "assumptions.** They are model predictions, not measured sensor readings — see "
        "the **Environment** page for the exact inputs used in your last inspection.\n"
        "- **Weather data falls back to configured defaults** when the OpenWeatherMap API "
        "is unavailable, rather than blocking the analysis — this is always disclosed "
        "on the Environment page when it happens.\n"
        "- **No sensor/telemetry integration.** All inputs come from the uploaded image "
        "plus a weather API lookup or manual entry — there is no live IoT/SCADA "
        "connection to a real installation.\n"
        "- **Single-image analysis.** Each inspection analyzes one uploaded photo; there "
        "is no temporal tracking of a specific physical panel across multiple "
        "inspections beyond what the History page's timestamps show.\n"
        "- **Efficiency-loss prediction requires the XGBoost artifact.** When it's "
        "absent (see **Model Status**), detection and classification still run, but "
        "efficiency/output figures are reported as unavailable, never estimated as zero.\n"
        "- **No XGBoost artifact currently exists, and none is planned until a genuine "
        "dataset is found.** Investigated on 2026-09-04: no dataset was found that pairs "
        "a real fault classification with paired environmental telemetry and a genuinely "
        "measured efficiency-loss value — see `training/prediction/DATASET_SOURCES.md`. "
        "This is a real, current limitation, not a placeholder pending routine training."
    )
