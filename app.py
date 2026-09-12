"""app.py — Solar AI: ground-level, dynamic multi-panel inspection (main
landing page), with multi-capture array coverage.

This is now the app's landing page — the project's actual target deployment
is a ground-level, pole-mounted camera (not aerial/satellite imagery), so
this is what a visitor sees first. See docs/PHASE_16_ARCHITECTURE_REVIEW.md.

The separate aerial/production pipeline (weights/yolo_solar.pt + MobileNet
whole-image/per-box classifier + XGBoost predictor) still exists, unchanged
and fully working, at **pages/12_☀️_Aerial_Inspect.py** — it solves a
different problem (wide-area installation survey from above) and is kept
available but is no longer the default entry point.

This page wires up the ground-domain research pipeline, now with TWO
independent condition engines cross-checking every panel crop:

    full image
      -> dynamic ground panel detector       (weights/yolo_solar_ground_research_v1.pt)
      -> P1..PN                              (training/detection/ground_pv/dynamic_panel_detection.py)
      -> per-panel crop
      -> PRIMARY:   condition CLASSIFIER      (weights/mobilenet_condition_4class_research_v1.pth, Phase 16-B)
      -> SECONDARY: condition box-DETECTOR    (weights/yolo_solar_condition_4class_research_v2.pt, localized)
      -> CLEAN / DUSTY / BIRD_DROP / PHYSICAL_DAMAGE, or UNCERTAIN (classifier never has NO_PREDICTION;
         the detector still can)

Why two engines: docs/CONDITION_MODEL_IMAGE_LEVEL_AGREEMENT.md (Phase
16-A.1) found the box-detector's weakness was mostly localization noise,
not misrecognition. docs/CONDITION_MODEL_4CLASS_CLASSIFIER_V1.md (Phase
16-B) then trained a classifier reformulation and measured a real
improvement on the SAME held-out test set (0.752 vs. 0.727 image-level
accuracy overall, 0.794 vs. 0.359 recall on DUSTY specifically) — so the
classifier is now the PRIMARY read shown. The detector's read is kept as a
SECONDARY, localized cross-check (still real value for BIRD_DROP /
PHYSICAL_DAMAGE, where "where on the panel" matters and the classifier
gives that up). Neither replaces the other; this is a genuine hybrid, not
a silent swap.

Both checkpoints are frozen, already-trained research artifacts — this page
performs no training and modifies no weights. Every claim on screen is
qualitative: no accuracy/precision/recall/mAP is computed for any NEW image
run through this page (the 0.752/0.727/etc. figures above are the models'
OWN already-measured held-out test performance, cited, not re-derived
here), and "uncertain"/"no_prediction" is never shown as "Clean".

Multi-capture array coverage
-----------------------------
The panel detector's real working range is close-up shots of ~1-3 panels
(matching the training data and the 6 documented real reference photos) — it
does NOT reliably handle one distant/wide shot of a full 6+ panel array
(verified empirically: on such photos it detects nothing, or draws one huge
low-confidence "group" box; tiled inference was also tested and produced
confident FALSE positives on sky/brick/background, so it is NOT used here).

The honest way to cover a whole array today is multiple closer photos, each
run through the same per-image pipeline below, with results combined into
one Array Summary. Panel IDs are scoped PER CAPTURE (`C1-P1`, `C2-P1`, ...)
and never implied to be the same physical panel across captures — this
prototype does not perform cross-photo panel identity matching (see
docs/PHASE_16_ARCHITECTURE_REVIEW.md §12). If captures overlap in coverage,
the same physical panel may be counted more than once; this is stated
explicitly in the UI, never silently assumed away.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from utils.auth import require_access
from utils.exceptions import ModelLoadError
from utils.logger import get_logger
from utils.ui_theme import apply_page_chrome, empty_state

logger = get_logger(__name__)

apply_page_chrome("Ground Inspection", page_icon="🔬")
require_access()

_PANEL_WEIGHTS = Path("weights/yolo_solar_ground_research_v1.pt")
_CONDITION_WEIGHTS = Path("weights/yolo_solar_condition_4class_research_v2.pt")
_CLASSIFIER_WEIGHTS = Path("weights/mobilenet_condition_4class_research_v1.pth")
_DEMO_DIR = Path("assets/demo/ground_pv_real")

# Alphabetical — matches torchvision.datasets.ImageFolder's class-index order,
# which is how weights/mobilenet_condition_4class_research_v1.pth was trained
# (scripts/train_condition_4class_classifier.py / the Kaggle notebook).
_CLASSIFIER_CLASSES = ["BIRD_DROP", "CLEAN", "DUSTY", "PHYSICAL_DAMAGE"]
_CLASSIFIER_INPUT_SIZE = 224
_CLASSIFIER_PREDICTED_THRESHOLD = 0.45  # same "trustworthy" convention used elsewhere in this project

_CLASS_COLOR = {
    "CLEAN": (46, 160, 67),
    "DUSTY": (217, 164, 6),
    "BIRD_DROP": (220, 50, 47),
    "PHYSICAL_DAMAGE": (198, 45, 142),
}
_UNCERTAIN_COLOR = (140, 140, 140)
_NO_PRED_COLOR = (110, 110, 110)
_STATUS_BADGE = {"predicted": "✅", "uncertain": "🟡", "no_prediction": "⚪"}


# ---------------------------------------------------------------------------
# Model loading (cached — loaded once per Streamlit process, like ModelManager)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading ground-domain research models…")
def _load_ground_models():
    """Load the ground panel detector + 4-class condition model. Fails
    closed (raises) rather than silently substituting a different
    checkpoint — mirrors models/model_manager.py's discipline."""
    from ultralytics import YOLO  # noqa: PLC0415 — deferred like model_manager._load_detector

    from training.detection.ground_pv.condition_taxonomy import assert_active_taxonomy

    if not _PANEL_WEIGHTS.is_file():
        raise ModelLoadError(
            "ground panel detector",
            f"weights not found at {_PANEL_WEIGHTS}. This research checkpoint is not "
            "part of the v1 production artifact set and must be present locally.",
        )
    if not _CONDITION_WEIGHTS.is_file():
        raise ModelLoadError(
            "4-class condition model",
            f"weights not found at {_CONDITION_WEIGHTS}.",
        )

    panel_model = YOLO(str(_PANEL_WEIGHTS))
    condition_model = YOLO(str(_CONDITION_WEIGHTS))

    cond_names = (
        condition_model.names
        if isinstance(condition_model.names, dict)
        else dict(enumerate(condition_model.names))
    )
    # Fails closed: refuses to run if this checkpoint isn't EXACTLY the
    # active 4-class taxonomy (e.g. the older 5-class research checkpoint).
    assert_active_taxonomy(cond_names)

    return panel_model, condition_model


@st.cache_resource(show_spinner="Loading condition classifier (Phase 16-B)…")
def _load_condition_classifier():
    """Load the panel-crop condition CLASSIFIER (MobileNetV2, Phase 16-B) —
    a SEPARATE research checkpoint from the box-detector condition model
    above, evaluated on the same held-out test set at the image level:
    0.752 accuracy vs. the box-detector's 0.727 image-level agreement (real,
    modest win overall), and specifically 0.794 vs. 0.359 recall on DUSTY
    (the class Phase 16-A.1 flagged as most likely to benefit from dropping
    the box-localization requirement) — see
    docs/CONDITION_MODEL_4CLASS_CLASSIFIER_V1.md. Fails closed like every
    other model loader on this page."""
    import torch
    from torchvision import models

    if not _CLASSIFIER_WEIGHTS.is_file():
        raise ModelLoadError(
            "condition classifier",
            f"weights not found at {_CLASSIFIER_WEIGHTS}. Train via "
            "scripts/train_condition_4class_classifier.py or the Kaggle notebook first.",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(_CLASSIFIER_CLASSES))
    try:
        state_dict = torch.load(str(_CLASSIFIER_WEIGHTS), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as exc:  # noqa: BLE001 — surfaced as a typed, actionable error
        raise ModelLoadError(
            "condition classifier",
            f"weights exist but failed to load as a {len(_CLASSIFIER_CLASSES)}-class "
            f"MobileNetV2: {type(exc).__name__}: {exc}",
        ) from exc
    model.to(device)
    model.eval()
    return model, device


def _classify_crop(classifier_model: Any, device: Any, crop_image: Image.Image) -> dict[str, Any]:
    """Run the condition classifier on ONE panel crop. Unlike the box
    detector, a classifier always outputs a full probability distribution
    (never zero detections) — so "no_prediction" cannot occur here, but a
    low-confidence top class still gets marked "uncertain", never silently
    treated as a confident read. Preprocessing (resize 224 + center-crop +
    ImageNet normalize) matches training exactly."""
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(_CLASSIFIER_INPUT_SIZE),
        transforms.CenterCrop(_CLASSIFIER_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(crop_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(classifier_model(tensor), dim=1).squeeze(0).cpu().tolist()
    class_id = max(range(len(probs)), key=lambda i: probs[i])
    label = _CLASSIFIER_CLASSES[class_id]
    confidence = probs[class_id]
    status = "predicted" if confidence >= _CLASSIFIER_PREDICTED_THRESHOLD else "uncertain"
    return {
        "status": status, "label": label, "confidence": confidence,
        "probabilities": dict(zip(_CLASSIFIER_CLASSES, probs)),
    }


def _run_capture(
    panel_model: Any, condition_model: Any, classifier_model: Any, classifier_device: Any,
    image: Image.Image, label: str, detection_conf: float,
) -> dict[str, Any]:
    """Run the detect -> crop -> [box-detector condition + classifier condition]
    pipeline on ONE capture. Both condition engines run on the IDENTICAL crop
    the panel detector produced — a real, apples-to-apples cross-check per
    panel, not two different views of the image."""
    from training.detection.ground_pv.dynamic_panel_detection import DynamicPanelConfig
    from training.detection.ground_pv.panel_condition_pipeline import (
        ConditionInferenceConfig,
        run_panel_condition_pipeline,
    )

    report = run_panel_condition_pipeline(
        panel_model,
        condition_model,
        image,
        image_name=label,
        panel_model_name=_PANEL_WEIGHTS.name,
        condition_model_name=_CONDITION_WEIGHTS.name,
        panel_config=DynamicPanelConfig(confidence_threshold=detection_conf),
        condition_config=ConditionInferenceConfig(),
    )
    for p in report.get("panels", []):
        crop_box = tuple(p["crop"]["bbox"])
        crop = image.crop(crop_box)
        p["classifier_condition"] = _classify_crop(classifier_model, classifier_device, crop)
    return report


def _display_id(capture_idx: int, panel_id: str, multi: bool) -> str:
    """Panel ids are only capture-prefixed when there's more than one capture
    — avoids implying cross-capture identity that this prototype doesn't
    compute, while keeping single-capture display exactly as before."""
    return f"C{capture_idx}-{panel_id}" if multi else panel_id


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔬 Ground Inspection")
st.warning(
    "**Research prototype, not production-validated.** Separate model stack from the "
    "**Aerial Inspect** page (production aerial-imagery pipeline — sidebar). On the same "
    "held-out test set: the condition classifier (primary) reaches 0.752 image-level "
    "accuracy, the box-detector (secondary) 0.727 — both real, measured, neither perfect "
    "— and the panel detector can miss a large foreground panel or merge adjacent panels "
    "into one box on real photos. No accuracy is claimed for any NEW image run through "
    "this page — only the models' own already-measured test performance is cited. See "
    "`docs/PHASE_16_ARCHITECTURE_REVIEW.md`, `docs/CONDITION_MODEL_4CLASS_CLASSIFIER_V1.md`."
)

with st.expander("Two condition engines, how they differ from Aerial Inspect, and how array coverage works"):
    st.markdown(
        "- **Aerial Inspect** (production, sidebar): aerial detector (`yolo_solar.pt`, "
        "mAP50 ≈ 0.74 on aerial imagery) + whole-image/per-box MobileNet classifier "
        "(Clean/Dusty/Hotspot). Built for wide-area installation survey from above — "
        "effectively blind on close-up ground photos, which is not its job.\n"
        "- **Condition classifier (PRIMARY, this page)**: MobileNetV2 trained on panel "
        "crops to output one condition label per crop — no localization, but measurably "
        "better: 0.752 image-level test accuracy, and 0.794 recall on DUSTY specifically "
        "(vs. 0.359 for the detector) — DUSTY's diffuse, boundary-less texture turns out "
        "to suit a classifier much better than a box detector. See "
        "`docs/CONDITION_MODEL_4CLASS_CLASSIFIER_V1.md`.\n"
        "- **Condition box-detector (SECONDARY, this page)**: a detector trained specifically "
        "on ground-level close-up photos, followed by a **dynamic** number of panels (not "
        "assumed to be 4/6/8), each cropped and independently run through a 4-class condition model "
        "(Clean/Dusty/Bird-Drop/Physical-Damage). Neither checkpoint here is the "
        "production artifact, and neither is modified by this page.\n"
        "- A condition read of **'uncertain'** or **'no prediction'** is never displayed "
        "as Clean — a failed/low-confidence read is a failed read, not a diagnosis.\n"
        "- **One distant photo of a whole 6+ panel array does not work reliably** — "
        "tested empirically, the detector either finds nothing or draws one giant box "
        "around the whole array. Its real working range is close-up shots of ~1-3 panels "
        "(like a pole camera positioned near the array). **Select or upload more than one "
        "photo of the same array below** to cover all its panels — results combine into "
        "one Array Summary. Panels are **not** matched across different captures to the "
        "same physical panel (that's future work — see the architecture review), so "
        "captures with overlapping coverage may count a panel more than once."
    )

st.divider()

# ---------------------------------------------------------------------------
# 1. Choose captures (one or more photos of the same array)
# ---------------------------------------------------------------------------

st.subheader("1. Choose one or more captures")
st.caption(
    "Select/upload multiple photos of the **same physical array** (e.g. one photo per "
    "2-3 panels) to cover all its panels — each is analyzed independently and combined below."
)
source_mode = st.radio(
    "Image source",
    ["Try real reference photos", "Upload your own photo(s)"],
    horizontal=True,
)

captures_in: list[tuple[str, Image.Image]] = []

if source_mode == "Try real reference photos":
    demo_files = sorted(_DEMO_DIR.glob("*.jpeg")) if _DEMO_DIR.is_dir() else []
    if not demo_files:
        st.error(f"No reference photos found under `{_DEMO_DIR}`.")
    else:
        chosen = st.multiselect(
            "Reference photo(s)", demo_files, default=demo_files[:1], format_func=lambda p: p.name,
        )
        for p in chosen:
            captures_in.append((p.name, Image.open(p).convert("RGB")))
        st.caption(
            "These are the project's own documented real-world holdout photos — see "
            "`datasets/ground_pv_research/reports/real_world_holdout.md`. No ground-truth "
            "condition labels exist for them, so results are qualitative only, never a "
            "scored accuracy. They are six SEPARATE close-up shots (not verified to be the "
            "same physical array), so treat a multi-select here as a coverage demo, not a "
            "real single-array inspection."
        )
else:
    uploaded_files = st.file_uploader(
        "Ground-level solar panel photo(s)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Close-up / ground-level photos work best — this stack is not the aerial detector.",
    )
    for uploaded in uploaded_files or []:
        try:
            raw = uploaded.read()
            with Image.open(io.BytesIO(raw)) as im:
                im.verify()
            with Image.open(io.BytesIO(raw)) as im:
                img = ImageOps.exif_transpose(im).convert("RGB")
            captures_in.append((uploaded.name, img))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            logger.warning("Rejected invalid upload on Ground Inspection page: %s", exc)
            st.error(f"'{uploaded.name}' isn't a valid/safe image — skipped.")

if captures_in:
    cols = st.columns(min(4, len(captures_in)))
    for i, (label, img) in enumerate(captures_in):
        cols[i % len(cols)].image(img, caption=f"Capture {i + 1}: {label}", use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 2. Run
# ---------------------------------------------------------------------------

st.subheader("2. Run inspection")

detection_conf = st.slider(
    "Detection sensitivity (panel confidence threshold)",
    min_value=0.10, max_value=0.60, value=0.25, step=0.05,
    help=(
        "Lower = shows more of the model's own detections, including lower-confidence "
        "ones (never fabricated — every box shown is a real model output with its real "
        "confidence score). Higher = only the model's most confident boxes. The default "
        "(0.25) matches this project's documented more-permissive convention for ground "
        "photos (see scripts/infer_condition_on_panels.py); the panel condition pipeline's "
        "own default is a stricter 0.45, which can hide real panels on multi-panel photos."
    ),
)
st.caption(
    "⚠️ Known limitation this slider can't fix: on some wide/angled shots the detector "
    "draws **one box around a group of adjacent panels** instead of separate boxes per "
    "panel (documented in `datasets/ground_pv_research/reports/real_world_holdout.md`, "
    "image 6). Lowering sensitivity reveals more *separate* detections the model already "
    "computed — it does not split an existing group box into individual panels. This is "
    "exactly why wide/full-array photos work poorly and multiple closer captures are "
    "recommended instead (see §1)."
)

run_clicked = st.button("🔍 Run Ground Inspection", type="primary", disabled=not captures_in)

if run_clicked and captures_in:
    try:
        panel_model, condition_model = _load_ground_models()
        classifier_model, classifier_device = _load_condition_classifier()
    except (ModelLoadError, ValueError, ImportError) as exc:
        st.error(f"Could not load the ground-domain research models: {exc}")
        st.stop()

    results: list[dict[str, Any]] = []
    with st.spinner(f"Detecting panels and analyzing condition across {len(captures_in)} capture(s)…"):
        for label, img in captures_in:
            try:
                report = _run_capture(
                    panel_model, condition_model, classifier_model, classifier_device,
                    img, label, detection_conf,
                )
            except Exception:  # noqa: BLE001 — never crash the app on a research-page inference failure
                logger.exception("Ground inspection pipeline failed for capture '%s'", label)
                st.error(f"Capture '{label}': an unexpected error occurred running the pipeline — skipped.")
                continue
            results.append({"label": label, "image": img, "report": report})

    if results:
        st.session_state["ground_inspection_captures"] = results
    else:
        st.error("No capture produced a result. Nothing to show.")
        st.stop()

st.divider()

# ---------------------------------------------------------------------------
# 3. Results
# ---------------------------------------------------------------------------

st.subheader("3. Results")

captures: list[dict[str, Any]] | None = st.session_state.get("ground_inspection_captures")
if not captures:
    empty_state("Choose capture(s) above and click **Run Ground Inspection** to see results.")
    st.stop()

multi = len(captures) > 1

# ---------------------------------------------------------------------------
# Combined array summary — flattens every capture's panels into one list.
# PRIMARY condition signal is now the CLASSIFIER (docs/CONDITION_MODEL_4CLASS_CLASSIFIER_V1.md:
# 0.752 image-level accuracy vs. the box-detector's 0.727, and dramatically
# better on DUSTY specifically: 0.794 vs. 0.359 recall). The box-detector's
# condition read is kept as "detector_condition" for the secondary,
# localized cross-check shown per panel below — not discarded, since it
# still carries real localization value the classifier gives up. Captures
# with 0 panels are reported, never silently dropped from the picture.
# ---------------------------------------------------------------------------
all_panels: list[dict[str, Any]] = []  # {"display_id", "condition": {...} (classifier), "detector_condition": {...}}
zero_panel_captures: list[str] = []
any_geometry_warning = False

for idx, cap in enumerate(captures, start=1):
    report = cap["report"]
    if report["panel_count"] == 0:
        zero_panel_captures.append(cap["label"])
        continue
    if report["analysis_status"] == "warning":
        any_geometry_warning = True
    for p in report["panels"]:
        all_panels.append({
            "display_id": _display_id(idx, p["panel_id"], multi),
            "condition": p["classifier_condition"],
            "detector_condition": p["condition"],
        })

if multi:
    st.info(
        f"📷 **{len(captures)} captures combined**, {len(all_panels)} total panel detection(s). "
        "Panel IDs are prefixed per capture (`C1-P1`, `C2-P1`, ...) — **no matching is "
        "performed across captures**, so if two photos show the same physical panel it "
        "will be counted twice here. This is a coverage aggregation, not a deduplicated "
        "array inventory."
    )

if zero_panel_captures:
    st.error(f"🚫 **0 panels detected in:** {', '.join(zero_panel_captures)} — reported as a detection failure for that capture, not skipped silently.")

if not all_panels:
    with st.expander("Raw pipeline JSON (all captures)"):
        st.json([c["report"] for c in captures])
    st.stop()

status_counts = {"predicted": 0, "uncertain": 0, "no_prediction": 0}
for p in all_panels:
    status_counts[p["condition"]["status"]] += 1

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Captures", len(captures))
m2.metric("Panel detections", len(all_panels))
m3.metric("Predicted", status_counts["predicted"])
m4.metric("Uncertain", status_counts["uncertain"])
m5.metric("No prediction", status_counts["no_prediction"])

# ---------------------------------------------------------------------------
# Attention summary — a plain-language inspection readout up top, before the
# technical per-panel detail. Only a "predicted" (>=45% confidence) non-CLEAN
# read counts as a flagged issue; "uncertain"/"no_prediction" are surfaced
# separately as reads that need a better photo — NEVER folded into either
# "confirmed issue" or "Clean".
# ---------------------------------------------------------------------------
flagged = [p for p in all_panels if p["condition"]["status"] == "predicted" and p["condition"]["label"] != "CLEAN"]
unread = [p for p in all_panels if p["condition"]["status"] in ("uncertain", "no_prediction")]
confirmed_clean = [p for p in all_panels if p["condition"]["status"] == "predicted" and p["condition"]["label"] == "CLEAN"]

if flagged:
    detail = ", ".join(f"**{p['display_id']}** {p['condition']['label']} ({p['condition']['confidence']:.0%})" for p in flagged)
    st.error(f"🚨 **{len(flagged)} panel(s) show a likely issue:** {detail}")
if unread:
    detail = ", ".join(p["display_id"] for p in unread)
    st.warning(
        f"🔍 **{len(unread)} panel(s) could not be confidently read** ({detail}) — recommend a "
        "closer/clearer photo of these panels. Not shown as Clean and not shown as a confirmed issue."
    )
if confirmed_clean and not flagged:
    detail = ", ".join(p["display_id"] for p in confirmed_clean)
    st.success(f"✅ **{len(confirmed_clean)} panel(s) confidently read as Clean:** {detail}")
elif confirmed_clean:
    st.caption(f"Also confidently Clean: {', '.join(p['display_id'] for p in confirmed_clean)}")

if any_geometry_warning:
    st.warning(
        "⚠️ At least one panel's detection geometry looks suspicious (overlap, extreme "
        "size, or isolation vs. the rest of that capture's array) — flagged per-panel "
        "below, never silently dropped."
    )

st.divider()

# ---------------------------------------------------------------------------
# Per-capture detail
# ---------------------------------------------------------------------------

for idx, cap in enumerate(captures, start=1):
    label, source_img, report = cap["label"], cap["image"], cap["report"]
    heading = f"Capture {idx}: {label}" if multi else "Detail"
    st.markdown(f"#### {heading}")

    if report["panel_count"] == 0:
        st.error(f"0 panels detected (failure_reason: `{report.get('failure_reason', 'unknown')}`).")
        with st.expander(f"Raw pipeline JSON — capture {idx}"):
            st.json(report)
        continue

    annotated = source_img.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for p in report["panels"]:
        b = p["detection"]["bbox"]
        box = [b["x1"], b["y1"], b["x2"], b["y2"]]
        cond = p["classifier_condition"]  # PRIMARY signal — see docs/CONDITION_MODEL_4CLASS_CLASSIFIER_V1.md
        did = _display_id(idx, p["panel_id"], multi)
        if cond["status"] == "predicted":
            color = _CLASS_COLOR.get(cond["label"], (56, 176, 255))
            box_label = f"{did} {cond['label']} {cond['confidence']:.0%}"
        else:
            color = _UNCERTAIN_COLOR
            box_label = f"{did} uncertain ({cond['label']} {cond['confidence']:.0%})"
        width = 4 if p["detection"]["status"] == "suspicious" else 3
        draw.rectangle(box, outline=color, width=width)
        draw.text((box[0] + 3, max(0, box[1] - 14)), box_label, fill=color)

    st.image(
        annotated,
        caption=(
            "Panel boxes colored by the CLASSIFIER's condition read (primary signal — see "
            "'Two condition engines' above). Thicker outline = suspicious detection geometry."
        ),
        use_container_width=True,
    )

    for p in report["panels"]:
        did = _display_id(idx, p["panel_id"], multi)
        with st.container(border=True):
            cols = st.columns([1, 3])
            crop_box = tuple(p["crop"]["bbox"])
            cols[0].image(source_img.crop(crop_box), caption=did, use_container_width=True)

            det = p["detection"]
            cls_cond = p["classifier_condition"]
            det_cond = p["condition"]
            geometry_note = "suspicious geometry" if det["status"] == "suspicious" else "valid geometry"
            cols[1].markdown(f"**{did}** — panel detection confidence {det['confidence']:.0%} ({geometry_note})")

            # PRIMARY: classifier read — always has a label (no "no_prediction" state possible).
            badge = _STATUS_BADGE[cls_cond["status"]]
            if cls_cond["status"] == "predicted":
                cols[1].markdown(f"{badge} **Condition (classifier): {cls_cond['label']}** — confidence {cls_cond['confidence']:.0%}")
            else:
                cols[1].markdown(
                    f"{badge} Condition (classifier) **uncertain** — top guess **{cls_cond['label']}** at only "
                    f"{cls_cond['confidence']:.0%} confidence (below the 45% trustworthy threshold)"
                )

            # SECONDARY: box-detector's LOCALIZED read — kept visible for BIRD_DROP/PHYSICAL_DAMAGE-style
            # defects where knowing WHERE on the panel still has value the classifier gives up.
            det_badge = _STATUS_BADGE[det_cond["status"]]
            if det_cond["status"] == "predicted":
                det_line = f"{det_badge} Detector (localized, secondary): **{det_cond['label']}** — {det_cond['confidence']:.0%}"
            elif det_cond["status"] == "uncertain":
                det_line = f"{det_badge} Detector (localized, secondary): uncertain — top guess {det_cond['label']} at {det_cond['confidence']:.0%}"
            else:
                det_line = f"{det_badge} Detector (localized, secondary): no prediction"
            if cls_cond["status"] == "predicted" and det_cond.get("label") not in (None, cls_cond["label"]):
                det_line += "  — ⚡ **disagrees with the classifier**"
            cols[1].caption(det_line)

            if p["crop"]["warnings"]:
                cols[1].caption("Crop quality warnings: " + ", ".join(p["crop"]["warnings"]))

    with st.expander(f"Raw pipeline JSON — capture {idx}"):
        st.json(report)

st.divider()
st.caption(
    f"Pipeline: full image → dynamic ground panel detector (`{_PANEL_WEIGHTS.name}`) → "
    f"per-panel crop → PRIMARY condition classifier (`{_CLASSIFIER_WEIGHTS.name}`) + "
    f"SECONDARY condition box-detector (`{_CONDITION_WEIGHTS.name}`), run independently "
    "per capture and combined above. All three are frozen research checkpoints, not "
    "production artifacts. No accuracy/precision/recall/mAP is computed or claimed for "
    "any NEW image here — only the models' own already-measured held-out test "
    "performance is cited — and no panel identity is matched across captures. See "
    "`docs/CONDITION_MODEL_4CLASS_V1.md`, `docs/CONDITION_MODEL_IMAGE_LEVEL_AGREEMENT.md`, "
    "`docs/CONDITION_MODEL_4CLASS_CLASSIFIER_V1.md`, and `docs/PHASE_16_ARCHITECTURE_REVIEW.md` "
    "for the full evidence trail and known failure modes."
)
