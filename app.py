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
      -> dynamic ground panel detector       (weights/yolo_solar_rooftop_v1.pt, Phase 19)
      -> P1..PN                              (training/detection/ground_pv/dynamic_panel_detection.py)
      -> per-panel crop
      -> PRIMARY:   condition CLASSIFIER      (weights/mobilenet_condition_4class_rooftop_mix_oversample1_v1.pth, Phase 19-B)
      -> SECONDARY: condition box-DETECTOR    (weights/yolo_solar_condition_4class_research_v2.pt, localized)
      -> CLEAN / DUSTY / BIRD_DROP / PHYSICAL_DAMAGE, or UNCERTAIN (classifier never has NO_PREDICTION;
         the detector still can)

Why two engines: docs/CONDITION_MODEL_IMAGE_LEVEL_AGREEMENT.md (Phase
16-A.1) found the box-detector's weakness was mostly localization noise,
not misrecognition. docs/CONDITION_MODEL_4CLASS_CLASSIFIER_V1.md (Phase
16-B) then trained a classifier reformulation and measured a real
improvement on that SAME held-out (Susan-derived) test set (0.752 vs. 0.727
image-level accuracy overall) — so the classifier is the PRIMARY read shown.
The detector's read is kept as a SECONDARY, localized cross-check (still
real value for BIRD_DROP / PHYSICAL_DAMAGE, where "where on the panel"
matters and the classifier gives that up). Neither replaces the other;
this is a genuine hybrid, not a silent swap.

docs/ROOFTOP_CONDITION_MIX_PHASE19B.md (Phase 19-B) then found that
Phase 16-B checkpoint scored only 0.271 (13/48) on this app's OWN rooftop's
held-out photos — a domain-shift failure, systematically misreading CLEAN
panels as DUSTY. Retrained blended into the same base dataset (oversampled
rooftop crops); the default PRIMARY classifier is that retrain. Its 0.938
(45/48) figure came from 48 AI-labelled crops (46 CLEAN + 2 DUSTY, no defect
examples) that were later deleted; a direct check against the owner's own
defect marks showed it called all 5 marked bird-drop panels CLEAN, so that
figure says nothing about defect detection. Disclosed cost: 0.731 vs. 0.752
on the original Susan-derived benchmark.

Both checkpoints are frozen, already-trained research artifacts — this page
performs no training and modifies no weights. Every claim on screen is
qualitative: no accuracy/precision/recall/mAP is computed for any NEW image
run through this page (the figures above are the models' OWN already-
measured held-out test performance, cited, not re-derived here), and
"uncertain"/"no_prediction" is never shown as "Clean".

Multi-capture array coverage
-----------------------------
The Phase 19 panel detector (trained on the owner's own rooftop photos) handles
close-ups and full-array frames on that rooftop: 39/48 held-out panels found at
the default sensitivity, never 100%, and it still loses panels in dense,
steep-angle shots with 10+ panels (docs/ROOFTOP_PANEL_DETECTOR_PHASE19.md).
Closer, overlapping captures remain the most reliable input. Tiled inference
was tested earlier and produced confident FALSE positives on sky/brick/
background, so it is NOT used here.

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

from services import ground_pipeline
from utils.auth import require_access
from utils.exceptions import ModelLoadError
from utils.logger import get_logger
from utils.ui_theme import apply_page_chrome, empty_state

logger = get_logger(__name__)

apply_page_chrome("Ground Inspection", page_icon="🔬")
require_access()

_PANEL_WEIGHTS = ground_pipeline.PANEL_WEIGHTS
_CONDITION_WEIGHTS = ground_pipeline.CONDITION_WEIGHTS
_CLASSIFIER_WEIGHTS = ground_pipeline.CLASSIFIER_WEIGHTS
_DEMO_DIR = Path("assets/demo/ground_pv_real")

# Soft cap on captures per run. Each capture runs a real panel detector +
# two condition-model passes per detected panel on CPU-plausible hardware
# (no GPU assumed for this page) — an unbounded multi-file upload would let
# one run silently take many minutes with only a single spinner for
# feedback. Enforced as a clear, honest truncation (never a crash, never a
# silent partial run) — see the truncation message where this is used.
_MAX_CAPTURES_PER_RUN = ground_pipeline.MAX_CAPTURES_PER_RUN

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
    return ground_pipeline.load_ground_models()


@st.cache_resource(show_spinner="Loading condition classifier…")
def _load_condition_classifier(backend: str):
    return ground_pipeline.load_condition_classifier(backend)


_run_capture = ground_pipeline.run_capture
_display_id = ground_pipeline.display_id
_summarize_panels = ground_pipeline.summarize_panels


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔬 Ground Inspection")
st.warning(
    "**Research prototype, not validated on this rooftop.** Separate model stack from the "
    "**Aerial Inspect** page (production aerial-imagery pipeline — sidebar). Neither condition "
    "classifier has been verified on the target camera. The default (rooftop-adapted MobileNetV2) "
    "was tuned on AI-labelled photos that the owner's own defect marks later contradicted; in a "
    "direct check it called all 5 owner-marked bird-drop panels CLEAN with high confidence, so a "
    "CLEAN result here does **not** mean the panel is clean. The optional web-trained ensemble "
    "measured macro-F1 0.878 only on public web photos (out-of-fold, same sources), not on a "
    "rooftop. The panel detector can also miss or merge panels. **Do not use these outputs for "
    "alerts or maintenance decisions** until fixed-camera photos have been used to measure accuracy. "
    "See `docs/ROOFTOP_CONDITION_MIX_PHASE19B.md` and "
    "`datasets/ground_pv_research/web_condition_v2/README.md`."
)

_CLASSIFIER_LABELS = {
    "rooftop_mix": "MobileNetV2, rooftop-adapted (default)",
    "web_ensemble": "Web-trained 5-model ensemble (research, ~5x slower)",
    "web_ensemble_lite": "Web-trained 2-model ensemble (research, ~2x slower)",
}
_backend_options = ["rooftop_mix"] + [b for b in ("web_ensemble", "web_ensemble_lite") if ground_pipeline.ensemble_available(b)]
_CLASSIFIER_BACKEND = st.selectbox(
    "Condition classifier (primary)", _backend_options, format_func=_CLASSIFIER_LABELS.get,
    help="The web ensemble loads only if weights/web_condition_ensemble_v1/ is present. Neither option is "
         "validated on the target rooftop camera.",
)

with st.expander("Two condition engines, how they differ from Aerial Inspect, and how array coverage works"):
    st.markdown(
        "- **Aerial Inspect** (production, sidebar): aerial detector (`yolo_solar.pt`, "
        "mAP50 ≈ 0.74 on aerial imagery) + whole-image/per-box MobileNet classifier "
        "(Clean/Dusty/Hotspot). Built for wide-area installation survey from above — "
        "effectively blind on close-up ground photos, which is not its job.\n"
        "- **Condition classifier (PRIMARY, this page)**: MobileNetV2 trained on panel "
        "crops to output one condition label per crop — no localization. The Phase 16-B "
        "version measurably beat the box-detector on DUSTY (0.794 vs. 0.359 recall) but, "
        "per Phase 19-B, scored only 0.271 on THIS rooftop's own held-out photos — a "
        "domain-shift failure. The default checkpoint is a Phase 19-B retrain blended "
        "with this rooftop's own photos. Its earlier **0.938** came from AI-labelled crops with "
        "no defect examples (46 CLEAN + 2 DUSTY); checked later against the owner's own defect "
        "marks it called **all 5 marked bird-drop panels CLEAN**, so it is NOT evidence of "
        "defect detection. 0.731 (down from 0.752) on the original Susan-derived benchmark. See "
        "`docs/CONDITION_MODEL_4CLASS_CLASSIFIER_V1.md`, `docs/ROOFTOP_CONDITION_MIX_PHASE19B.md`.\n"
        "- **Condition box-detector (SECONDARY, this page)**: a detector trained specifically "
        "on ground-level close-up photos, followed by a **dynamic** number of panels (not "
        "assumed to be 4/6/8), each cropped and independently run through a 4-class condition model "
        "(Clean/Dusty/Bird-Drop/Physical-Damage). Neither checkpoint here is the "
        "production artifact, and neither is modified by this page.\n"
        "- A condition read of **'uncertain'** or **'no prediction'** is never displayed "
        "as Clean — a failed/low-confidence read is a failed read, not a diagnosis.\n"
        "- **Whole-array photos work much better since Phase 19, but not perfectly** — "
        "on this rooftop a 6-panel frame detects 6 panels (39/48 held-out panels overall); "
        "dense, steep-angle shots with 10+ panels still lose some. Closer photos remain "
        "the most reliable input. **Select or upload more than one "
        "photo of the same array below** to cover all its panels — results combine into "
        "one Array Summary. Within ONE run, panels are **not** matched across different "
        "captures to the same physical panel (no cross-capture matching within a single "
        "run — that's still future work), so overlapping coverage may count a panel more "
        "than once in that run's total.\n"
        "- **Across separate runs, panels ARE matched** (Phase 17) — a best-effort, "
        "estimated match by fixed-camera position, not verified identity. Run Ground "
        "Inspection again later on the same array and the **Panel History** page (sidebar) "
        "shows which panels were matched and any detected condition change (e.g. "
        "CLEAN → DUSTY). See `docs/PANEL_IDENTITY_TRACKING.md` for exactly what this "
        "assumes and how it degrades when the camera isn't genuinely fixed."
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

if len(captures_in) > _MAX_CAPTURES_PER_RUN:
    st.warning(
        f"⚠️ {len(captures_in)} photos selected — using only the first {_MAX_CAPTURES_PER_RUN} "
        f"for this run (each capture runs a full detection + condition pass; more than "
        f"{_MAX_CAPTURES_PER_RUN} at once would make one run take a very long time with no way "
        "to see partial progress). Run the rest as a separate inspection."
    )
    captures_in = captures_in[:_MAX_CAPTURES_PER_RUN]

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
        classifier_model, classifier_device = _load_condition_classifier(_CLASSIFIER_BACKEND)
    except (ModelLoadError, ValueError, ImportError) as exc:
        st.error(f"Could not load the ground-domain research models: {exc}")
        st.stop()

    results: list[dict[str, Any]] = []
    n = len(captures_in)
    progress = st.progress(0.0, text=f"Starting analysis of {n} capture(s)…")
    for i, (label, img) in enumerate(captures_in, start=1):
        progress.progress((i - 1) / n, text=f"Analyzing capture {i}/{n}: {label}")
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
    progress.progress(1.0, text=f"Done — {len(results)}/{n} capture(s) analyzed.")

    if results:
        st.session_state["ground_inspection_captures"] = results
        summary = _summarize_panels(results)
        run_id: int | None = None
        try:
            from services import storage
            run_id = storage.record_ground_inspection(
                results, flagged=summary["flagged"], unread=summary["unread"],
                confirmed_clean=summary["confirmed_clean"],
            )
        except Exception:  # noqa: BLE001 — history recording must never break the live result
            logger.exception("Failed to record Ground Inspection run to local history")
            st.warning(
                "⚠️ This run completed successfully (shown below), but could not be saved "
                "to history — it will not appear on Overview, History, Analytics, or Alerts."
            )

        # Persistent panel identity (Phase 17) — best-effort geometry match
        # against the previous run, separate try/except so a failure here
        # never costs the already-saved ground_inspections row above.
        if run_id is not None:
            try:
                identity_result = storage.record_panel_identities(run_id, results)
                st.session_state["ground_inspection_identity_result"] = identity_result
            except Exception:  # noqa: BLE001 — identity tracking must never break the live result
                logger.exception("Failed to match panel identities for run %d", run_id)
                st.session_state["ground_inspection_identity_result"] = None
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

summary = _summarize_panels(captures)
multi = summary["multi"]
all_panels = summary["all_panels"]
zero_panel_captures = summary["zero_panel_captures"]
any_geometry_warning = summary["any_geometry_warning"]
flagged = summary["flagged"]
unread = summary["unread"]
confirmed_clean = summary["confirmed_clean"]

# PRIMARY condition signal is the CLASSIFIER — Phase 19-B rooftop-domain retrain
# (docs/ROOFTOP_CONDITION_MIX_PHASE19B.md: 0.938 on this rooftop's own held-out
# photos, vs. 0.271 for the Phase 16-B predecessor there; 0.731 vs. 0.752 on the
# original Susan-derived benchmark, a deliberate disclosed trade). The box-detector's
# condition read is kept as "detector_condition" for the secondary,
# localized cross-check shown per panel below — not discarded, since it
# still carries real localization value the classifier gives up. Captures
# with 0 panels are reported, never silently dropped from the picture.
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

# ---------------------------------------------------------------------------
# Panel identity (Phase 17) — best-effort geometry match against the
# previous run. See docs/PANEL_IDENTITY_TRACKING.md for what this assumes
# (a genuinely fixed camera, same capture order each run) and how it
# degrades when that assumption doesn't hold. Never a verified identity —
# every matched/new claim here is an estimate, and changes are only ever
# flagged between two CONFIDENT reads, never from an uncertain one.
# ---------------------------------------------------------------------------
identity_result = st.session_state.get("ground_inspection_identity_result")
if identity_result is not None:
    id_parts = []
    if identity_result["new_panel_ids"]:
        id_parts.append(f"🆕 {len(identity_result['new_panel_ids'])} new panel(s) now tracked ({', '.join(identity_result['new_panel_ids'])})")
    if identity_result["matched"]:
        id_parts.append(f"🔄 {len(identity_result['matched'])} panel(s) matched to a previous capture")
    if id_parts:
        st.info(
            " · ".join(id_parts) + " — **estimated** by fixed-camera position matching, not verified. "
            "See the **Panel History** page for the full per-panel timeline."
        )
    for c in identity_result["changes"]:
        st.warning(
            f"🔔 **{c['persistent_id']}** ({c['display_id']}) changed: "
            f"**{c['previous_label']} ({c['previous_confidence']:.0%}) → {c['new_label']} ({c['new_confidence']:.0%})**"
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
    f"per-panel crop → PRIMARY condition classifier (`{_CLASSIFIER_LABELS[_CLASSIFIER_BACKEND]}`"
    f"{'' if _CLASSIFIER_BACKEND != 'rooftop_mix' else ' · `' + _CLASSIFIER_WEIGHTS.name + '`'}) + "
    f"SECONDARY condition box-detector (`{_CONDITION_WEIGHTS.name}`), run independently "
    "per capture and combined above. All three are frozen research checkpoints, not "
    "production artifacts. No accuracy/precision/recall/mAP is computed or claimed for "
    "any NEW image here — only the models' own already-measured held-out test "
    "performance is cited — and no panel identity is matched across captures. See "
    "`docs/CONDITION_MODEL_4CLASS_V1.md`, `docs/CONDITION_MODEL_IMAGE_LEVEL_AGREEMENT.md`, "
    "`docs/CONDITION_MODEL_4CLASS_CLASSIFIER_V1.md`, and `docs/PHASE_16_ARCHITECTURE_REVIEW.md` "
    "for the full evidence trail and known failure modes."
)
