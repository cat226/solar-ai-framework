"""services/ground_pipeline.py — the ground-domain detect -> crop -> classify
pipeline, factored out of app.py so it has exactly ONE implementation shared
by every caller that needs to run it:

    - app.py (Streamlit UI, manual photo upload, camera_id=None)
    - pi_agent/capture_and_send.py (Raspberry Pi camera devices, Phase 18 —
      "capture + analyze + log": the Pi runs this SAME code locally and only
      sends the already-computed result to the ingestion API, so the model
      logic a Pi camera sees is never a second, divergent copy)

This module has NO Streamlit dependency (no `st.cache_resource`, no
`st.session_state`) so it can be imported from a plain Python script running
on a Pi. Callers that want caching wrap these functions themselves —
app.py's `@st.cache_resource`-decorated loaders in app.py call straight
through to `load_ground_models()` / `load_condition_classifier()` below.

Both checkpoints are frozen, already-trained research artifacts — this
module performs no training and modifies no weights. See
docs/PHASE_16_ARCHITECTURE_REVIEW.md and docs/CONDITION_MODEL_4CLASS_CLASSIFIER_V1.md
for the full evidence trail this pipeline's design decisions are based on.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image

from utils.exceptions import ModelLoadError
from utils.logger import get_logger

logger = get_logger(__name__)

PANEL_WEIGHTS = Path("weights/yolo_solar_rooftop_v1.pt")
# Predecessor kept for provenance/comparison, no longer the active detector:
# weights/yolo_solar_ground_research_v1.pt (Susan-only; missed 42/48 panels on the
# Phase 19 rooftop holdout — see docs/ROOFTOP_PANEL_DETECTOR_PHASE19.md).
CONDITION_WEIGHTS = Path("weights/yolo_solar_condition_4class_research_v2.pt")
CLASSIFIER_WEIGHTS = Path("weights/mobilenet_condition_4class_rooftop_mix_oversample1_v1.pth")
# Predecessor kept for provenance/comparison, no longer the active classifier:
# weights/mobilenet_condition_4class_research_v1.pth (Susan-only; measured only 27.1%
# accuracy — 13/48 — on this rooftop's own held-out photos, systematically misreading
# CLEAN panels as DUSTY/PHYSICAL_DAMAGE at high confidence, a domain-shift failure —
# see docs/ROOFTOP_CONDITION_MIX_PHASE19B.md).

# Alphabetical — matches torchvision.datasets.ImageFolder's class-index order,
# which is how weights/mobilenet_condition_4class_research_v1.pth was trained
# (scripts/train_condition_4class_classifier.py / the Kaggle notebook); the
# rooftop MIX retrain inherits the identical class order (see that script's
# base_train_ds.class_to_idx).
CLASSIFIER_CLASSES = ["BIRD_DROP", "CLEAN", "DUSTY", "PHYSICAL_DAMAGE"]
CLASSIFIER_INPUT_SIZE = 224
CLASSIFIER_PREDICTED_THRESHOLD = 0.45  # same "trustworthy" convention used elsewhere in this project

# Condition-classifier backends. "rooftop_mix" (default) is the MobileNetV2 above. "web_ensemble" is the final
# web-only 5-backbone ensemble (training/detection/ground_pv/web_ensemble.py): out-of-fold macro-F1 0.878 on WEB
# photos, never validated on the target rooftop camera. It is opt-in and is never selected silently.
CLASSIFIER_BACKENDS = ("rooftop_mix", "web_ensemble")
CLASSIFIER_BACKEND_ENV = "SOLAR_AI_CLASSIFIER"
ENSEMBLE_MANIFEST = Path("weights/web_condition_ensemble_v1/ensemble.json")


def active_classifier_backend(backend: str | None = None) -> str:
    """Resolve the classifier backend: explicit arg, else $SOLAR_AI_CLASSIFIER, else "rooftop_mix".
    An unknown value fails closed instead of falling back to a different model."""
    chosen = (backend or os.environ.get(CLASSIFIER_BACKEND_ENV) or "rooftop_mix").strip().lower()
    if chosen not in CLASSIFIER_BACKENDS:
        raise ModelLoadError("condition classifier",
                             f"unknown classifier backend {chosen!r}; expected one of {CLASSIFIER_BACKENDS}")
    return chosen


def ensemble_available() -> bool:
    return ENSEMBLE_MANIFEST.is_file()


# Soft cap on captures per run/ingestion batch — shared by app.py (Streamlit
# upload) and api/main.py (Pi ingestion) so neither can be tricked into an
# unbounded processing/storage batch by one oversized request.
MAX_CAPTURES_PER_RUN = 20


def load_ground_models() -> tuple[Any, Any]:
    """Load the ground panel detector + 4-class condition box-detector. Fails
    closed (raises) rather than silently substituting a different
    checkpoint — mirrors models/model_manager.py's discipline. Not cached
    here — callers that run repeatedly in one process (app.py, the Pi agent)
    should cache the return value themselves."""
    from ultralytics import YOLO  # noqa: PLC0415 — deferred like model_manager._load_detector

    from training.detection.ground_pv.condition_taxonomy import assert_active_taxonomy

    if not PANEL_WEIGHTS.is_file():
        raise ModelLoadError(
            "ground panel detector",
            f"weights not found at {PANEL_WEIGHTS}. This research checkpoint is not "
            "part of the v1 production artifact set and must be present locally.",
        )
    if not CONDITION_WEIGHTS.is_file():
        raise ModelLoadError(
            "4-class condition model",
            f"weights not found at {CONDITION_WEIGHTS}.",
        )

    panel_model = YOLO(str(PANEL_WEIGHTS))
    condition_model = YOLO(str(CONDITION_WEIGHTS))

    cond_names = (
        condition_model.names
        if isinstance(condition_model.names, dict)
        else dict(enumerate(condition_model.names))
    )
    # Fails closed: refuses to run if this checkpoint isn't EXACTLY the
    # active 4-class taxonomy (e.g. the older 5-class research checkpoint).
    assert_active_taxonomy(cond_names)

    return panel_model, condition_model


def load_condition_classifier(backend: str | None = None) -> tuple[Any, Any]:
    """Load the panel-crop condition CLASSIFIER (MobileNetV2, Phase 16-B) —
    a SEPARATE research checkpoint from the box-detector condition model
    above, evaluated on the same held-out test set at the image level:
    0.752 accuracy vs. the box-detector's 0.727 image-level agreement (real,
    modest win overall), and specifically 0.794 vs. 0.359 recall on DUSTY —
    see docs/CONDITION_MODEL_4CLASS_CLASSIFIER_V1.md. Fails closed like
    every other model loader in this project."""
    if active_classifier_backend(backend) == "web_ensemble":
        if not ensemble_available():
            raise ModelLoadError(
                "condition classifier",
                f"web ensemble manifest not found at {ENSEMBLE_MANIFEST}. Fetch the Kaggle outputs "
                "(see datasets/ground_pv_research/web_condition_v2/README.md) or use the default backend.",
            )
        from training.detection.ground_pv.web_ensemble import WebEnsemble  # noqa: PLC0415

        try:
            return WebEnsemble(ENSEMBLE_MANIFEST), None
        except Exception as exc:  # noqa: BLE001 — surfaced as a typed, actionable error
            raise ModelLoadError("condition classifier",
                                 f"web ensemble failed to load: {type(exc).__name__}: {exc}") from exc

    import torch
    from torchvision import models

    if not CLASSIFIER_WEIGHTS.is_file():
        raise ModelLoadError(
            "condition classifier",
            f"weights not found at {CLASSIFIER_WEIGHTS}. Train via "
            "scripts/train_condition_4class_classifier.py or the Kaggle notebook first.",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASSIFIER_CLASSES))
    try:
        state_dict = torch.load(str(CLASSIFIER_WEIGHTS), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as exc:  # noqa: BLE001 — surfaced as a typed, actionable error
        raise ModelLoadError(
            "condition classifier",
            f"weights exist but failed to load as a {len(CLASSIFIER_CLASSES)}-class "
            f"MobileNetV2: {type(exc).__name__}: {exc}",
        ) from exc
    model.to(device)
    model.eval()
    return model, device


def classify_crop(classifier_model: Any, device: Any, crop_image: Image.Image) -> dict[str, Any]:
    """Run the condition classifier on ONE panel crop. Unlike the box
    detector, a classifier always outputs a full probability distribution
    (never zero detections) — so "no_prediction" cannot occur here, but a
    low-confidence top class still gets marked "uncertain", never silently
    treated as a confident read. Preprocessing (resize 224 + center-crop +
    ImageNet normalize) matches training exactly."""
    if hasattr(classifier_model, "predict_proba"):  # web ensemble: own preprocessing + own (canonical) class order
        probs = classifier_model.predict_proba(crop_image)
        classes = classifier_model.classes
        class_id = max(range(len(probs)), key=lambda i: probs[i])
        confidence = probs[class_id]
        return {
            "status": "predicted" if confidence >= CLASSIFIER_PREDICTED_THRESHOLD else "uncertain",
            "label": classes[class_id], "confidence": confidence, "probabilities": dict(zip(classes, probs)),
        }

    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(CLASSIFIER_INPUT_SIZE),
        transforms.CenterCrop(CLASSIFIER_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(crop_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(classifier_model(tensor), dim=1).squeeze(0).cpu().tolist()
    class_id = max(range(len(probs)), key=lambda i: probs[i])
    label = CLASSIFIER_CLASSES[class_id]
    confidence = probs[class_id]
    status = "predicted" if confidence >= CLASSIFIER_PREDICTED_THRESHOLD else "uncertain"
    return {
        "status": status, "label": label, "confidence": confidence,
        "probabilities": dict(zip(CLASSIFIER_CLASSES, probs)),
    }


def run_capture(
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
        panel_model_name=PANEL_WEIGHTS.name,
        condition_model_name=CONDITION_WEIGHTS.name,
        panel_config=DynamicPanelConfig(confidence_threshold=detection_conf),
        condition_config=ConditionInferenceConfig(),
    )
    for p in report.get("panels", []):
        crop_box = tuple(p["crop"]["bbox"])
        crop = image.crop(crop_box)
        p["classifier_condition"] = classify_crop(classifier_model, classifier_device, crop)
    return report


def display_id(capture_idx: int, panel_id: str, multi: bool) -> str:
    """Panel ids are only capture-prefixed when there's more than one capture
    — avoids implying cross-capture identity that this prototype doesn't
    compute, while keeping single-capture display exactly as before."""
    return f"C{capture_idx}-{panel_id}" if multi else panel_id


def summarize_panels(captures: list[dict[str, Any]]) -> dict[str, Any]:
    """Flatten every capture's panels into flagged/unread/confirmed_clean
    lists (CLASSIFIER read = primary condition signal) plus zero-panel
    capture labels and whether any geometry warning fired.

    Shared by the live results display AND history persistence so the two
    can never silently drift apart — the number a user (or admin panel)
    sees is exactly the number written to history."""
    multi = len(captures) > 1
    all_panels: list[dict[str, Any]] = []
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
                "display_id": display_id(idx, p["panel_id"], multi),
                "condition": p["classifier_condition"],
                "detector_condition": p["condition"],
            })

    flagged = [p for p in all_panels if p["condition"]["status"] == "predicted" and p["condition"]["label"] != "CLEAN"]
    unread = [p for p in all_panels if p["condition"]["status"] in ("uncertain", "no_prediction")]
    confirmed_clean = [p for p in all_panels if p["condition"]["status"] == "predicted" and p["condition"]["label"] == "CLEAN"]

    return {
        "multi": multi,
        "all_panels": all_panels,
        "zero_panel_captures": zero_panel_captures,
        "any_geometry_warning": any_geometry_warning,
        "flagged": flagged,
        "unread": unread,
        "confirmed_clean": confirmed_clean,
    }
