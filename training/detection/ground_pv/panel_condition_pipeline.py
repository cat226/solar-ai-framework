"""training/detection/ground_pv/panel_condition_pipeline.py — Phase 15A:
full camera image -> dynamic panel detection -> panel crops -> 4-class
condition inference -> stable per-panel schema.

    FULL CAMERA IMAGE
          -> dynamic panel detector (Phase 14A, unmodified)
          -> P1 .. PN  (N never assumed — see dynamic_panel_detection.py)
          -> crop each accepted panel detection
          -> [optional] crop-quality diagnostics (never reject aggressively)
          -> 4-class condition model on the crop (CLEAN/DUSTY/BIRD_DROP/PHYSICAL_DAMAGE)
          -> per-panel condition result
          -> assembled JSON report

This module NEVER re-answers "is this a panel" — that is the dynamic panel
detector's job (Phase 14A). It only ever runs the condition model on crops
the panel detector already produced. It never trains, fine-tunes, or
modifies any checkpoint.

No-prediction semantics (Phase 15A req 7 — a failed/low-confidence
condition read must NEVER be presented as CLEAN):
    "predicted"     — the condition model's top detection on this crop is
                       at or above `predicted_threshold`.
    "uncertain"      — the model returned a detection, but below
                       `predicted_threshold` (still >= `detection_floor`,
                       the YOLO inference-time conf cutoff, or it would
                       never have been returned at all).
    "no_prediction"  — the model returned ZERO detections on this crop.
                       Never reinterpreted as "the panel is Clean".

Confidence policy (Phase 15A req 8): `detection_floor` (0.25) reuses the
EXISTING documented convention from `scripts/infer_condition_on_panels.py`
(`--condition-conf` default). `predicted_threshold` (0.45) reuses the
project's other EXISTING documented "trustworthy" confidence convention —
the same 0.45 the ground panel detector uses as its real-world-holdout bar
(see docs/DYNAMIC_PANEL_DETECTION.md). Neither value is invented from
nothing; both are pulled from conventions already established elsewhere in
this codebase, for consistency.

Multi-detection support (Phase 15A req 5): the condition model is a
DETECTOR (same architecture family as the existing 5-class research
checkpoint, `weights/yolo_solar_condition_research_v1.pt` — see
[[ground_pv_condition_model_baseline]]), so a single crop can naturally
carry more than one condition detection (e.g. DUSTY and BIRD_DROP boxes on
the same panel). `condition_detections` carries the FULL list; `condition`
carries only the single highest-confidence one, for the stable schema
shown in the module's CLI counterpart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from training.detection.ground_pv.condition_taxonomy import (
    CONDITION_4CLASS_ORDER,
    SCHEMA_VERSION,
    assert_active_taxonomy,
)
from training.detection.ground_pv.dynamic_panel_detection import (
    DynamicPanelConfig,
    detect_dynamic_panels,
)

# ---------------------------------------------------------------------------
# Confidence policy — see module docstring for why these exact values.
# ---------------------------------------------------------------------------
DEFAULT_CONDITION_DETECTION_FLOOR = 0.25   # matches infer_condition_on_panels.py's existing --condition-conf
DEFAULT_CONDITION_PREDICTED_THRESHOLD = 0.45  # reuses the project's existing 0.45 "trustworthy" convention

# ---------------------------------------------------------------------------
# Crop-quality diagnostics — no existing convention for these in the
# codebase, so documented, configurable defaults are defined here (never
# used to aggressively reject a crop — diagnostics only, per req 9).
# ---------------------------------------------------------------------------
DEFAULT_CROP_PAD = 0.04                      # matches infer_condition_on_panels.py's existing --crop-pad
DEFAULT_MIN_CROP_SIDE_PX = 20                 # a crop smaller than this in either dimension is unusable
DEFAULT_MIN_CROP_RELATIVE_AREA = 0.0003       # crop area vs full-image area
DEFAULT_MIN_ASPECT_RATIO = 0.15
DEFAULT_MAX_ASPECT_RATIO = 6.5
DEFAULT_MIN_SURVIVING_CROP_FRACTION = 0.85    # actual crop area / intended padded-crop area
DEFAULT_BLUR_VARIANCE_THRESHOLD = 15.0        # Laplacian-variance floor; below this -> "blurry" warning


@dataclass
class ConditionInferenceConfig:
    """All tunable thresholds in one place — see module docstring for the
    provenance of each default."""

    detection_floor: float = DEFAULT_CONDITION_DETECTION_FLOOR
    predicted_threshold: float = DEFAULT_CONDITION_PREDICTED_THRESHOLD
    iou: float = 0.50
    imgsz: int = 640
    crop_pad: float = DEFAULT_CROP_PAD
    min_crop_side_px: int = DEFAULT_MIN_CROP_SIDE_PX
    min_crop_relative_area: float = DEFAULT_MIN_CROP_RELATIVE_AREA
    min_aspect_ratio: float = DEFAULT_MIN_ASPECT_RATIO
    max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO
    min_surviving_crop_fraction: float = DEFAULT_MIN_SURVIVING_CROP_FRACTION
    blur_variance_threshold: float = DEFAULT_BLUR_VARIANCE_THRESHOLD

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ConditionDetection:
    label: str
    confidence: float
    bbox_in_crop: list[float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox_in_crop": [round(v, 2) for v in self.bbox_in_crop],
        }


# ---------------------------------------------------------------------------
# Crop geometry + quality diagnostics (pure, no model — directly testable)
# ---------------------------------------------------------------------------


def compute_padded_crop_box(
    bbox: list[float], image_width: int, image_height: int, pad: float,
) -> tuple[int, int, int, int]:
    """Panel bbox [x1,y1,x2,y2] -> a padded crop box clipped to the image."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    return (
        max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh)),
        min(image_width, int(x2 + pad * bw)), min(image_height, int(y2 + pad * bh)),
    )


def _laplacian_variance(gray: np.ndarray) -> float | None:
    """Simple, dependency-free (no cv2/scipy) blur metric: variance of a
    discrete Laplacian. Lower variance -> smoother/blurrier image. Returns
    None if the crop is too small to compute a 3x3 neighborhood."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return None
    g = gray.astype(np.float64)
    center = g[1:-1, 1:-1]
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * center
    return float(lap.var())


def check_crop_quality(
    crop_image: Any,
    intended_crop_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    config: ConditionInferenceConfig,
) -> list[str]:
    """Diagnostics on ONE panel crop, never used to reject it (req 9) —
    only to attach warnings the caller/UI can surface."""
    warnings: list[str] = []
    cw, ch = crop_image.size

    if cw < config.min_crop_side_px or ch < config.min_crop_side_px:
        warnings.append("panel_crop_too_small")

    crop_rel_area = (cw * ch) / (image_width * image_height) if image_width * image_height > 0 else 0.0
    if crop_rel_area < config.min_crop_relative_area:
        warnings.append("panel_crop_insufficient_area")

    aspect = cw / ch if ch > 0 else float("inf")
    if aspect < config.min_aspect_ratio or aspect > config.max_aspect_ratio:
        warnings.append("panel_crop_extreme_aspect_ratio")

    ix1, iy1, ix2, iy2 = intended_crop_box
    intended_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    actual_area = cw * ch
    if intended_area > 0 and actual_area / intended_area < config.min_surviving_crop_fraction:
        warnings.append("panel_crop_truncated")

    try:
        gray = np.asarray(crop_image.convert("L"), dtype=np.float64)
        var = _laplacian_variance(gray)
        if var is not None and var < config.blur_variance_threshold:
            warnings.append("panel_crop_possibly_blurry")
    except Exception:  # noqa: BLE001 — blur check is a diagnostic, never fatal
        pass

    return warnings


# ---------------------------------------------------------------------------
# Condition inference on one crop (duck-typed model — testable with a fake)
# ---------------------------------------------------------------------------


def run_condition_on_crop(
    condition_model: Any,
    crop_image: Any,
    config: ConditionInferenceConfig | None = None,
) -> dict[str, Any]:
    """Run the 4-class condition model on ONE panel crop.

    `condition_model` needs only `.predict(source=, conf=, iou=, imgsz=,
    verbose=)` and `.names` (Ultralytics YOLO satisfies this; so does a
    fake model in tests). Fails closed via `assert_active_taxonomy` if the
    injected model is not the active 4-class taxonomy — a 5-class (or
    other) checkpoint is NEVER silently treated as if it were the active
    4-class model.

    Returns a dict with `status` ("predicted"/"uncertain"/"no_prediction"),
    `label`/`confidence` for the single top detection (None/None for
    no_prediction), and `detections` (the full list, for multi-defect
    representation).
    """
    config = config or ConditionInferenceConfig()
    names = condition_model.names if isinstance(getattr(condition_model, "names", None), dict) \
        else dict(enumerate(getattr(condition_model, "names", [])))
    assert_active_taxonomy(names)

    res = condition_model.predict(
        source=crop_image, conf=config.detection_floor, iou=config.iou, imgsz=config.imgsz, verbose=False,
    )[0]

    detections: list[ConditionDetection] = []
    for b in res.boxes:
        cid = int(b.cls.item())
        label = str(names.get(cid, cid))
        confidence = float(b.conf.item())
        bbox = [float(v) for v in b.xyxy[0].tolist()]
        detections.append(ConditionDetection(label=label, confidence=confidence, bbox_in_crop=bbox))

    detections.sort(key=lambda d: -d.confidence)

    if not detections:
        return {"status": "no_prediction", "label": None, "confidence": None, "detections": []}

    top = detections[0]
    status = "predicted" if top.confidence >= config.predicted_threshold else "uncertain"
    return {
        "status": status, "label": top.label, "confidence": top.confidence,
        "detections": detections,
    }


# ---------------------------------------------------------------------------
# Full pipeline: image -> dynamic panels -> crops -> condition -> schema
# ---------------------------------------------------------------------------


def run_panel_condition_pipeline(
    panel_model: Any,
    condition_model: Any,
    image: Any,
    *,
    image_name: str,
    panel_model_name: str,
    condition_model_name: str,
    panel_config: DynamicPanelConfig | None = None,
    condition_config: ConditionInferenceConfig | None = None,
) -> dict[str, Any]:
    """End-to-end Phase 15A pipeline for one full camera image.

    `panel_count` and the `panels` list are ALWAYS derived from however
    many accepted panel detections the dynamic panel detector returns for
    THIS image — never assumed to be 4, 6, 8, or anything else. The number
    of condition inferences always equals the accepted panel count.
    """
    panel_config = panel_config or DynamicPanelConfig()
    condition_config = condition_config or ConditionInferenceConfig()

    panel_report = detect_dynamic_panels(
        panel_model, image, image_name=image_name, model_name=panel_model_name, config=panel_config,
    )

    if panel_report["panel_count"] == 0:
        return {
            "schema_version": SCHEMA_VERSION,
            "image": image_name,
            "image_width": panel_report["image_width"],
            "image_height": panel_report["image_height"],
            "panel_model": panel_model_name,
            "condition_model": condition_model_name,
            "condition_taxonomy": list(CONDITION_4CLASS_ORDER),
            "panel_count": 0,
            "analysis_status": "failed",
            "failure_reason": panel_report.get("failure_reason", "no_panel_detected"),
            "panels": [],
        }

    W, H = panel_report["image_width"], panel_report["image_height"]
    panels_out: list[dict[str, Any]] = []
    any_suspicious_panel = any(p["status"] == "suspicious" for p in panel_report["panels"])

    for p in panel_report["panels"]:
        bbox = [p["bbox"]["x1"], p["bbox"]["y1"], p["bbox"]["x2"], p["bbox"]["y2"]]
        crop_box = compute_padded_crop_box(bbox, W, H, condition_config.crop_pad)
        crop = image.crop(crop_box)

        crop_warnings = check_crop_quality(crop, crop_box, W, H, condition_config)

        cond = run_condition_on_crop(condition_model, crop, condition_config)

        panels_out.append({
            "panel_id": p["panel_id"],
            "detection": {
                "confidence": p["confidence"],
                "bbox": p["bbox"],
                "status": p["status"],  # from the panel-detection layer (valid/suspicious) — separate concern
            },
            "condition": {
                "status": cond["status"],
                "label": cond["label"],
                "confidence": round(cond["confidence"], 4) if cond["confidence"] is not None else None,
            },
            "condition_detections": [d.as_dict() for d in cond["detections"]],
            "crop": {
                "bbox": list(crop_box),
                "warnings": crop_warnings,
            },
        })

    analysis_status = "warning" if any_suspicious_panel else "ok"

    return {
        "schema_version": SCHEMA_VERSION,
        "image": image_name,
        "image_width": W,
        "image_height": H,
        "panel_model": panel_model_name,
        "condition_model": condition_model_name,
        "condition_taxonomy": list(CONDITION_4CLASS_ORDER),
        "panel_count": len(panels_out),
        "analysis_status": analysis_status,
        "panels": panels_out,
    }
