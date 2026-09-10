"""training/detection/ground_pv/dynamic_panel_detection.py — dynamic
multi-panel detection + deterministic spatial ordering (Phase 14A).

Deployment context changed: the camera is no longer assumed to see exactly
4 panels. A single capture may show 4, 6, 8, or any other number of visible
panels, in a single row, single column, or grid. This module implements the
architecture for that:

    FULL CAMERA IMAGE
          -> PANEL DETECTOR (existing ground YOLO checkpoint, unmodified)
          -> N raw "panel"-class detections (N is NOT assumed; never hard-coded)
          -> geometry-based spatial ordering -> P1 .. PN
          -> geometry / overlap sanity diagnostics (valid vs suspicious)
          -> JSON report + optional annotated image

**The system does not assume a fixed number of panels.** Nothing here reads
"if four_panels" or constructs a fixed P1..P4 schema; `panel_count` and the
`panels` list are always derived from however many accepted detections exist
for THIS image, including 0.

**Panel IDs (`P1..PN`) are spatial identities for a single capture only.**
They are assigned by sorting detections geometrically (top-to-bottom,
left-to-right) within THIS image, so the same physical arrangement produces
the same P1..PN regardless of the order the detector happened to emit boxes
in. They are NOT yet guaranteed to represent persistent physical panel
identities across different captures over time (e.g. "P3 today" is not
guaranteed to be the same physical panel as "P3 tomorrow" if the camera
moves, the array changes, or a panel drops out of frame). That temporal
identity problem is out of scope for this phase — see
docs/DYNAMIC_PANEL_DETECTION.md.

This module does NOT perform condition classification (Clean / Dusty /
BirdDrop / PhysicalDamage / ElectricalDamage). It is panel localization and
panel identity/order only. See
training/detection/ground_pv/panel_condition.py for the (separate) defect/
condition association logic that operates once panels exist.

No model is trained, retrained, or modified here. This module only
orchestrates inference against whatever checkpoint the caller passes in
(currently `weights/yolo_solar_ground_research_v1.pt`, unmodified — see
[[ground_pv_research_detector]] / [[ground_pv_detector_v2_experiment]]).

Only pure-python geometry (no third-party dependency in this half of the
module) so it is directly unit-testable without a GPU or Ultralytics
installed — mirrors the existing convention in
training/detection/ground_pv/tiled_inference.py and panel_condition.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

SCHEMA_VERSION = "1.0"

# Confidence: 0.45 matches the existing documented real-world-holdout
# convention for this checkpoint (see docs/GROUND_PV_RESEARCH.md sec.19,
# scripts/infer_ground_pv_tiled.py's --panel-conf default) — NOT silently
# lowered to inflate panel count.
_DEFAULT_CONF = 0.45
_DEFAULT_IOU = 0.50  # YOLO's own NMS threshold (existing convention)
_DEFAULT_IMGSZ = 640

# Row-grouping tolerance: fraction of image HEIGHT within which two panel
# centers are considered "the same row". Perspective-tolerant (does not
# require identical y). Configurable — see DynamicPanelConfig.
_DEFAULT_ROW_GROUPING_TOLERANCE = 0.10

# Post-NMS residual-overlap diagnostic. YOLO's own NMS (iou=0.50 by default)
# already removes near-duplicate boxes; this is a SEPARATE, lower-threshold
# sanity layer over what NMS leaves behind, exposed as diagnostics only
# (never silently deletes a detection).
_DEFAULT_OVERLAP_WARNING_IOU = 0.30
_DEFAULT_CONTAINMENT_WARNING_FRACTION = 0.80

# Geometric sanity thresholds (relative to the full image, or to the rest of
# the detected array). Diagnostics-first: these flag "suspicious", they do
# NOT reject/delete a detection.
_DEFAULT_MIN_RELATIVE_AREA = 0.001   # panel smaller than 0.1% of the frame
_DEFAULT_MAX_RELATIVE_AREA = 0.60    # panel bigger than 60% of the frame
_DEFAULT_MIN_ASPECT_RATIO = 0.20
_DEFAULT_MAX_ASPECT_RATIO = 5.0
_DEFAULT_SIZE_INCONSISTENCY_RATIO = 3.0        # vs median panel area in this image
_DEFAULT_ISOLATION_DISTANCE_MULTIPLIER = 3.0   # vs median nearest-neighbor distance

_PANEL_CLASS_NAME = "panel"


@dataclass
class DynamicPanelConfig:
    """All tunable thresholds for dynamic panel detection, in one place —
    nothing below is a scattered magic number."""

    confidence_threshold: float = _DEFAULT_CONF
    iou_threshold: float = _DEFAULT_IOU
    imgsz: int = _DEFAULT_IMGSZ
    row_grouping_tolerance: float = _DEFAULT_ROW_GROUPING_TOLERANCE
    overlap_warning_iou: float = _DEFAULT_OVERLAP_WARNING_IOU
    containment_warning_fraction: float = _DEFAULT_CONTAINMENT_WARNING_FRACTION
    min_relative_area: float = _DEFAULT_MIN_RELATIVE_AREA
    max_relative_area: float = _DEFAULT_MAX_RELATIVE_AREA
    min_aspect_ratio: float = _DEFAULT_MIN_ASPECT_RATIO
    max_aspect_ratio: float = _DEFAULT_MAX_ASPECT_RATIO
    size_inconsistency_ratio: float = _DEFAULT_SIZE_INCONSISTENCY_RATIO
    isolation_distance_multiplier: float = _DEFAULT_ISOLATION_DISTANCE_MULTIPLIER
    panel_class_name: str = _PANEL_CLASS_NAME

    def as_dict(self) -> dict[str, Any]:
        return {
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "imgsz": self.imgsz,
            "row_grouping_tolerance": self.row_grouping_tolerance,
            "overlap_warning_iou": self.overlap_warning_iou,
            "containment_warning_fraction": self.containment_warning_fraction,
            "min_relative_area": self.min_relative_area,
            "max_relative_area": self.max_relative_area,
            "min_aspect_ratio": self.min_aspect_ratio,
            "max_aspect_ratio": self.max_aspect_ratio,
            "size_inconsistency_ratio": self.size_inconsistency_ratio,
            "isolation_distance_multiplier": self.isolation_distance_multiplier,
            "panel_class_name": self.panel_class_name,
        }


@dataclass
class RawPanelDetection:
    """One accepted (panel-class, above-threshold) raw detection, in
    original-image pixel coordinates. Derived geometry fields are computed
    properties, never independently settable, so they can't drift out of
    sync with the box."""

    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return self.x1 + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y1 + self.height / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def bbox(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def as_raw_dict(self) -> dict[str, Any]:
        """Minimum raw-detection fields required by req A: class, confidence,
        x1/y1/x2/y2, width, height, center_x, center_y, area."""
        return {
            "class": self.class_name,
            "confidence": round(self.confidence, 4),
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "center_x": round(self.center_x, 2),
            "center_y": round(self.center_y, 2),
            "area": round(self.area, 2),
        }


# ---------------------------------------------------------------------------
# Detection (model orchestration — thin wrapper, duck-typed like
# tiled_inference.tiled_detect: `model` needs only .predict(...) and .names)
# ---------------------------------------------------------------------------


def run_panel_detector(
    model: Any,
    image: Any,
    *,
    conf: float = _DEFAULT_CONF,
    iou: float = _DEFAULT_IOU,
    imgsz: int = _DEFAULT_IMGSZ,
    panel_class_name: str = _PANEL_CLASS_NAME,
) -> list[RawPanelDetection]:
    """Run the ground panel detector on one full image and return ONLY
    accepted panel-class detections above `conf`.

    Any other class the checkpoint may emit (e.g. `cracked`, `dusty`) is
    explicitly filtered out here — never silently included, never used to
    inflate `panel_count`.
    """
    names = model.names if isinstance(getattr(model, "names", None), dict) else dict(enumerate(getattr(model, "names", [])))
    res = model.predict(source=image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]

    accepted: list[RawPanelDetection] = []
    for b in res.boxes:
        cls_id = int(b.cls.item())
        name = str(names.get(cls_id, cls_id))
        if name.strip().lower() != panel_class_name.strip().lower():
            continue  # explicit class filter — only PANEL is accepted (req A)
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        confidence = float(b.conf.item())
        accepted.append(RawPanelDetection(class_name=name, confidence=confidence, x1=x1, y1=y1, x2=x2, y2=y2))
    return accepted


# ---------------------------------------------------------------------------
# Pure geometry helpers (no model, no I/O — directly unit-testable)
# ---------------------------------------------------------------------------


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _contained_fraction(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Fraction of `inner`'s area that lies within `outer`."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_inner = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / area_inner if area_inner > 0 else 0.0


@dataclass
class _OrderedItem:
    detection: RawPanelDetection
    row_index: int
    column_index: int


def order_panels_spatially(
    detections: Sequence[RawPanelDetection],
    image_width: int,
    image_height: int,
    *,
    row_grouping_tolerance: float = _DEFAULT_ROW_GROUPING_TOLERANCE,
) -> tuple[list[_OrderedItem], int, bool]:
    """Assign deterministic P1..PN spatial order.

    Algorithm (matches the recommended approach exactly):
      1. Compute + normalize panel centers relative to image dimensions.
      2. Sort ALL detections by normalized center_y ascending FIRST — this
         single sort is what makes the final order independent of whatever
         order the detector originally emitted boxes in (see
         test_ground_pv_dynamic_panel_detection.py::test_order_is_independent_of_input_order).
      3. Greedily cluster into rows: a detection joins the current row if its
         normalized center_y is within `row_grouping_tolerance` of that row's
         running mean center_y (tolerant of perspective skew — does not
         require identical y); otherwise it starts a new row.
      4. Sort rows top-to-bottom (already true from step 2/3's construction).
      5. Sort panels within each row left-to-right by normalized center_x.
      6. Assign P1..PN sequentially, row-major.

    Works for a single row (all one cluster), a single column (N clusters of
    1), or any grid — none of that is hard-coded, it falls out of the
    clustering.

    Returns (ordered items, row_count, rows_overlap) where `rows_overlap` is
    True when the geometry looks ambiguous (two adjacent row clusters'
    vertical extents overlap, meaning "row" was not a clean concept for this
    layout) — the caller uses this to mark ordering as suspicious rather than
    pretending certainty.
    """
    if not detections or image_width <= 0 or image_height <= 0:
        return [], 0, False

    entries = sorted(
        (
            {
                "det": d,
                "cy": d.center_y / image_height,
                "cx": d.center_x / image_width,
            }
            for d in detections
        ),
        key=lambda e: e["cy"],
    )

    rows: list[list[dict]] = []
    for e in entries:
        if rows:
            row_mean_cy = sum(r["cy"] for r in rows[-1]) / len(rows[-1])
            if abs(e["cy"] - row_mean_cy) <= row_grouping_tolerance:
                rows[-1].append(e)
                continue
        rows.append([e])

    # Ambiguity check: do consecutive rows' vertical pixel extents overlap?
    # If a "row" isn't cleanly separated from the next, top-to-bottom order
    # is not a safe claim for this geometry.
    rows_overlap = False
    if len(rows) >= 2:
        row_y_ranges: list[tuple[float, float]] = []
        for row in rows:
            ys_top = [r["det"].y1 for r in row]
            ys_bottom = [r["det"].y2 for r in row]
            row_y_ranges.append((min(ys_top), max(ys_bottom)))
        for i in range(len(row_y_ranges) - 1):
            if row_y_ranges[i][1] > row_y_ranges[i + 1][0]:
                rows_overlap = True
                break

    ordered: list[_OrderedItem] = []
    for row_idx, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda e: e["cx"])
        for col_idx, e in enumerate(row_sorted):
            ordered.append(_OrderedItem(detection=e["det"], row_index=row_idx, column_index=col_idx))

    return ordered, len(rows), rows_overlap


# ---------------------------------------------------------------------------
# Diagnostics (post-ordering, so warnings can reference final P-ids)
# ---------------------------------------------------------------------------


def _check_overlaps(
    ordered: Sequence[_OrderedItem],
    *,
    overlap_warning_iou: float,
    containment_warning_fraction: float,
) -> dict[str, list[str]]:
    """Flag pairs whose boxes overlap excessively AFTER the model's own NMS
    has already run. Never deletes a detection — diagnostic only (req D)."""
    warnings_by_id: dict[str, list[str]] = {}
    n = len(ordered)
    for i in range(n):
        for j in range(i + 1, n):
            bi = ordered[i].detection.bbox
            bj = ordered[j].detection.bbox
            iou_val = _iou(bi, bj)
            contain_ij = _contained_fraction(bi, bj)
            contain_ji = _contained_fraction(bj, bi)
            if iou_val >= overlap_warning_iou or contain_ij >= containment_warning_fraction or contain_ji >= containment_warning_fraction:
                pid_i, pid_j = f"P{i + 1}", f"P{j + 1}"
                warnings_by_id.setdefault(pid_i, []).append(f"high_overlap_with_{pid_j}")
                warnings_by_id.setdefault(pid_j, []).append(f"high_overlap_with_{pid_i}")
    return warnings_by_id


def _check_size_and_aspect(
    ordered: Sequence[_OrderedItem],
    image_width: int,
    image_height: int,
    config: DynamicPanelConfig,
) -> dict[str, list[str]]:
    """Per-panel tiny / large / extreme-aspect-ratio diagnostics (req E.1-3)."""
    warnings_by_id: dict[str, list[str]] = {}
    img_area = image_width * image_height
    for idx, item in enumerate(ordered):
        pid = f"P{idx + 1}"
        d = item.detection
        rel_area = d.area / img_area if img_area > 0 else 0.0
        aspect = d.width / d.height if d.height > 0 else float("inf")
        w: list[str] = []
        if rel_area < config.min_relative_area:
            w.append("extremely_small_area")
        if rel_area > config.max_relative_area:
            w.append("extremely_large_area")
        if aspect < config.min_aspect_ratio or aspect > config.max_aspect_ratio:
            w.append("extreme_aspect_ratio")
        if w:
            warnings_by_id[pid] = w
    return warnings_by_id


def _check_array_consistency(
    ordered: Sequence[_OrderedItem],
    image_width: int,
    image_height: int,
    config: DynamicPanelConfig,
) -> dict[str, list[str]]:
    """Size-inconsistency and isolation diagnostics relative to the rest of
    the detected array in THIS image (req E.5-6). Needs >= 2 panels."""
    warnings_by_id: dict[str, list[str]] = {}
    n = len(ordered)
    if n < 2:
        return warnings_by_id

    areas = sorted(item.detection.area for item in ordered)
    mid = n // 2
    median_area = areas[mid] if n % 2 else (areas[mid - 1] + areas[mid]) / 2.0

    centers = [
        (item.detection.center_x / image_width, item.detection.center_y / image_height)
        for item in ordered
    ]
    nn_dists: list[float] = []
    for i in range(n):
        best = min(
            math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            for j in range(n)
            if j != i
        )
        nn_dists.append(best)
    sorted_nn = sorted(nn_dists)
    median_nn = sorted_nn[mid] if n % 2 else (sorted_nn[mid - 1] + sorted_nn[mid]) / 2.0

    for idx, item in enumerate(ordered):
        pid = f"P{idx + 1}"
        w: list[str] = []
        area = item.detection.area
        if median_area > 0 and (
            area > median_area * config.size_inconsistency_ratio
            or area < median_area / config.size_inconsistency_ratio
        ):
            w.append("size_inconsistent_with_array")
        if median_nn > 0 and nn_dists[idx] > median_nn * config.isolation_distance_multiplier:
            w.append("isolated_detection")
        if w:
            warnings_by_id.setdefault(pid, []).extend(w)
    return warnings_by_id


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_dynamic_panel_report(
    raw_detections: Sequence[RawPanelDetection],
    *,
    image_name: str,
    image_width: int,
    image_height: int,
    model_name: str,
    config: DynamicPanelConfig | None = None,
) -> dict[str, Any]:
    """Turn accepted raw detections into the full JSON-serializable report
    (schema in the module docstring / docs/DYNAMIC_PANEL_DETECTION.md).

    `panel_count` and `panels` are always derived from `len(raw_detections)`
    at call time — never a fixed number, never assumed to be 4.
    """
    config = config or DynamicPanelConfig()

    if not raw_detections:
        # req F — zero detections is reported as a failure to detect, NEVER
        # reinterpreted as "no solar panels exist".
        return {
            "schema_version": SCHEMA_VERSION,
            "image": image_name,
            "image_width": image_width,
            "image_height": image_height,
            "model": model_name,
            "confidence_threshold": config.confidence_threshold,
            "panel_count": 0,
            "analysis_status": "failed",
            "failure_reason": "no_panel_detected",
            "panels": [],
            "geometry": {"row_count": 0, "ordering_status": "n/a"},
        }

    ordered, row_count, rows_overlap = order_panels_spatially(
        raw_detections, image_width, image_height,
        row_grouping_tolerance=config.row_grouping_tolerance,
    )
    ordering_status = "ambiguous" if rows_overlap else "stable"

    overlap_warnings = _check_overlaps(
        ordered, overlap_warning_iou=config.overlap_warning_iou,
        containment_warning_fraction=config.containment_warning_fraction,
    )
    size_warnings = _check_size_and_aspect(ordered, image_width, image_height, config)
    consistency_warnings = _check_array_consistency(ordered, image_width, image_height, config)

    panels: list[dict[str, Any]] = []
    any_suspicious = False
    for idx, item in enumerate(ordered):
        pid = f"P{idx + 1}"
        d = item.detection
        warnings: list[str] = []
        warnings.extend(overlap_warnings.get(pid, []))
        warnings.extend(size_warnings.get(pid, []))
        warnings.extend(consistency_warnings.get(pid, []))
        if ordering_status == "ambiguous":
            warnings.append("suspicious_row_geometry")  # req E.7

        status = "suspicious" if warnings else "valid"
        if status == "suspicious":
            any_suspicious = True

        panels.append({
            "panel_id": pid,
            "confidence": round(d.confidence, 4),
            "bbox": {
                "x1": round(d.x1, 2), "y1": round(d.y1, 2),
                "x2": round(d.x2, 2), "y2": round(d.y2, 2),
            },
            "center": {"x": round(d.center_x, 2), "y": round(d.center_y, 2)},
            "width": round(d.width, 2),
            "height": round(d.height, 2),
            "area": round(d.area, 2),
            "row_index": item.row_index,
            "column_index": item.column_index,
            "status": status,
            "warnings": warnings,
        })

    analysis_status = "warning" if any_suspicious else "ok"

    return {
        "schema_version": SCHEMA_VERSION,
        "image": image_name,
        "image_width": image_width,
        "image_height": image_height,
        "model": model_name,
        "confidence_threshold": config.confidence_threshold,
        "panel_count": len(panels),
        "analysis_status": analysis_status,
        "panels": panels,
        "geometry": {"row_count": row_count, "ordering_status": ordering_status},
    }


def detect_dynamic_panels(
    model: Any,
    image: Any,
    *,
    image_name: str,
    model_name: str,
    config: DynamicPanelConfig | None = None,
) -> dict[str, Any]:
    """End-to-end: full camera image -> panel detector -> ordered, diagnosed
    JSON report. `model` only needs `.predict(...)` + `.names` (Ultralytics
    YOLO satisfies this; so does the fake model used in tests).
    """
    config = config or DynamicPanelConfig()
    width, height = image.size
    raw = run_panel_detector(
        model, image,
        conf=config.confidence_threshold, iou=config.iou_threshold, imgsz=config.imgsz,
        panel_class_name=config.panel_class_name,
    )
    return build_dynamic_panel_report(
        raw, image_name=image_name, image_width=width, image_height=height,
        model_name=model_name, config=config,
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

_COLOR_VALID = (56, 176, 255)
_COLOR_SUSPICIOUS = (255, 92, 92)


def draw_annotated(image: Any, report: dict[str, Any]) -> Any:
    """Draw bounding box + panel id + confidence per panel, plus a
    "Panels detected: N" header. No condition labels (req H)."""
    from PIL import ImageDraw

    im = image.convert("RGB").copy()
    draw = ImageDraw.Draw(im)

    for p in report["panels"]:
        b = p["bbox"]
        color = _COLOR_VALID if p["status"] == "valid" else _COLOR_SUSPICIOUS
        draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=color, width=3)
        draw.text((b["x1"] + 2, max(0, b["y1"] - 12)), f"{p['panel_id']} {p['confidence']:.2f}", fill=color)

    draw.rectangle([0, 0, im.width - 1, 16], fill=(0, 0, 0))
    draw.text((3, 3), f"Panels detected: {report['panel_count']}", fill=(255, 255, 255))
    return im
