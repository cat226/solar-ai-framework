"""training/detection/ground_pv/panel_condition.py — Associate detected
defects with detected panels and derive a per-panel *research* visual
condition assessment.

This is NOT a validated engineering or safety diagnosis. The wording used
throughout is "research visual condition assessment" / "visual defect
detection".

Input: the raw detections from the research detector for one image, as a
list of dicts:
    {"class_name": "panel"|"bird_drop"|"cracked"|"dusty",
     "confidence": float,
     "bbox": [x1, y1, x2, y2]}   # pixel coords in the original image

Output: one record per detected panel:
    {"panel_id": int,
     "panel_bbox": [x1, y1, x2, y2],
     "panel_confidence": float,
     "defects": [{"type": str, "confidence": float, "bbox": [...], "assoc": str}],
     "condition": ["Clean"] | ["Dust", "BirdDrop", ...]}

Association rule (documented, deterministic):
  A defect box is attached to a panel when EITHER
    (a) the defect box centre lies inside the panel box, OR
    (b) IoU(defect, panel) >= `min_iou` (default 0.10) — covers large
        soiling regions that overlap a panel heavily without their centre
        happening to fall inside it.
  A defect matching multiple panels is attached to the one with the
  greatest overlap area (ties -> highest panel confidence -> lowest id).
  A defect matching no panel is reported separately as `unassociated_defects`
  and never silently dropped.

No third-party dependency (pure python) so it is unit-testable directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from training.detection.ground_pv.class_map import DEFECT_TO_CONDITION

NO_DEFECT_CONDITION = "Clean"
DEFAULT_MIN_IOU = 0.10


def _area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = _intersection_area(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _centre_inside(inner: Sequence[float], outer: Sequence[float]) -> bool:
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


@dataclass
class PanelAssessment:
    panel_id: int
    panel_bbox: list[float]
    panel_confidence: float
    defects: list[dict[str, Any]] = field(default_factory=list)
    condition: list[str] = field(default_factory=lambda: [NO_DEFECT_CONDITION])

    def as_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "panel_bbox": self.panel_bbox,
            "panel_confidence": self.panel_confidence,
            "defects": self.defects,
            "condition": self.condition,
        }


def assess_image(
    detections: Sequence[dict[str, Any]],
    *,
    min_iou: float = DEFAULT_MIN_IOU,
) -> dict[str, Any]:
    """Group `detections` into per-panel research condition assessments."""
    panels = [d for d in detections if d.get("class_name") == "panel"]
    defects = [d for d in detections if d.get("class_name") in DEFECT_TO_CONDITION]

    # Sort panels for deterministic ids: by (-confidence, x1, y1).
    panels_sorted = sorted(
        panels, key=lambda d: (-float(d.get("confidence", 0.0)), d["bbox"][0], d["bbox"][1])
    )
    assessments = [
        PanelAssessment(
            panel_id=i + 1,
            panel_bbox=[float(v) for v in p["bbox"]],
            panel_confidence=float(p.get("confidence", 0.0)),
        )
        for i, p in enumerate(panels_sorted)
    ]

    unassociated: list[dict[str, Any]] = []
    for d in defects:
        dbox = [float(v) for v in d["bbox"]]
        best_idx = -1
        best_overlap = 0.0
        best_assoc = ""
        for idx, pa in enumerate(assessments):
            pbox = pa.panel_bbox
            inside = _centre_inside(dbox, pbox)
            iou_val = _iou(dbox, pbox)
            if not inside and iou_val < min_iou:
                continue
            overlap = _intersection_area(dbox, pbox)
            key = (overlap, pa.panel_confidence, -pa.panel_id)
            best_key = (best_overlap, assessments[best_idx].panel_confidence if best_idx >= 0 else -1,
                        -assessments[best_idx].panel_id if best_idx >= 0 else 0)
            if best_idx < 0 or key > best_key:
                best_idx = idx
                best_overlap = overlap
                best_assoc = "centre_inside" if inside else "iou"
        entry = {
            "type": d["class_name"],
            "confidence": float(d.get("confidence", 0.0)),
            "bbox": dbox,
            "assoc": best_assoc or "none",
        }
        if best_idx < 0:
            unassociated.append(entry)
        else:
            assessments[best_idx].defects.append(entry)

    for pa in assessments:
        if pa.defects:
            conds: list[str] = []
            for dfx in pa.defects:
                cond = DEFECT_TO_CONDITION[dfx["type"]]
                if cond not in conds:
                    conds.append(cond)
            pa.condition = conds
        else:
            pa.condition = [NO_DEFECT_CONDITION]

    return {
        "num_panels": len(assessments),
        "num_defect_detections": len(defects),
        "panels": [pa.as_dict() for pa in assessments],
        "unassociated_defects": unassociated,
        "assessment_type": "research visual condition assessment (not a validated diagnosis)",
        "association_rule": {
            "centre_inside_panel": True,
            "min_iou": min_iou,
        },
    }
