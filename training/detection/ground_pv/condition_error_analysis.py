"""training/detection/ground_pv/condition_error_analysis.py — Phase 15C
deterministic error-analysis primitives for the active 4-class RGB
condition model.

This module contains ONLY pure, dependency-light logic (no torch, no
ultralytics, no disk I/O beyond parsing already-loaded text) so it can be
unit tested without a GPU, a checkpoint, or the dataset present. The
orchestration that actually runs inference and writes files lives in
`scripts/analyze_condition_4class_errors.py`.

Core primitive: match model predictions to ground-truth boxes with a
greedy, confidence-ordered, IoU-threshold matcher (the same algorithm
family every YOLO-style detection evaluator uses), then classify the
outcome as TP / FP / FN per (image, class). Everything else in this
module (size binning, FP-cause categorization, FN nearby-signal lookup,
confidence summaries) is built on top of that matching output.

Purpose (Phase 15C): explain WHERE and WHY `weights/yolo_solar_condition_4class_research_v3.pt`
fails, not to improve it. Every category this module assigns is derived
from explicit, inspectable geometry (box overlap, position, size) — never
from a claimed visual judgement a human has not actually made. See
`docs/CONDITION_MODEL_4CLASS_ERROR_ANALYSIS.md` for the human-inspected
supplement to these automated categories.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2), pixel coords


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def iou_xyxy(a: Box, b: Box) -> float:
    """Standard axis-aligned intersection-over-union for two (x1,y1,x2,y2)
    boxes. Returns 0.0 for degenerate/non-overlapping boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def box_area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_area_fraction(b: Box, img_w: int, img_h: int) -> float:
    """Box area as a fraction of total image area — resolution-independent
    size measure, used for the tiny/small/medium/large bins."""
    img_area = float(img_w * img_h)
    if img_area <= 0:
        return 0.0
    return box_area(b) / img_area


def touches_border(b: Box, img_w: int, img_h: int, margin_px: float = 2.0) -> bool:
    """True if the box's edge is within `margin_px` of the image boundary
    on any side — a geometrically objective proxy for "boundary-truncated
    object", never a claim about what caused the truncation."""
    x1, y1, x2, y2 = b
    return (
        x1 <= margin_px
        or y1 <= margin_px
        or x2 >= (img_w - margin_px)
        or y2 >= (img_h - margin_px)
    )


# ---------------------------------------------------------------------------
# YOLO label parsing
# ---------------------------------------------------------------------------

def parse_yolo_label_text(text: str, img_w: int, img_h: int) -> list[tuple[int, Box]]:
    """Parses YOLO-format `class cx cy w h` (normalized 0-1) lines into
    `(class_id, (x1, y1, x2, y2))` pixel-coordinate boxes. Skips blank
    lines and lines that are not exactly 5 whitespace-separated fields —
    never raises on a malformed line, since this is read-only analysis of
    an already-built, already-validated dataset (see
    `build_condition_4class_dataset.py`), not a place to silently accept
    or repair bad data."""
    out: list[tuple[int, Box]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            continue
        x1 = (cx - w / 2.0) * img_w
        y1 = (cy - h / 2.0) * img_h
        x2 = (cx + w / 2.0) * img_w
        y2 = (cy + h / 2.0) * img_h
        out.append((cls_id, (x1, y1, x2, y2)))
    return out


# ---------------------------------------------------------------------------
# Matching — the core primitive
# ---------------------------------------------------------------------------

def match_class_instances(
    gt_boxes: Sequence[Box],
    preds: Sequence[tuple[Box, float]],
    iou_threshold: float = 0.5,
) -> tuple[list[int], list[int], list[int]]:
    """Greedy, confidence-first matcher for ONE (image, class) pair.

    `preds` is a list of `(box, confidence)`, already restricted to this
    class and image. Predictions are processed highest-confidence first;
    each is matched to the highest-IoU still-unmatched GT box of this
    same class/image if that IoU is >= `iou_threshold`, otherwise it is a
    false positive. Any GT box left unmatched afterward is a false
    negative. This is the same greedy strategy COCO/Ultralytics-style
    evaluators use for a single IoU threshold.

    Returns `(tp_pred_indices, fp_pred_indices, fn_gt_indices)`, each a
    list of indices into the ORIGINAL `preds`/`gt_boxes` sequences (not
    re-sorted), so callers can look up the original records.
    """
    order = sorted(range(len(preds)), key=lambda i: preds[i][1], reverse=True)
    matched_gt: set[int] = set()
    tp_idx: list[int] = []
    fp_idx: list[int] = []

    for pi in order:
        pbox, _conf = preds[pi]
        best_gi, best_iou = -1, 0.0
        for gi, gbox in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = iou_xyxy(pbox, gbox)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_gi)
            tp_idx.append(pi)
        else:
            fp_idx.append(pi)

    fn_idx = [gi for gi in range(len(gt_boxes)) if gi not in matched_gt]
    return tp_idx, fp_idx, fn_idx


# ---------------------------------------------------------------------------
# Size binning (data-driven quartiles — never hand-picked to fit a story)
# ---------------------------------------------------------------------------

def quartile_cuts(values: Sequence[float]) -> tuple[float, float, float]:
    """Returns (Q1, Q2, Q3) of `values`. Requires at least 4 values so the
    quartiles are meaningful; raises ValueError otherwise rather than
    silently returning degenerate bins."""
    if len(values) < 4:
        raise ValueError(f"need at least 4 values for quartile bins, got {len(values)}")
    qs = statistics.quantiles(values, n=4, method="inclusive")
    return qs[0], qs[1], qs[2]


def size_bin_label(value: float, cuts: tuple[float, float, float]) -> str:
    """Bins `value` into tiny/small/medium/large using the (Q1,Q2,Q3)
    cut points from `quartile_cuts` — i.e. each bin holds ~25% of the
    reference population by construction, not an arbitrarily chosen
    threshold."""
    q1, q2, q3 = cuts
    if value <= q1:
        return "tiny"
    if value <= q2:
        return "small"
    if value <= q3:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# False-positive cause categorization — geometry-only, deterministic
# ---------------------------------------------------------------------------

FP_CATEGORIES = (
    "duplicate/overlapping prediction",
    "other defect mistaken for this class",
    "localization (low-IoU overlap with same-class GT)",
    "edge/crop artifact",
    "tiny/low-information region",
    "background / non-panel surface",
    "ambiguous visual appearance",
)


def categorize_false_positive(
    fp_box: Box,
    same_image_other_accepted_boxes: Sequence[Box],
    same_image_gt_same_class: Sequence[Box],
    same_image_gt_other_class: Sequence[Box],
    img_w: int,
    img_h: int,
    *,
    tiny_area_fraction: float,
    duplicate_iou: float = 0.3,
    cross_class_iou: float = 0.5,
    border_margin_px: float = 2.0,
) -> str:
    """Assigns exactly one category to a false-positive detection, using
    only objectively computable geometry — never a fabricated visual
    judgement. Precedence (checked in this fixed order, first match
    wins), documented explicitly so results are reproducible:

    1. duplicate/overlapping prediction — heavily overlaps (IoU >=
       `duplicate_iou`) another already-accepted prediction of the SAME
       class in the same image (a near-miss of a real instance already
       counted, or an NMS leftover).
    2. other defect mistaken for this class — overlaps (IoU >=
       `cross_class_iou`) a GT box of a DIFFERENT active class.
    3. localization (low-IoU overlap with same-class GT) — has SOME
       positive overlap with a same-class GT box, but below the matching
       IoU threshold (a real instance was there, the box was just
       wrong/loose, not a hallucination).
    4. edge/crop artifact — box touches the image boundary.
    5. tiny/low-information region — box area fraction is below
       `tiny_area_fraction` (a data-driven threshold the caller supplies,
       e.g. the same-class GT Q1).
    6. background / non-panel surface — zero overlap with any GT box of
       any class.
    7. ambiguous visual appearance — none of the above apply; this is the
       explicit "geometry alone cannot explain this" bucket, not a
       default guess.
    """
    for other in same_image_other_accepted_boxes:
        if iou_xyxy(fp_box, other) >= duplicate_iou:
            return "duplicate/overlapping prediction"

    for gt in same_image_gt_other_class:
        if iou_xyxy(fp_box, gt) >= cross_class_iou:
            return "other defect mistaken for this class"

    max_same_class_iou = max((iou_xyxy(fp_box, gt) for gt in same_image_gt_same_class), default=0.0)
    if max_same_class_iou > 0.0:
        return "localization (low-IoU overlap with same-class GT)"

    if touches_border(fp_box, img_w, img_h, margin_px=border_margin_px):
        return "edge/crop artifact"

    if box_area_fraction(fp_box, img_w, img_h) <= tiny_area_fraction:
        return "tiny/low-information region"

    any_gt_overlap = any(iou_xyxy(fp_box, gt) > 0.0 for gt in (*same_image_gt_same_class, *same_image_gt_other_class))
    if not any_gt_overlap:
        return "background / non-panel surface"

    return "ambiguous visual appearance"


# ---------------------------------------------------------------------------
# False-negative "nearby signal" — did the model produce ANY same-class
# detection near this missed instance, even below the matching threshold?
# ---------------------------------------------------------------------------

def nearest_same_class_confidence(
    fn_box: Box,
    low_conf_same_class_preds: Sequence[tuple[Box, float]],
) -> Optional[float]:
    """Among predictions of the SAME class in the same image that have
    ANY positive IoU overlap with `fn_box` (regardless of confidence),
    returns the highest confidence found. Returns None if the model
    produced literally no overlapping prediction of this class at any
    confidence — a genuine total miss, distinct from a low-confidence
    near-miss."""
    overlapping = [conf for box, conf in low_conf_same_class_preds if iou_xyxy(fn_box, box) > 0.0]
    return max(overlapping) if overlapping else None


# ---------------------------------------------------------------------------
# Confidence-distribution summaries
# ---------------------------------------------------------------------------

def summarize_confidences(values: Sequence[float]) -> dict[str, Any]:
    """n/mean/median/p25/p75/min/max summary. Returns n=0 and None for
    every stat if `values` is empty — never fabricates a distribution
    from nothing."""
    vals = list(values)
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    vals_sorted = sorted(vals)
    quantiles = statistics.quantiles(vals_sorted, n=4, method="inclusive") if len(vals_sorted) >= 4 else None
    return {
        "n": len(vals_sorted),
        "mean": statistics.fmean(vals_sorted),
        "median": statistics.median(vals_sorted),
        "p25": quantiles[0] if quantiles else None,
        "p75": quantiles[2] if quantiles else None,
        "min": vals_sorted[0],
        "max": vals_sorted[-1],
    }


# ---------------------------------------------------------------------------
# Per-class instance-count metrics derived from OUR OWN matching (distinct
# from, and a cross-check against, Ultralytics' own AP/precision/recall,
# which integrate across all confidence levels rather than a single
# threshold — see docs/CONDITION_MODEL_4CLASS_ERROR_ANALYSIS.md §3).
# ---------------------------------------------------------------------------

@dataclass
class ClassCounts:
    gt_count: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> Optional[float]:
        denom = self.tp + self.fp
        return (self.tp / denom) if denom > 0 else None

    @property
    def recall(self) -> Optional[float]:
        denom = self.tp + self.fn
        return (self.tp / denom) if denom > 0 else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "gt_count": self.gt_count, "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision_at_match_threshold": self.precision,
            "recall_at_match_threshold": self.recall,
        }


def find_overlapping_gt_pairs(
    gt_list: Sequence[tuple[int, Box]],
    iou_threshold: float = 0.8,
) -> list[tuple[int, int, float]]:
    """Pure data-quality helper: within ONE image's GT boxes (each item is
    `(class_id, box)`), finds every pair whose IoU is >= `iou_threshold` —
    regardless of whether the two boxes share a class. Two heavily
    overlapping boxes of the SAME class are a near-duplicate-annotation
    candidate; two heavily overlapping boxes of DIFFERENT classes are an
    ambiguous/conflicting-label candidate. This function only reports the
    geometric fact (which pairs overlap, and by how much) — the caller
    decides what a same-class vs different-class overlap implies, and
    NOTHING here ever edits or drops an annotation (see Phase 15C rule:
    flag, never silently correct).

    Returns a list of `(index_a, index_b, iou)` with `index_a < index_b`,
    indices into `gt_list`.
    """
    pairs: list[tuple[int, int, float]] = []
    n = len(gt_list)
    for i in range(n):
        for j in range(i + 1, n):
            iou = iou_xyxy(gt_list[i][1], gt_list[j][1])
            if iou >= iou_threshold:
                pairs.append((i, j, iou))
    return pairs


@dataclass
class FamilyCounts:
    family_id: str
    class_name: str
    gt_count: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def recall(self) -> Optional[float]:
        denom = self.tp + self.fn
        return (self.tp / denom) if denom > 0 else None
