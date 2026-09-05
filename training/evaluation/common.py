"""training/evaluation/common.py — Pure, independently-testable helper
functions shared by the evaluation scripts in this package.

Kept dependency-light (PIL + numpy only, both already required by the
application) and free of any model-loading or I/O side effects so every
function here can be unit tested directly.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """SHA-256 digest of a file's raw bytes, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Difference hash (dHash): a standard, simple perceptual hash.

    Converts to grayscale, resizes to (hash_size+1) x hash_size, and encodes
    whether each pixel is brighter than its right-hand neighbour as one bit.
    Two images that look visually similar produce hashes with a small
    Hamming distance, even under mild recompression/resizing - unlike
    SHA-256, which changes completely for a single-byte difference. This is
    a real, standard technique (used by e.g. the `imagehash` PyPI package's
    dhash implementation), reimplemented here directly on PIL+numpy to avoid
    adding a new dependency for one audit script.
    """
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two same-width integer hashes."""
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# Detection geometry
# ---------------------------------------------------------------------------

def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes in the same
    coordinate space. Returns 0.0 for two zero-area or non-overlapping boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0.0:
        return 0.0
    return inter_area / union


def match_detections_to_ground_truth(
    pred_boxes: Sequence[Sequence[float]],
    pred_confidences: Sequence[float],
    gt_boxes: Sequence[Sequence[float]],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int, list[float]]:
    """Greedy IoU matching of predicted boxes to ground-truth boxes for one
    image, following the standard single-class detection-metric protocol:
    predictions are considered highest-confidence-first, each ground-truth
    box may be claimed by at most one prediction, and a prediction with no
    available ground-truth box at IoU >= threshold is a false positive.

    Returns:
        (true_positives, false_positives, false_negatives, matched_ious)
        matched_ious: the IoU of each true-positive match, for reporting.
    """
    order = sorted(range(len(pred_boxes)), key=lambda i: pred_confidences[i], reverse=True)
    claimed = [False] * len(gt_boxes)
    tp = 0
    fp = 0
    matched_ious: list[float] = []

    for i in order:
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gt_boxes):
            if claimed[j]:
                continue
            cur = iou(pred_boxes[i], gt)
            if cur > best_iou:
                best_iou = cur
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            claimed[best_j] = True
            tp += 1
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = claimed.count(False)
    return tp, fp, fn, matched_ious


def yolo_label_to_xyxy(
    line: str, image_width: int, image_height: int
) -> tuple[int, tuple[float, float, float, float]]:
    """Parse one line of a YOLO-format label file
    (`class cx cy w h`, all normalized 0-1) into (class_id, [x1, y1, x2, y2])
    pixel coordinates."""
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Malformed YOLO label line (expected 5 fields): {line!r}")
    class_id = int(parts[0])
    cx, cy, w, h = (float(p) for p in parts[1:])
    x1 = (cx - w / 2) * image_width
    y1 = (cy - h / 2) * image_height
    x2 = (cx + w / 2) * image_width
    y2 = (cy + h / 2) * image_height
    return class_id, (x1, y1, x2, y2)


def load_yolo_ground_truth(label_path: Path, image_width: int, image_height: int) -> list[tuple[float, float, float, float]]:
    """Read a YOLO-format label file (possibly empty/absent = no objects)
    and return ground-truth boxes in pixel [x1, y1, x2, y2] coordinates."""
    if not label_path.is_file():
        return []
    boxes: list[tuple[float, float, float, float]] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        _, box = yolo_label_to_xyxy(line, image_width, image_height)
        boxes.append(box)
    return boxes


# ---------------------------------------------------------------------------
# Classification metrics (pure, no sklearn dependency required)
# ---------------------------------------------------------------------------

def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> list[list[int]]:
    """Row = true class, column = predicted class, in the order of `labels`."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        matrix[index[t]][index[p]] += 1
    return matrix


def per_class_prf1(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Precision/recall/F1/support per class, computed directly from counts
    (no external dependency)."""
    result: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        result[label] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }
    return result


def macro_and_weighted_prf1(
    per_class: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Macro (unweighted mean) and support-weighted precision/recall/F1
    aggregated from a per-class metrics dict (see per_class_prf1)."""
    total_support = sum(v["support"] for v in per_class.values())
    n = len(per_class)
    macro_p = sum(v["precision"] for v in per_class.values()) / n
    macro_r = sum(v["recall"] for v in per_class.values()) / n
    macro_f1 = sum(v["f1"] for v in per_class.values()) / n
    if total_support:
        weighted_p = sum(v["precision"] * v["support"] for v in per_class.values()) / total_support
        weighted_r = sum(v["recall"] * v["support"] for v in per_class.values()) / total_support
        weighted_f1 = sum(v["f1"] * v["support"] for v in per_class.values()) / total_support
    else:
        weighted_p = weighted_r = weighted_f1 = 0.0
    return {
        "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1,
        "weighted_precision": weighted_p, "weighted_recall": weighted_r, "weighted_f1": weighted_f1,
    }


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


# ---------------------------------------------------------------------------
# Output location (E: drive policy, mirrors training/detection/train_yolo.py's
# _compute_default_project_dir - see this project's storage policy)
# ---------------------------------------------------------------------------

def default_output_root(env: Optional[dict] = None, platform: Optional[str] = None) -> Path:
    """Pure function (env/platform injectable) so this is unit-testable
    without an actual E: drive - mirrors
    training/detection/train_yolo.py's _compute_default_project_dir."""
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    if env.get("SOLAR_AI_DATA_ROOT"):
        return Path(env["SOLAR_AI_DATA_ROOT"]) / "evaluation_runs"
    if platform == "win32":
        return Path("E:/Solar AI Training Images/evaluation_runs")
    return Path("training/evaluation/runs")
