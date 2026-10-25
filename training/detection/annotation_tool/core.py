"""training/detection/annotation_tool/core.py — Pure, testable logic for
the local YOLO bounding-box annotation tool.

Deliberately separated from server.py (the thin HTTP wrapper) and from
any browser code, so every save/load/progress rule is directly
unit-testable without spinning up a server or a browser.

This module draws boxes for exactly one purpose: to let a HUMAN save
what they drew. Nothing here infers, predicts, or auto-generates a box -
see training/detection/CLOSEUP_ANNOTATION_TEMPLATE.md and this
campaign's ANNOTATION_GUIDE.md for why that line is never crossed.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

CLASS_ID = 0
CLASS_NAME = "solar_panel"
_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def list_images(images_dir: Path) -> list[str]:
    """Deterministic (sorted) filename list - the order the tool's
    Next/Previous navigation uses."""
    return sorted(
        p.name for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )


def label_path_for(labels_dir: Path, image_filename: str) -> Path:
    # Path(...).name strips any directory component a caller might smuggle
    # in - the label always lands directly in labels_dir, never elsewhere.
    stem = Path(image_filename).name
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    return labels_dir / f"{stem}.txt"


def load_boxes(labels_dir: Path, image_filename: str) -> list[dict[str, float]] | None:
    """Returns:
      - None if the image has never been saved (no label file at all -
        "not yet visited", distinct from a confirmed zero-panel image).
      - [] if a label file exists but is empty ("visited, zero panels").
      - a list of box dicts otherwise.

    Malformed lines are silently skipped here (this is the *editor's*
    read path, not the QC validator - training/detection/validate_yolo_annotations.py
    is the authority on flagging malformed content; this function's job
    is just to show the user whatever can be parsed so they can fix it)."""
    path = label_path_for(labels_dir, image_filename)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    boxes: list[dict[str, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            continue
        boxes.append({"class_id": cls_id, "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes


def validate_box(box: dict[str, Any]) -> None:
    """Raises ValueError with a human-readable reason if a box is not
    save-able. Called before every write so a UI bug or bad input can
    never corrupt a label file with garbage coordinates."""
    required = ("cx", "cy", "w", "h")
    for key in required:
        if key not in box:
            raise ValueError(f"box missing required field '{key}': {box}")
    cx, cy, w, h = (float(box[k]) for k in required)
    if not all(math.isfinite(v) for v in (cx, cy, w, h)):
        raise ValueError(f"box contains a non-finite value: {box}")
    if w <= 0.0 or h <= 0.0:
        raise ValueError(f"zero/negative-area box rejected: {box}")
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        raise ValueError(f"box center out of [0,1] bounds: {box}")
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2
    tol = 1e-6
    if x1 < -tol or y1 < -tol or x2 > 1 + tol or y2 > 1 + tol:
        raise ValueError(f"box extends outside the image bounds: {box} -> edges ({x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f})")


def save_boxes(labels_dir: Path, image_filename: str, boxes: list[dict[str, Any]]) -> Path:
    """Validates every box, then writes atomically (temp file + os.replace,
    which is atomic on both POSIX and Windows) so a crash or interrupted
    write mid-save can never leave a truncated/corrupted label file
    behind - the previous version stays intact until the new one is
    fully written and the rename completes."""
    for box in boxes:
        validate_box(box)

    labels_dir.mkdir(parents=True, exist_ok=True)
    path = label_path_for(labels_dir, image_filename)
    lines = [
        f"{CLASS_ID} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
        for b in boxes
    ]
    content = "\n".join(lines) + ("\n" if lines else "")

    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)  # atomic on POSIX and Windows (NTFS)
    return path


def compute_progress(images_dir: Path, labels_dir: Path) -> dict[str, Any]:
    images = list_images(images_dir)
    annotated_flags = {img: label_path_for(labels_dir, img).is_file() for img in images}
    annotated = [img for img, done in annotated_flags.items() if done]
    box_counts = {}
    for img in annotated:
        boxes = load_boxes(labels_dir, img)
        box_counts[img] = len(boxes) if boxes is not None else 0
    return {
        "total": len(images),
        "annotated": len(annotated),
        "remaining": len(images) - len(annotated),
        "images": [
            {"filename": img, "annotated": annotated_flags[img], "box_count": box_counts.get(img)}
            for img in images
        ],
        "total_boxes_so_far": sum(box_counts.values()),
    }
