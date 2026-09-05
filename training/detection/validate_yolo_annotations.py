#!/usr/bin/env python3
"""training/detection/validate_yolo_annotations.py — Syntax/geometry QC
for a human-annotated YOLO bounding-box directory.

This checks ONLY mechanical correctness (can every line be parsed, are
coordinates in range, is the class id right, are there exact-duplicate
boxes). It intentionally NEVER judges whether a box is drawn in the
right place - that is the human annotator's and reviewer's call, always
(see training/detection/CLOSEUP_ANNOTATION_TEMPLATE.md and this
campaign's ANNOTATION_GUIDE.md). A file passing this check is
mechanically well-formed, not necessarily visually correct.

Checks performed, each reported separately (never conflated):
    - malformed lines (wrong field count / non-numeric fields)
    - class id other than the single allowed value (default 0)
    - coordinates outside [0, 1] (YOLO format is already normalized, so
      an in-range box can never fall outside the image by construction -
      this check catches the actual failure mode: an annotator or
      export tool emitting pixel coordinates instead of normalized ones)
    - zero-area (degenerate) boxes
    - missing label file for an image (vs. present-but-empty, which is
      the correct way to represent a genuine zero-panel image)
    - images with zero annotations (flagged for review, not treated as
      an error - a real negative image is legitimate, but this is
      unexpected for this specific SolNET-derived campaign since every
      source image was originally collected as a panel photo)
    - exact-duplicate boxes within one label file (same class + same
      coordinates to within a small tolerance)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
_COORD_TOLERANCE = 1e-6


def _parse_label_line(line: str, lineno: int, allowed_class_id: int) -> tuple[dict[str, Any] | None, str | None]:
    parts = line.split()
    if len(parts) != 5:
        return None, f"line {lineno}: expected 5 fields, got {len(parts)}"
    try:
        cls_id = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:])
    except ValueError:
        return None, f"line {lineno}: could not parse fields as int/float"
    if cls_id != allowed_class_id:
        return None, f"line {lineno}: class id {cls_id} is not the allowed {allowed_class_id}"
    if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
        return None, f"line {lineno}: coordinates outside [0,1]: {(cx, cy, w, h)}"
    if w <= 0.0 or h <= 0.0:
        return None, f"line {lineno}: zero/negative-area box: w={w}, h={h}"
    return {"class_id": cls_id, "cx": cx, "cy": cy, "w": w, "h": h, "line": lineno}, None


def _boxes_equal(a: dict, b: dict, tol: float = _COORD_TOLERANCE) -> bool:
    return (
        a["class_id"] == b["class_id"]
        and abs(a["cx"] - b["cx"]) < tol
        and abs(a["cy"] - b["cy"]) < tol
        and abs(a["w"] - b["w"]) < tol
        and abs(a["h"] - b["h"]) < tol
    )


def validate_label_file(label_path: Path, allowed_class_id: int = 0) -> dict[str, Any]:
    result: dict[str, Any] = {
        "filename": label_path.name, "errors": [], "boxes": [], "is_empty": False,
    }
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        result["is_empty"] = True
        return result

    boxes: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        box, error = _parse_label_line(line, lineno, allowed_class_id)
        if error:
            result["errors"].append(error)
            continue
        boxes.append(box)

    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if _boxes_equal(a, b):
                result["errors"].append(
                    f"lines {a['line']} and {b['line']}: exact-duplicate boxes"
                )

    result["boxes"] = boxes
    return result


def validate_directory(
    images_dir: Path, labels_dir: Path, allowed_class_id: int = 0
) -> dict[str, Any]:
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED_IMAGE_EXTENSIONS
    )
    per_file: list[dict[str, Any]] = []
    missing_labels: list[str] = []
    empty_labels: list[str] = []
    files_with_errors: list[dict[str, Any]] = []
    total_boxes = 0

    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.is_file():
            missing_labels.append(img_path.name)
            continue
        file_result = validate_label_file(label_path, allowed_class_id)
        file_result["image"] = img_path.name
        per_file.append(file_result)
        total_boxes += len(file_result["boxes"])
        if file_result["is_empty"]:
            empty_labels.append(img_path.name)
        if file_result["errors"]:
            files_with_errors.append({"image": img_path.name, "errors": file_result["errors"]})

    # Label files with no matching image (orphaned) - worth flagging.
    label_paths = {p.stem for p in labels_dir.glob("*.txt") if p.name != "NOTES.txt"}
    image_stems = {p.stem for p in image_paths}
    orphaned_labels = sorted(label_paths - image_stems)

    annotated_count = len(image_paths) - len(missing_labels)
    summary = {
        "total_images": len(image_paths),
        "annotated_images": annotated_count,
        "missing_label_files": missing_labels,
        "missing_label_count": len(missing_labels),
        "images_with_zero_annotations": empty_labels,
        "images_with_zero_annotations_count": len(empty_labels),
        "orphaned_label_files": orphaned_labels,
        "total_boxes": total_boxes,
        "files_with_errors": files_with_errors,
        "files_with_errors_count": len(files_with_errors),
        "clean_annotated_files_count": annotated_count - len(files_with_errors),
        "annotation_complete": len(missing_labels) == 0 and annotated_count == len(image_paths),
        "ready_for_next_stage": (
            len(missing_labels) == 0 and len(files_with_errors) == 0 and len(orphaned_labels) == 0
        ),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--allowed-class-id", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = validate_directory(args.images_dir, args.labels_dir, args.allowed_class_id)

    print(f"Images: {summary['total_images']}  Annotated: {summary['annotated_images']}  "
          f"Missing labels: {summary['missing_label_count']}  Total boxes: {summary['total_boxes']}")
    print(f"Images with zero annotations (review these - a real negative is legitimate, "
          f"an oversight is not): {summary['images_with_zero_annotations_count']}")
    print(f"Files with mechanical errors: {summary['files_with_errors_count']}")
    if summary["orphaned_label_files"]:
        print(f"WARNING: {len(summary['orphaned_label_files'])} label file(s) have no matching image: "
              f"{summary['orphaned_label_files'][:10]}")
    for entry in summary["files_with_errors"][:20]:
        print(f"  {entry['image']}:")
        for e in entry["errors"]:
            print(f"    - {e}")

    if summary["ready_for_next_stage"]:
        print("\nAll images annotated, no mechanical errors found. Human visual review still required.")
    else:
        print("\nNOT ready for the next stage yet - see missing/error counts above.")

    if args.output:
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote: {args.output}")

    return 0 if summary["ready_for_next_stage"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
