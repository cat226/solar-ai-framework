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
    - coordinates outside [0, 1]
    - zero-area (degenerate) boxes
    - a box whose EDGES (cx±w/2, cy±h/2), not just its center/dimensions
      individually, extend outside [0,1] - a box can have cx, cy, w, h
      each individually in-range while still hanging off the edge of the
      image (e.g. cx=0.05, w=0.5 -> left edge at -0.2); this is a real
      gap the tool-side validator (annotation_tool/core.py) checks live
      at save time, and is re-checked here independently as the
      authoritative offline QC pass
    - missing label file for an image (vs. present-but-empty, which is
      the correct way to represent a genuine zero-panel image)
    - images with zero annotations (flagged for review, not treated as
      an error - a real negative image is legitimate, but this is
      unexpected for this specific SolNET-derived campaign since every
      source image was originally collected as a panel photo)
    - exact-duplicate boxes within one label file (same class + same
      coordinates to within a small tolerance)

Also provided, run separately from the mechanical checks above:
    - verify_split_lock(): confirms every image in images_dir is one of
      the exact 200 images selected_200.csv locked in, with the exact
      same SHA-256 (catches both an image quietly migrating between
      train/val/test and any accidental content modification), and that
      the locked split counts (152/24/24) haven't drifted.
    - compute_statistics(): boxes-per-image distribution and per-split
      box counts, for the annotation report - never a claim about
      whether the boxes are visually correct, only how many exist.
    - render_contact_sheets(): draws every box on its image and tiles
      them into fixed-size grids for a human's visual QA pass - this
      renders what a human should look at, it never judges it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
_COORD_TOLERANCE = 1e-6
_EXPECTED_SPLIT_COUNTS = {"train": 152, "val": 24, "test": 24}


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
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2
    if x1 < -_COORD_TOLERANCE or y1 < -_COORD_TOLERANCE or x2 > 1 + _COORD_TOLERANCE or y2 > 1 + _COORD_TOLERANCE:
        return None, (
            f"line {lineno}: box edges extend outside the image: "
            f"({x1:.4f},{y1:.4f})-({x2:.4f},{y2:.4f})"
        )
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_split_lock(images_dir: Path, selected_csv_path: Path) -> dict[str, Any]:
    """Confirms images_dir contains EXACTLY the 200 images
    select_annotation_sample.py locked into selected_200.csv, with
    unchanged content (SHA-256 re-verified, not just filename presence),
    and that the locked split counts have not drifted. This is what
    stands between "an image quietly got moved/replaced/re-split" and
    finding out only after training on a corrupted split boundary."""
    rows = list(csv.DictReader(selected_csv_path.open(encoding="utf-8")))
    locked = {row["filename"]: row for row in rows}

    actual_files = {
        p.name for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_IMAGE_EXTENSIONS
    }
    locked_files = set(locked.keys())

    missing_from_images_dir = sorted(locked_files - actual_files)
    extra_in_images_dir = sorted(actual_files - locked_files)

    sha256_mismatches: list[dict[str, str]] = []
    for filename in sorted(locked_files & actual_files):
        actual_hash = _sha256_file(images_dir / filename)
        expected_hash = locked[filename]["sha256"]
        if actual_hash != expected_hash:
            sha256_mismatches.append({
                "filename": filename, "expected_sha256": expected_hash, "actual_sha256": actual_hash,
            })

    from collections import Counter
    actual_split_counts = dict(Counter(row["source_split"] for row in rows))
    split_counts_match = actual_split_counts == _EXPECTED_SPLIT_COUNTS

    locked_correctly = (
        not missing_from_images_dir
        and not extra_in_images_dir
        and not sha256_mismatches
        and split_counts_match
        and len(rows) == sum(_EXPECTED_SPLIT_COUNTS.values())
    )

    return {
        "locked_correctly": locked_correctly,
        "total_locked": len(rows),
        "expected_split_counts": _EXPECTED_SPLIT_COUNTS,
        "actual_split_counts_in_manifest": actual_split_counts,
        "split_counts_match": split_counts_match,
        "missing_from_images_dir": missing_from_images_dir,
        "extra_in_images_dir": extra_in_images_dir,
        "sha256_mismatches": sha256_mismatches,
    }


def compute_statistics(images_dir: Path, labels_dir: Path, selected_csv_path: Path) -> dict[str, Any]:
    """Boxes-per-image and per-split box counts - counts only, never a
    claim about correctness. Requires annotation to be complete (every
    image has a label file) to produce a meaningful per-split breakdown;
    reports partial statistics honestly otherwise."""
    rows = list(csv.DictReader(selected_csv_path.open(encoding="utf-8")))
    split_by_filename = {row["filename"]: row["source_split"] for row in rows}
    class_by_filename = {row["filename"]: row["source_class"] for row in rows}

    box_counts: dict[str, int] = {}
    per_split_boxes: dict[str, int] = {s: 0 for s in _EXPECTED_SPLIT_COUNTS}
    per_split_images_annotated: dict[str, int] = {s: 0 for s in _EXPECTED_SPLIT_COUNTS}
    per_class_boxes: dict[str, int] = {}

    for filename, split in split_by_filename.items():
        label_path = labels_dir / f"{Path(filename).stem}.txt"
        if not label_path.is_file():
            continue
        result = validate_label_file(label_path)
        n = len(result["boxes"])
        box_counts[filename] = n
        per_split_boxes[split] = per_split_boxes.get(split, 0) + n
        per_split_images_annotated[split] = per_split_images_annotated.get(split, 0) + 1
        cls = class_by_filename.get(filename, "unknown")
        per_class_boxes[cls] = per_class_boxes.get(cls, 0) + n

    counts = sorted(box_counts.values())
    n = len(counts)
    stats = {
        "images_with_a_label_file": n,
        "total_images_in_campaign": len(rows),
        "total_boxes": sum(counts),
        "min_boxes_per_image": counts[0] if n else None,
        "max_boxes_per_image": counts[-1] if n else None,
        "mean_boxes_per_image": (sum(counts) / n) if n else None,
        "median_boxes_per_image": (
            counts[n // 2] if n % 2 else (counts[n // 2 - 1] + counts[n // 2]) / 2
        ) if n else None,
        "zero_box_images": sum(1 for c in counts if c == 0),
        "per_split_box_counts": per_split_boxes,
        "per_split_images_annotated": per_split_images_annotated,
        "per_split_expected_images": _EXPECTED_SPLIT_COUNTS,
        "per_source_class_box_counts": per_class_boxes,
    }
    return stats


def render_contact_sheets(
    images_dir: Path, labels_dir: Path, output_dir: Path, images_per_sheet: int = 20, thumb_size: int = 220
) -> list[Path]:
    """Draws every saved box on its image and tiles them into grids for
    a human QA pass. Purely a rendering aid - draws exactly what is in
    the label files, judges nothing. Returns the list of sheet paths
    written. Requires Pillow (already a project dependency)."""
    from PIL import Image, ImageDraw

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED_IMAGE_EXTENSIONS
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets: list[Path] = []

    cols = 5
    rows_per_sheet = max(1, -(-images_per_sheet // cols))
    sheet_w = cols * thumb_size
    sheet_h = rows_per_sheet * (thumb_size + 16)

    for sheet_start in range(0, len(image_paths), images_per_sheet):
        batch = image_paths[sheet_start:sheet_start + images_per_sheet]
        sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 20))
        draw_sheet = ImageDraw.Draw(sheet)
        for i, img_path in enumerate(batch):
            label_path = labels_dir / f"{img_path.stem}.txt"
            with Image.open(img_path) as im:
                thumb = im.convert("RGB").resize((thumb_size, thumb_size))
            tdraw = ImageDraw.Draw(thumb)
            box_count = 0
            if label_path.is_file():
                result = validate_label_file(label_path)
                box_count = len(result["boxes"])
                for b in result["boxes"]:
                    x1 = (b["cx"] - b["w"] / 2) * thumb_size
                    y1 = (b["cy"] - b["h"] / 2) * thumb_size
                    x2 = (b["cx"] + b["w"] / 2) * thumb_size
                    y2 = (b["cy"] + b["h"] / 2) * thumb_size
                    tdraw.rectangle([x1, y1, x2, y2], outline=(50, 220, 90), width=2)
            col, row = i % cols, i // cols
            x, y = col * thumb_size, row * (thumb_size + 16)
            sheet.paste(thumb, (x, y))
            label_color = (255, 90, 90) if not label_path.is_file() else (200, 200, 200)
            draw_sheet.text((x + 2, y + thumb_size + 1), f"{img_path.stem[:22]} ({box_count})", fill=label_color)
        sheet_index = sheet_start // images_per_sheet
        out_path = output_dir / f"contact_sheet_{sheet_index:03d}.png"
        sheet.save(out_path)
        sheets.append(out_path)
    return sheets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--allowed-class-id", type=int, default=0)
    parser.add_argument("--selected-csv", type=Path, default=None,
                         help="selected_200.csv - enables split-lock verification and statistics.")
    parser.add_argument("--contact-sheets-dir", type=Path, default=None,
                         help="If given, render annotated contact sheets here for human QA.")
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

    output_payload: dict[str, Any] = {"mechanical_validation": summary}

    if args.selected_csv:
        lock = verify_split_lock(args.images_dir, args.selected_csv)
        output_payload["split_lock"] = lock
        print(f"\nSplit lock: {'OK' if lock['locked_correctly'] else 'FAILED'} "
              f"(expected {lock['expected_split_counts']}, manifest has {lock['actual_split_counts_in_manifest']})")
        if lock["missing_from_images_dir"]:
            print(f"  MISSING from images/: {lock['missing_from_images_dir'][:10]}")
        if lock["extra_in_images_dir"]:
            print(f"  EXTRA (not in locked selection): {lock['extra_in_images_dir'][:10]}")
        if lock["sha256_mismatches"]:
            print(f"  SHA-256 MISMATCH (content changed): {lock['sha256_mismatches'][:5]}")

        stats = compute_statistics(args.images_dir, args.labels_dir, args.selected_csv)
        output_payload["statistics"] = stats
        print(f"\nStatistics: {stats['images_with_a_label_file']}/{stats['total_images_in_campaign']} labeled, "
              f"{stats['total_boxes']} total boxes, "
              f"mean={stats['mean_boxes_per_image']}, median={stats['median_boxes_per_image']}")
        print(f"  Per-split boxes: {stats['per_split_box_counts']}")

    if args.contact_sheets_dir:
        sheets = render_contact_sheets(args.images_dir, args.labels_dir, args.contact_sheets_dir)
        print(f"\nWrote {len(sheets)} contact sheet(s) to {args.contact_sheets_dir}")
        output_payload["contact_sheets"] = [str(s) for s in sheets]

    if args.output:
        args.output.write_text(json.dumps(output_payload, indent=2))
        print(f"\nWrote: {args.output}")

    return 0 if summary["ready_for_next_stage"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
