#!/usr/bin/env python3
"""scripts/build_condition_4class_classifier_dataset.py — Phase 16-B dataset
build: derive an ImageFolder-style CLASSIFICATION dataset from the existing,
already-audited, already-leakage-split `condition_4class_v1` DETECTION
dataset.

Motivation: docs/CONDITION_MODEL_IMAGE_LEVEL_AGREEMENT.md (Phase 16-A.1)
found the active box-detection condition model's weakness is dominated by
localization noise, not class recognition — evidence a crop-level
CLASSIFIER (no localization requirement) is worth trying.

No new photography, no new annotation. Every image keeps its EXISTING
train/valid/test split assignment from `splits_4class_manifest.csv`
verbatim — no re-grouping, no re-splitting, so this derived dataset is
leakage-clean by construction (a relabeling of an already leakage-clean
split is still leakage-clean).

Label derivation (documented, deterministic, never fabricated):
  For each image, parse its EXISTING YOLO box annotations (unchanged,
  never edited) and pick ONE "dominant" class via
  training.detection.ground_pv.condition_taxonomy.dominant_class(): the
  class with the most box instances, ties broken by total box area, then
  alphabetically. This is an explicit DERIVED label, not an independently
  verified image-level judgement — an image with e.g. 6 DUSTY boxes and 1
  BIRD_DROP box becomes a single "DUSTY" classifier example, discarding
  the minority annotation for this reformulation. That information loss is
  disclosed in the report, not hidden.

Output layout (torchvision.datasets.ImageFolder-compatible):
    <output_dir>/{train,valid,test}/{CLEAN,DUSTY,BIRD_DROP,PHYSICAL_DAMAGE}/<image>.jpg

Images are COPIED (never modified) from condition_4class_v1. The source
dataset is never written to.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.condition_error_analysis import parse_yolo_label_text  # noqa: E402
from training.detection.ground_pv.condition_taxonomy import (  # noqa: E402
    CONDITION_4CLASS_NAMES,
    CONDITION_4CLASS_ORDER,
    dominant_class,
)

_SPLITS = ("train", "valid", "test")
_DEFAULT_SOURCE = Path("datasets/ground_pv_research/condition_4class_v1")
_DEFAULT_OUTPUT = Path("datasets/ground_pv_research/condition_4class_classifier_v1")


def _sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=_DEFAULT_SOURCE)
    ap.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    ap.add_argument("--report", type=Path,
                     default=Path("datasets/ground_pv_research/condition_4class_classifier_v1_report.json"))
    args = ap.parse_args(argv)

    if not args.source.is_dir():
        raise SystemExit(f"source dataset not found: {args.source}")

    if args.output.exists():
        raise SystemExit(f"output already exists, refusing to overwrite: {args.output} "
                          "(delete it first if you intend to rebuild)")

    per_split_class_counts: dict[str, Counter] = {s: Counter() for s in _SPLITS}
    minority_annotation_images = 0  # images with >1 distinct class present (info-loss disclosure)
    images_no_boxes = 0
    per_image_records: list[dict[str, Any]] = []

    for split in _SPLITS:
        img_dir = args.source / "images" / split / "images"
        lbl_dir = args.source / "images" / split / "labels"
        if not img_dir.is_dir():
            raise SystemExit(f"missing expected source split dir: {img_dir}")

        image_files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        for img_path in image_files:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            text = lbl_path.read_text(encoding="utf-8") if lbl_path.is_file() else ""
            boxes = parse_yolo_label_text(text, 640, 640)  # dataset is fixed 640x640 (verified at build time)
            if not boxes:
                images_no_boxes += 1
                continue

            instances: list[tuple[str, float]] = []
            distinct_classes: set[str] = set()
            for cls_id, (x1, y1, x2, y2) in boxes:
                cls_name = CONDITION_4CLASS_NAMES.get(cls_id)
                if cls_name is None:
                    continue  # not an active-taxonomy class id — should not occur in this source dataset
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                instances.append((cls_name, area))
                distinct_classes.add(cls_name)
            if not instances:
                images_no_boxes += 1
                continue

            label = dominant_class(instances)
            if len(distinct_classes) > 1:
                minority_annotation_images += 1

            dest_dir = args.output / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest_dir / img_path.name)

            per_split_class_counts[split][label] += 1
            per_image_records.append({
                "split": split, "image": img_path.name, "derived_label": label,
                "distinct_classes_present": sorted(distinct_classes),
                "n_boxes": len(instances),
            })

    # Verify actual image dimensions match the 640x640 assumption used above (sanity check,
    # not a silent assumption) — spot-check a handful rather than every file for speed.
    from PIL import Image as PILImage
    sample_paths = [args.source / "images" / r["split"] / "images" / r["image"] for r in per_image_records[:25]]
    for p in sample_paths:
        with PILImage.open(p) as im:
            if im.size != (640, 640):
                raise SystemExit(f"assumption violated: {p} is {im.size}, expected (640, 640) — "
                                  "box-area computation above would be wrong for non-square/non-640 images.")

    report = {
        "schema_version": "1.0",
        "phase": "16-B — classifier dataset derivation",
        "source_dataset": str(args.source),
        "output_dataset": str(args.output),
        "label_derivation_rule": (
            "dominant_class(): majority by box instance COUNT, ties broken by total box AREA, "
            "then alphabetically — see training/detection/ground_pv/condition_taxonomy.py"
        ),
        "split_assignment": "verbatim from splits_4class_manifest.csv — no re-splitting, no re-grouping",
        "per_split_class_counts": {s: dict(c) for s, c in per_split_class_counts.items()},
        "total_images_written": len(per_image_records),
        "images_with_no_boxes_skipped": images_no_boxes,
        "images_with_minority_annotation_discarded": minority_annotation_images,
        "minority_annotation_disclosure": (
            f"{minority_annotation_images} of {len(per_image_records)} images had more than one "
            "distinct active class present in their box annotations; only the dominant class was "
            "kept as this image's single classifier label, per the reformulation's single-label "
            "design. This is a real information loss versus the detection dataset, disclosed here "
            "not hidden."
        ),
        "active_taxonomy": list(CONDITION_4CLASS_ORDER),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    manifest_path = args.output / "classifier_labels_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "image", "derived_label", "distinct_classes_present", "n_boxes"])
        writer.writeheader()
        for r in per_image_records:
            row = dict(r)
            row["distinct_classes_present"] = ";".join(row["distinct_classes_present"])
            writer.writerow(row)

    print(json.dumps(report["per_split_class_counts"], indent=2))
    print(f"total images written: {len(per_image_records)}")
    print(f"images with no boxes skipped: {images_no_boxes}")
    print(f"images with minority annotation discarded: {minority_annotation_images}")
    print(f"report: {args.report}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
