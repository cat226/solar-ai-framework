"""Create a small, deterministic smoke-test subset from an already-prepared,
audited YOLO dataset (the output of prepare_dataset.py).

This never re-derives anything from raw source data, never re-runs dataset
preparation, never modifies the original prepared dataset, and never alters
a label. It only *selects and copies* a small, deterministic slice of
already-audited image+label pairs - the same provenance chain documented in
training/detection/PROVENANCE_VERIFICATION.md and AUDIT_REPORT.md applies
unchanged to the subset, since nothing about the underlying data changes.

Selection is deterministic: within each split, records are sorted by their
stable `id` field and the first N are taken - the same source manifest
always produces the same subset, with no randomness involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

SPLITS = ["train", "val", "test"]
DEFAULT_PER_SPLIT = 30


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_label_file(label_path: Path, num_classes: int) -> list[str]:
    """Parse a YOLO-format label file and return a list of error strings
    (empty if valid). An empty file is valid (a negative/background image).
    Never repairs anything - a bad line is reported, not silently dropped
    or fixed."""
    errors: list[str] = []
    text = label_path.read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}:{line_no}: expected 5 fields, got {len(parts)}: {line!r}")
            continue
        cls_str, cx_str, cy_str, w_str, h_str = parts
        if not cls_str.isdigit():
            errors.append(f"{label_path}:{line_no}: class id is not a non-negative integer: {cls_str!r}")
            continue
        cls_id = int(cls_str)
        if not (0 <= cls_id < num_classes):
            errors.append(f"{label_path}:{line_no}: class id {cls_id} outside valid range [0, {num_classes})")
        try:
            cx, cy, w, h = (float(v) for v in (cx_str, cy_str, w_str, h_str))
        except ValueError:
            errors.append(f"{label_path}:{line_no}: bounding box values are not valid floats: {line!r}")
            continue
        for name, value in (("x_center", cx), ("y_center", cy), ("width", w), ("height", h)):
            if not (0.0 <= value <= 1.0):
                errors.append(f"{label_path}:{line_no}: {name}={value} outside normalized range [0,1]")
        if w <= 0.0 or h <= 0.0:
            errors.append(f"{label_path}:{line_no}: non-positive box dimensions (w={w}, h={h})")
    return errors


def create_smoke_dataset(
    source_root: Path,
    output_root: Path,
    *,
    per_split: dict[str, int] | int = DEFAULT_PER_SPLIT,
) -> Path:
    """Build a small deterministic subset of an already-prepared dataset.

    Returns the path to the written smoke-dataset manifest.json.

    Raises RuntimeError if the source manifest is missing, if a selected
    record's image or label file is missing, or if any selected label file
    fails validation - this never silently skips or repairs a problem.
    """
    manifest_path = source_root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"no manifest.json found at {manifest_path} - run prepare_dataset.py first; "
            "this utility only subsets an already-prepared dataset, it never creates one."
        )
    source_manifest_hash = _sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    class_names: list[str] = manifest["class_names"]

    per_split_map: dict[str, int] = (
        dict(per_split) if isinstance(per_split, dict) else {s: per_split for s in SPLITS}
    )
    unknown_splits = set(per_split_map) - set(SPLITS)
    if unknown_splits:
        raise ValueError(f"per_split contains unknown split name(s): {unknown_splits}")

    records_by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for rec in manifest["records"]:
        split = rec.get("split")
        if split in records_by_split:
            records_by_split[split].append(rec)

    selected: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        n = per_split_map.get(split, 0)
        # Deterministic: sort by the stable `id` field, always the same order
        # for the same source manifest, regardless of dict/JSON ordering.
        sorted_records = sorted(records_by_split[split], key=lambda r: r["id"])
        if n > len(sorted_records):
            raise ValueError(
                f"requested {n} '{split}' records but only {len(sorted_records)} exist in the source manifest"
            )
        selected[split] = sorted_records[:n]

    for split in SPLITS:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

    smoke_records: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    for split in SPLITS:
        for rec in selected[split]:
            src_img = Path(rec["image_path"])
            src_lbl = Path(rec["label_path"])
            if not src_img.is_file():
                raise RuntimeError(f"source image missing for record {rec['id']!r}: {src_img}")
            if not src_lbl.is_file():
                raise RuntimeError(f"source label missing for record {rec['id']!r}: {src_lbl}")

            label_errors = _validate_label_file(src_lbl, num_classes=len(class_names))
            if label_errors:
                # Collected, not raised immediately, so a single run reports every
                # problem in the selection at once rather than stopping at the first.
                validation_errors.extend(label_errors)
                continue

            dst_img = output_root / split / "images" / src_img.name
            dst_lbl = output_root / split / "labels" / src_lbl.name
            dst_img.write_bytes(src_img.read_bytes())
            dst_lbl.write_bytes(src_lbl.read_bytes())

            smoke_records.append({
                "id": rec["id"],
                "split": split,
                "sha256": rec["sha256"],
                "num_instances": rec["num_instances"],
                "image_path": str(dst_img),
                "label_path": str(dst_lbl),
            })

    if validation_errors:
        raise RuntimeError(
            "refusing to build smoke dataset: the source (already-audited) dataset "
            f"contains {len(validation_errors)} invalid label(s), first few:\n"
            + "\n".join(validation_errors[:10])
        )

    data_yaml = {
        "path": str(output_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    (output_root / "data.yaml").write_text(yaml.safe_dump(data_yaml), encoding="utf-8")

    smoke_manifest = {
        "source_dataset_root": str(source_root.resolve()),
        "source_manifest_hash": source_manifest_hash,
        "class_names": class_names,
        "per_split_requested": per_split_map,
        "counts": {split: len(selected[split]) for split in SPLITS},
        "records": smoke_records,
    }
    smoke_manifest_path = output_root / "manifest.json"
    smoke_manifest_path.write_text(json.dumps(smoke_manifest, indent=2), encoding="utf-8")
    return smoke_manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Output of prepare_dataset.py")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the smoke dataset")
    parser.add_argument("--train", type=int, default=DEFAULT_PER_SPLIT)
    parser.add_argument("--val", type=int, default=DEFAULT_PER_SPLIT)
    parser.add_argument("--test", type=int, default=DEFAULT_PER_SPLIT)
    args = parser.parse_args()

    manifest_path = create_smoke_dataset(
        args.source_root, args.output,
        per_split={"train": args.train, "val": args.val, "test": args.test},
    )
    print(f"smoke dataset manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
