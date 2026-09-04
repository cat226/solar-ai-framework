"""Create a small, deterministic smoke-test subset from an already-prepared,
audited MobileNet classification dataset (the output of prepare_dataset.py).

Mirrors training/detection/create_smoke_dataset.py's design for the YOLO
pipeline, adapted to ImageFolder-style class-subfolder layout instead of
images/labels pairs. Never re-derives anything from raw source data, never
re-runs dataset preparation, never modifies the original prepared dataset.
It only *selects and copies* a small, deterministic slice of already-audited
images - the same provenance chain documented in
training/classification/DATASET_SOURCES.md / LICENSE_COMPATIBILITY.md
applies unchanged to the subset.

Selection is deterministic: within each (split, class) pair, records are
sorted by their sha256 digest and the first N are taken - the same source
manifest always produces the same subset, with no randomness involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

SPLITS = ["train", "val", "test"]
DEFAULT_PER_CLASS_PER_SPLIT = 6


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_smoke_dataset(
    source_root: Path,
    output_root: Path,
    *,
    classes: list[str],
    per_class_per_split: int = DEFAULT_PER_CLASS_PER_SPLIT,
) -> Path:
    """Build a small deterministic subset of an already-prepared dataset,
    restricted to `classes` (must all exist in the source manifest with at
    least `per_class_per_split` records in every split).

    Returns the path to the written smoke-dataset manifest.json.

    Raises RuntimeError if the source manifest is missing, if `classes`
    contains a name absent from the source, if any (class, split) has fewer
    than `per_class_per_split` records, or if a selected record's file is
    missing - this never silently skips or repairs a problem.
    """
    manifest_path = source_root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"no manifest.json found at {manifest_path} - run prepare_dataset.py first; "
            "this utility only subsets an already-prepared dataset, it never creates one."
        )
    source_manifest_hash = _sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    records_by_split_class: dict[str, dict[str, list[dict[str, Any]]]] = {
        s: defaultdict(list) for s in SPLITS
    }
    for rec in manifest["records"]:
        split = rec.get("split")
        if split in records_by_split_class:
            records_by_split_class[split][rec["class_name"]].append(rec)

    missing_classes = [c for c in classes if not any(
        c in records_by_split_class[s] for s in SPLITS
    )]
    if missing_classes:
        raise RuntimeError(f"class(es) not present in source manifest at all: {missing_classes}")

    for split in SPLITS:
        for cls in classes:
            (output_root / split / cls).mkdir(parents=True, exist_ok=True)

    smoke_records: list[dict[str, Any]] = []
    for split in SPLITS:
        for cls in classes:
            available = records_by_split_class[split].get(cls, [])
            # Deterministic: sort by sha256, always the same order for the
            # same source manifest, regardless of dict/JSON ordering.
            sorted_records = sorted(available, key=lambda r: r["sha256"])
            if len(sorted_records) < per_class_per_split:
                raise ValueError(
                    f"requested {per_class_per_split} '{cls}'/{split} records but only "
                    f"{len(sorted_records)} exist in the source manifest"
                )
            for rec in sorted_records[:per_class_per_split]:
                src = Path(rec["source_path"])
                if not src.is_file():
                    raise RuntimeError(f"source image missing: {src}")
                dst = output_root / split / cls / rec["original_filename"]
                shutil.copyfile(src, dst)
                smoke_records.append({
                    "split": split,
                    "class_name": cls,
                    "sha256": rec["sha256"],
                    "original_filename": rec["original_filename"],
                    "dest_path": str(dst),
                })

    smoke_manifest = {
        "source_dataset_root": str(source_root.resolve()),
        "source_manifest_hash": source_manifest_hash,
        "classes": classes,
        "per_class_per_split_requested": per_class_per_split,
        "counts": {
            split: {cls: sum(1 for r in smoke_records if r["split"] == split and r["class_name"] == cls) for cls in classes}
            for split in SPLITS
        },
        "records": smoke_records,
    }
    smoke_manifest_path = output_root / "manifest.json"
    smoke_manifest_path.write_text(json.dumps(smoke_manifest, indent=2), encoding="utf-8")
    return smoke_manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Output of prepare_dataset.py")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the smoke dataset")
    parser.add_argument("--classes", type=str, nargs="+", required=True)
    parser.add_argument("--per-class-per-split", type=int, default=DEFAULT_PER_CLASS_PER_SPLIT)
    args = parser.parse_args()

    manifest_path = create_smoke_dataset(
        args.source_root, args.output,
        classes=args.classes,
        per_class_per_split=args.per_class_per_split,
    )
    print(f"smoke dataset manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
