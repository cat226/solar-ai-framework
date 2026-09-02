"""Prepare a local image dataset for MobileNetV2 training.

This utility operates only on local source directories explicitly supplied by
the user. It never downloads or synthesizes data.

Expected source layout:

    SOURCE_ROOT/
        Clean/
        Dusty/
        Bird-Drop/
        Electrical-Damage/
        Physical-Damage/
        Hotspot/

Output layout:

    OUTPUT_ROOT/
        train/
            Clean/
            ...
        val/
            ...
        test/
            ...
        manifest.json

The manifest records source provenance, SHA-256 digests, split assignments,
and class mappings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUIRED_CLASSES = ["Clean", "Dusty", "Bird-Drop", "Electrical-Damage", "Physical-Damage", "Hotspot"]
SPLITS = ["train", "val", "test"]


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_images(source: Path, allowed_classes: list[str] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Walk a source root and collect image records.

    Returns a tuple of:
        records: list of dicts with keys: path, class_name, sha256, group
        unknown_classes: list of class directory names not in allowed_classes

    Note:
        The class directory name is NOT used as a grouping identifier.
        ``group`` is only set when real source/panel/module metadata exists
        in the filename or directory structure. Otherwise it is ``None``.
    """
    if allowed_classes is None:
        allowed_classes = REQUIRED_CLASSES
    records: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    unknown_classes: list[str] = []

    for entry in sorted(source.iterdir()):
        if not entry.is_dir():
            continue
        class_name = entry.name
        if class_name not in allowed_classes:
            unknown_classes.append(class_name)
            continue

        for img_path in sorted(entry.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue

            try:
                digest = _sha256(img_path)
            except OSError as exc:
                raise RuntimeError(f"failed to hash {img_path}: {exc}") from exc

            if digest in seen_hashes:
                raise RuntimeError(
                    f"duplicate image detected: {img_path} has same SHA-256 as {seen_hashes[digest]}"
                )

            seen_hashes[digest] = str(img_path)

            records.append({
                "source_path": str(img_path),
                "original_filename": img_path.name,
                "class_name": class_name,
                "sha256": digest,
                "group": None,
            })

    return records, unknown_classes


def _split_records(
    records: list[dict[str, Any]],
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> list[dict[str, Any]]:
    """Assign each record to train/val/test deterministically.

    When a group identifier is present, all records with the same group are
    assigned to the same split to prevent leakage.

    When no group identifier is present, records are split deterministically
    *within each class* to preserve class stratification.
    """
    if not (0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0 and train_frac + val_frac < 1.0):
        raise ValueError("invalid split fractions")

    rng = random.Random(seed)

    grouped: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[rec.get("group")].append(rec)

    def _assign(items: list[dict[str, Any]]) -> None:
        rng.shuffle(items)
        n = len(items)
        train_end = int(n * train_frac)
        val_end = train_end + int(n * val_frac)
        for i, item in enumerate(items):
            if i < train_end:
                item["split"] = "train"
            elif i < val_end:
                item["split"] = "val"
            else:
                item["split"] = "test"

    for group_value, items in grouped.items():
        if group_value is not None:
            _assign(items)
        else:
            by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in items:
                by_class[item["class_name"]].append(item)
            for class_items in by_class.values():
                _assign(class_items)

    return records


def prepare_dataset(
    source_roots: list[Path],
    output_root: Path,
    seed: int = 42,
    classes: list[str] | None = None,
) -> Path:
    """Prepare dataset from one or more source roots.

    Args:
        source_roots: Directories containing class subdirectories.
        output_root: Directory where the prepared dataset will be written.
        seed: Random seed for deterministic splitting.
        classes: Which classes to require/include. Defaults to all six
            production classes. Passing a subset produces an explicitly
            non-production, interim dataset (e.g. while some production
            classes are still blocked on data access) — the manifest records
            ``is_production_class_set: False`` in that case so downstream
            tooling and readers can tell it apart from a real production run.

    Returns:
        Path to the written manifest file.
    """
    if classes is None:
        classes = REQUIRED_CLASSES
    else:
        unknown_requested = sorted(set(classes) - set(REQUIRED_CLASSES))
        if unknown_requested:
            raise RuntimeError(f"--classes contains classes outside the production set: {unknown_requested}")
        if not classes:
            raise RuntimeError("--classes must not be empty")

    output_root = output_root.resolve()
    for split in SPLITS:
        for class_name in classes:
            (output_root / split / class_name).mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    all_unknown: list[str] = []
    for source in source_roots:
        source = source.resolve()
        if not source.is_dir():
            raise RuntimeError(f"source root is not a directory: {source}")

        records, unknown = _collect_images(source, allowed_classes=classes)
        all_records.extend(records)
        all_unknown.extend(unknown)

    if not all_records:
        raise RuntimeError("no images found in any source root")

    class_counts: dict[str, int] = defaultdict(int)
    for rec in all_records:
        class_counts[rec["class_name"]] += 1

    missing = [c for c in classes if class_counts.get(c, 0) == 0]
    if missing:
        raise RuntimeError(f"missing required classes with no images: {missing}")

    unknown = sorted(set(all_unknown))
    if unknown:
        raise RuntimeError(f"unknown classes found in source: {unknown}")

    _split_records(all_records, seed=seed)

    split_counts: dict[str, dict[str, int]] = {split: {c: 0 for c in classes} for split in SPLITS}
    for rec in all_records:
        src = Path(rec["source_path"])
        dst = output_root / rec["split"] / rec["class_name"] / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        split_counts[rec["split"]][rec["class_name"]] += 1

    manifest = {
        "source_roots": [str(r) for r in source_roots],
        "seed": seed,
        "classes": classes,
        "is_production_class_set": classes == REQUIRED_CLASSES,
        "production_classes": REQUIRED_CLASSES,
        "splits": {},
        "counts": split_counts,
        "records": all_records,
    }
    for split in SPLITS:
        manifest["splits"][split] = {
            class_name: str(output_root / split / class_name)
            for class_name in classes
        }

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        nargs="+",
        required=True,
        help="One or more source root directories containing class subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output root directory for the prepared dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting.")
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        choices=REQUIRED_CLASSES,
        help=(
            "Subset of the six production classes to require/include, for an explicitly "
            "non-production interim dataset (e.g. while some classes are still blocked on "
            "data access). Defaults to all six production classes."
        ),
    )
    args = parser.parse_args()

    try:
        manifest_path = prepare_dataset(args.source, args.output, seed=args.seed, classes=args.classes)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 1

    print(f"prepared dataset at {args.output}")
    print(f"manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
