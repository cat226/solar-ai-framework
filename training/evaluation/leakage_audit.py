#!/usr/bin/env python3
"""training/evaluation/leakage_audit.py — Exact-duplicate and near-duplicate
audit across a prepared dataset's train/val/test splits.

Three checks, each clearly labeled and never conflated:

1. Exact duplicates: SHA-256 collisions across splits (byte-identical files
   presented under different filenames/paths). Expected: 0.
2. Near duplicates: perceptual-hash (dHash) similarity across splits -
   catches recompressed/resized/lightly-cropped copies of the same photo
   that SHA-256 cannot. Reports *candidate* pairs at a stated Hamming-
   distance threshold; does not assert these are proven leakage without
   visual/metadata corroboration.
3. Source/group signal: inspects filenames for shared prefixes (e.g. burst
   photos from the same capture session) crossing splits, since this
   dataset's own manifest records no formal grouping metadata (group=None
   for every record - see training/classification/INTERIM_MODEL_REPORT.md's
   and the mobilenet training-registry entry's own honest caveat about this).

Read-only: never deletes, moves, or relabels anything.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image

from training.evaluation.common import default_output_root, dhash, hamming_distance, sha256_file


def _iter_split_images(data_root: Path, splits: list[str]) -> dict[str, list[Path]]:
    """data_root/<split>/<Class>/<image> layout (classification-style)."""
    out: dict[str, list[Path]] = {}
    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            out[split] = []
            continue
        paths: list[Path] = []
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                paths.extend(sorted(p for p in class_dir.iterdir() if p.is_file()))
        out[split] = paths
    return out


def _exact_duplicate_audit(images_by_split: dict[str, list[Path]]) -> dict:
    hash_to_locations: dict[str, list[str]] = defaultdict(list)
    for split, paths in images_by_split.items():
        for p in paths:
            h = sha256_file(p)
            hash_to_locations[h].append(f"{split}/{p.parent.name}/{p.name}")

    cross_split_collisions = []
    for h, locations in hash_to_locations.items():
        splits_involved = {loc.split("/")[0] for loc in locations}
        if len(splits_involved) > 1:
            cross_split_collisions.append({"sha256": h, "locations": locations})

    return {
        "total_images_hashed": sum(len(v) for v in images_by_split.values()),
        "unique_hashes": len(hash_to_locations),
        "cross_split_collisions": cross_split_collisions,
        "cross_split_collision_count": len(cross_split_collisions),
    }


def _near_duplicate_audit(images_by_split: dict[str, list[Path]], threshold: int, max_pairs_per_boundary: int) -> dict:
    hashes: dict[str, list[tuple[Path, int]]] = {}
    for split, paths in images_by_split.items():
        hashes[split] = [(p, dhash(Image.open(p).convert("RGB"))) for p in paths]

    boundaries = list(itertools.combinations(images_by_split.keys(), 2))
    results: dict[str, list[dict]] = {}
    for a, b in boundaries:
        pairs = []
        for pa, ha in hashes[a]:
            for pb, hb in hashes[b]:
                d = hamming_distance(ha, hb)
                if d <= threshold:
                    pairs.append({"a": f"{a}/{pa.parent.name}/{pa.name}", "b": f"{b}/{pb.parent.name}/{pb.name}", "hamming_distance": d})
        pairs.sort(key=lambda r: r["hamming_distance"])
        results[f"{a}<->{b}"] = pairs[:max_pairs_per_boundary]
        results[f"{a}<->{b}_total_found"] = len(pairs)  # type: ignore[assignment]

    return {"threshold": threshold, "boundaries": results}


def _source_signal_audit(images_by_split: dict[str, list[Path]]) -> dict:
    """Real photo filenames from a phone/camera often share a common prefix
    (timestamp, burst id) for images taken moments apart of the same scene.
    This is a *signal* to inspect manually, never proof of leakage on its
    own - two images can share a prefix pattern coincidentally, and this
    dataset's manifest already records no formal grouping metadata."""
    # Strip common suffixes seen in this dataset (e.g. "_2_11zon", "(0)")
    # to compare base capture identity across splits.
    def base_id(name: str) -> str:
        stem = Path(name).stem
        stem = re.sub(r"\(\d+\)$", "", stem)
        stem = re.sub(r"(_\d+)?(_\d+_11zon)*(_11zon)*$", "", stem)
        return stem

    base_to_locations: dict[str, list[str]] = defaultdict(list)
    for split, paths in images_by_split.items():
        for p in paths:
            base_to_locations[base_id(p.name)].append(f"{split}/{p.parent.name}/{p.name}")

    cross_split_shared_base = [
        {"base_id": base, "locations": locs}
        for base, locs in base_to_locations.items()
        if len({loc.split("/")[0] for loc in locs}) > 1
    ]
    return {
        "cross_split_shared_filename_base_count": len(cross_split_shared_base),
        "examples": cross_split_shared_base[:25],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--dhash-threshold", type=int, default=5, help="Max Hamming distance to flag as a near-duplicate candidate (64-bit hash; 0=identical, <=5 is a conventional 'very similar' cutoff).")
    parser.add_argument("--max-pairs-per-boundary", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or default_output_root()
    output_dir.mkdir(parents=True, exist_ok=True)

    images_by_split = _iter_split_images(args.data_root, args.splits)
    for split, paths in images_by_split.items():
        print(f"{split}: {len(paths)} images")

    print("\n[1/3] Exact-duplicate (SHA-256) audit...")
    exact = _exact_duplicate_audit(images_by_split)
    print(f"  cross-split SHA-256 collisions: {exact['cross_split_collision_count']}")

    print(f"\n[2/3] Near-duplicate (dHash, threshold<={args.dhash_threshold}) audit...")
    near = _near_duplicate_audit(images_by_split, args.dhash_threshold, args.max_pairs_per_boundary)
    for boundary, pairs in near["boundaries"].items():
        if boundary.endswith("_total_found"):
            print(f"  {boundary}: {pairs}")

    print("\n[3/3] Filename-base source-signal audit...")
    source_signal = _source_signal_audit(images_by_split)
    print(f"  cross-split shared filename-base groups: {source_signal['cross_split_shared_filename_base_count']}")

    report = {
        "data_root": str(args.data_root),
        "splits": args.splits,
        "exact_duplicates": exact,
        "near_duplicates": near,
        "source_signal": source_signal,
    }
    out_path = output_dir / "leakage_audit.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
