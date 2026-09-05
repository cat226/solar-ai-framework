#!/usr/bin/env python3
"""training/evaluation/leakage_audit.py — Exact-duplicate and near-duplicate
audit across a prepared dataset's train/val/test splits.

Four things, each clearly labeled and never conflated:

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
4. Classification + clean-subset construction: every near-duplicate
   candidate pair is placed into one of four categories (exact_duplicate /
   highly_likely_near_duplicate / probable_false_positive / uncertain)
   using corroborating signals (matching capture-timestamp filename
   prefix), not the hash distance alone - Phase 6A manually verified both
   a genuine same-photo duplicate (hamming=0, matching timestamp) and a
   hash false positive (hamming=0, no matching timestamp, visually
   confirmed to be two different photos) on this exact dataset, so hash
   distance alone is not treated as proof. A "clean" test-split subset
   (test images with no highly_likely_near_duplicate elsewhere) is derived
   and written out as a separate, explicitly-labeled evaluation artifact -
   the original test split is never modified.

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


def _base_id(name: str) -> str:
    """Strip common suffixes seen in this dataset (e.g. "_2_11zon", "(0)")
    to compare base capture identity across splits. Real photo filenames
    from a phone/camera often share a common timestamp prefix for images
    taken moments apart of the same scene, or re-exported through a resize
    tool - this normalizes away the resize-tool suffix chain so the
    underlying capture identity can be compared.

    Only strips suffix chunks that are actually part of an "_N_11zon"
    resize-tool marker (one or more repetitions), never a bare trailing
    "_<digits>" on its own - this dataset's real filenames are
    "YYYYMMDD_HHMMSS.jpg", and a naive "(_\\d+)?" branch previously matched
    and stripped that trailing "_HHMMSS" segment too, incorrectly treating
    every photo from the same *day* (regardless of time-of-day) as sharing
    one base id. Fixed after a unit test caught it - see
    tests/test_leakage_audit.py."""
    stem = Path(name).stem
    stem = re.sub(r"\(\d+\)$", "", stem)
    stem = re.sub(r"(_\d+_11zon)+$", "", stem)
    return stem


def _near_duplicate_audit(images_by_split: dict[str, list[Path]], threshold: int, max_pairs_per_boundary: int) -> dict:
    hashes: dict[str, list[tuple[Path, int]]] = {}
    for split, paths in images_by_split.items():
        hashes[split] = [(p, dhash(Image.open(p).convert("RGB"))) for p in paths]

    boundaries = list(itertools.combinations(images_by_split.keys(), 2))
    results: dict[str, list[dict]] = {}
    all_pairs: list[dict] = []
    for a, b in boundaries:
        pairs = []
        for pa, ha in hashes[a]:
            for pb, hb in hashes[b]:
                d = hamming_distance(ha, hb)
                if d <= threshold:
                    pair = {
                        "a": f"{a}/{pa.parent.name}/{pa.name}", "b": f"{b}/{pb.parent.name}/{pb.name}",
                        "a_path": str(pa), "b_path": str(pb), "hamming_distance": d,
                    }
                    pairs.append(pair)
        pairs.sort(key=lambda r: r["hamming_distance"])
        all_pairs.extend(pairs)
        results[f"{a}<->{b}"] = [{k: v for k, v in p.items() if k not in ("a_path", "b_path")} for p in pairs[:max_pairs_per_boundary]]
        results[f"{a}<->{b}_total_found"] = len(pairs)  # type: ignore[assignment]

    return {"threshold": threshold, "boundaries": results, "_all_pairs": all_pairs}


def _classify_pairs(all_pairs: list[dict]) -> dict:
    """Classify every near-duplicate candidate pair using a corroborating
    signal (matching capture-timestamp filename base) rather than hash
    distance alone - see module docstring for the manually-verified
    evidence behind this rule.

    - highly_likely_near_duplicate: hamming_distance == 0 AND the two
      filenames share a normalized base id (matching capture timestamp).
    - probable_false_positive: hamming_distance == 0 but the filenames do
      NOT share a base id - matches the manually-verified hash-collision
      case in docs/ML_EVALUATION_v1.0.0.md.
    - uncertain: hamming_distance > 0 (some similarity, not exact hash
      match) - not asserted either way without visual inspection.
    """
    classified = {"highly_likely_near_duplicate": [], "probable_false_positive": [], "uncertain": []}
    for pair in all_pairs:
        a_name = pair["a"].rsplit("/", 1)[-1]
        b_name = pair["b"].rsplit("/", 1)[-1]
        same_base = _base_id(a_name) == _base_id(b_name)
        record = {"a": pair["a"], "b": pair["b"], "hamming_distance": pair["hamming_distance"], "same_filename_base": same_base}
        if pair["hamming_distance"] == 0 and same_base:
            classified["highly_likely_near_duplicate"].append(record)
        elif pair["hamming_distance"] == 0 and not same_base:
            classified["probable_false_positive"].append(record)
        else:
            classified["uncertain"].append(record)
    return {
        "counts": {k: len(v) for k, v in classified.items()},
        "highly_likely_near_duplicate": classified["highly_likely_near_duplicate"],
        "probable_false_positive": classified["probable_false_positive"][:25],
        "uncertain": classified["uncertain"][:25],
    }


def _build_clean_test_subset(images_by_split: dict[str, list[Path]], classified: dict) -> dict:
    """Test-split images with no highly_likely_near_duplicate elsewhere in
    the dataset (train or val). The original test split/dataset files are
    never modified - this is a derived list only."""
    contaminated_test_locations = {
        rec["b"] if rec["b"].startswith("test/") else rec["a"]
        for rec in classified["highly_likely_near_duplicate"]
        if rec["a"].startswith("test/") or rec["b"].startswith("test/")
    }
    all_test = [f"test/{p.parent.name}/{p.name}" for p in images_by_split.get("test", [])]
    clean = [loc for loc in all_test if loc not in contaminated_test_locations]
    return {
        "original_test_count": len(all_test),
        "contaminated_test_count": len(contaminated_test_locations),
        "clean_test_count": len(clean),
        "clean_test_images": clean,
        "excluded_images": sorted(contaminated_test_locations),
    }


def _source_signal_audit(images_by_split: dict[str, list[Path]]) -> dict:
    """Real photo filenames from a phone/camera often share a common prefix
    (timestamp, burst id) for images taken moments apart of the same scene.
    This is a *signal* to inspect manually, never proof of leakage on its
    own - two images can share a prefix pattern coincidentally, and this
    dataset's manifest already records no formal grouping metadata."""
    base_id = _base_id
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

    print("\n[3/4] Filename-base source-signal audit...")
    source_signal = _source_signal_audit(images_by_split)
    print(f"  cross-split shared filename-base groups: {source_signal['cross_split_shared_filename_base_count']}")

    print("\n[4/4] Classifying near-duplicate pairs + building clean test subset...")
    classified = _classify_pairs(near.pop("_all_pairs"))
    print(f"  classification counts: {classified['counts']}")
    clean_subset = _build_clean_test_subset(images_by_split, classified)
    print(f"  clean test subset: {clean_subset['clean_test_count']} of {clean_subset['original_test_count']} "
          f"images (excluded {clean_subset['contaminated_test_count']} with a highly-likely near duplicate elsewhere)")

    report = {
        "data_root": str(args.data_root),
        "splits": args.splits,
        "exact_duplicates": exact,
        "near_duplicates": near,
        "source_signal": source_signal,
        "classification": classified,
        "clean_test_subset": clean_subset,
    }
    out_path = output_dir / "leakage_audit.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
