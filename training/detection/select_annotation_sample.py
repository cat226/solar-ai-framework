#!/usr/bin/env python3
"""training/detection/select_annotation_sample.py — Deterministic,
diversity-aware sample selection for a human bounding-box annotation
campaign, drawn from SolNET's already-licensed (CC BY 4.0) Clean/Dusty
close-up RGB images (see docs/ML_DOMAIN_REMEDIATION.md's Phase 1 audit
for why this pool, and only this pool, is eligible: PVMD's "Hotspot"
folder is thermal-infrared, not RGB, and is never included here).

What this script does NOT do
-----------------------------
It never produces a bounding box. Clustering/feature-based grouping here
is used ONLY to pick a visually diverse *sample of images* for a human to
annotate next - never to guess where a panel is. That would be exactly
the "AI-generated labels as ground truth" this project's standing policy
prohibits (see training/detection/CLOSEUP_ANNOTATION_TEMPLATE.md).

Method (fully deterministic - no RNG, no ML model, no GPU)
------------------------------------------------------------
1. Inventory every image under the SolNET Clean/Dusty prepared tree:
   path, SHA-256, dimensions, format.
2. Compute simple, transparent, classical-CV visual features per image:
   aspect ratio, mean brightness, brightness contrast (std), mean
   saturation, and edge density (a coarse gradient-magnitude fraction,
   a rough proxy for scene complexity such as multiple panels/grid
   lines vs a single smooth panel). None of these describe *where* a
   panel is - only coarse whole-image visual character, which is a
   legitimate, non-label-generating use of image statistics.
3. Collapse near-duplicates (dHash, reusing the same already-audited
   training.evaluation.common.dhash/hamming_distance technique
   training/evaluation/leakage_audit.py uses) to one representative each,
   so the 200-image sample is not accidentally padded with burst-shot
   duplicates.
4. Stratify the deduplicated pool by (source_class, brightness tercile,
   saturation tercile, edge-density tercile) - bin edges are computed
   from the actual data (33rd/66th percentiles), not hand-picked.
5. Allocate the 200-image budget across strata proportional to stratum
   size (largest-remainder method), with a floor of 1 image for every
   non-empty stratum, so rare visual profiles are still represented
   rather than drowned out by common ones.
6. Within a stratum, select deterministically by sorting candidates on
   their own SHA-256 hex string and taking the first N - reproducible
   without depending on any particular RNG implementation/seed handling.

Every selected image's CSV row records which stratum it came from and
why, in place of "an AI decided this image was interesting."
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from PIL import Image

from training.evaluation.common import dhash, hamming_distance

SOURCE_CLASSES = ["Clean", "Dusty"]  # PVMD "Hotspot" is thermal-infrared - excluded, not RGB.
SPLITS = ["train", "val", "test"]
_NEAR_DUP_THRESHOLD = 5  # matches prepare_closeup_dataset.py / leakage_audit.py


_FEATURE_THUMBNAIL_SIZE = (200, 200)  # source images run up to ~12MP phone photos;
# these are coarse whole-image statistics (brightness/saturation/edge density),
# not per-pixel measurements, so downsampling first is both far faster and a
# more appropriate match to what's actually being measured (global visual
# character, not fine detail or JPEG-noise-level texture).


def _compute_visual_features(rgb_image: Image.Image, width: int, height: int) -> dict[str, Any]:
    """Pure function over an already-decoded PIL RGB image - kept separate
    from I/O so it stays directly unit-testable without touching disk."""
    features: dict[str, Any] = {"dhash": dhash(rgb_image)}
    thumb = rgb_image.resize(_FEATURE_THUMBNAIL_SIZE, Image.BILINEAR)
    arr = np.asarray(thumb, dtype=np.float32) / 255.0

    gray = arr.mean(axis=-1)
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    sat = np.where(maxc > 0, (maxc - minc) / np.maximum(maxc, 1e-6), 0.0)

    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_density = float(((gx > 0.05).mean() + (gy > 0.05).mean()) / 2)

    features["aspect_ratio"] = width / height
    features["mean_brightness"] = float(gray.mean())
    features["std_brightness"] = float(gray.std())
    features["mean_saturation"] = float(sat.mean())
    features["edge_density"] = edge_density
    return features


def inventory(prepared_root: Path, with_features: bool = True) -> list[dict[str, Any]]:
    """Scan the SolNET Clean/Dusty prepared tree, opening each file at most
    twice (a PIL-recommended verify() pass, matching the same
    decompression-bomb-safe pattern used throughout this project - see
    utils/image_utils.py::load_pil_image - followed by exactly one real
    read that computes the SHA-256 from the same in-memory bytes and
    extracts dimensions/features from the same decode). Never modifies
    originals. `with_features=False` skips the (slower) visual-feature
    computation, e.g. for a quick inventory-only pass."""
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        for cls in SOURCE_CLASSES:
            class_dir = prepared_root / split / cls
            if not class_dir.is_dir():
                continue
            for p in sorted(class_dir.iterdir()):
                if not p.is_file():
                    continue
                record: dict[str, Any] = {
                    "absolute_path": str(p.resolve()),
                    "filename": p.name,
                    "source_split": split,
                    "source_class": cls,
                    "source_dataset": "SolNET",
                    "license": "CC-BY-4.0",
                    "readable": False,
                }
                try:
                    with Image.open(p) as im:
                        im.verify()
                    data = p.read_bytes()
                    record["sha256"] = hashlib.sha256(data).hexdigest()
                    with Image.open(io.BytesIO(data)) as im:
                        record["width"], record["height"] = im.size  # from header, before any decode
                        record["format"] = im.format
                        if with_features:
                            # Hint libjpeg to decode at a coarser DCT scale
                            # (1/2, 1/4, 1/8) instead of full resolution then
                            # resizing down - a no-op for non-JPEG formats.
                            # These are coarse whole-image statistics, so the
                            # draft-scale decode is not a quality compromise,
                            # only a real speedup on the ~12MP source photos.
                            im.draft("RGB", _FEATURE_THUMBNAIL_SIZE)
                            rgb = im.convert("RGB")
                            record.update(_compute_visual_features(rgb, record["width"], record["height"]))
                    record["readable"] = True
                except Exception as exc:  # noqa: BLE001
                    record["read_error"] = f"{type(exc).__name__}: {exc}"
                records.append(record)
    return records


def _dedupe_near_duplicates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One representative (first encountered, deterministic input order)
    per near-duplicate cluster, computed SEPARATELY WITHIN each
    source_split. Uses the same dHash/Hamming<=5 convention as the rest
    of this project.

    Deduplicating per-split (rather than globally) is deliberate: this
    project's original train/val/test boundary for SolNET was already
    leakage-audited in an earlier phase (Phase 6A/6B) - re-litigating
    cross-split near-duplicates is out of scope here. What matters for
    *this* selection step is not letting one split's images be
    systematically excluded as "duplicates" of another split's images
    processed earlier - a global, split-order-dependent dedup pass would
    silently shrink the smallest split's (test's) candidate pool just
    because train is iterated first, which showed up as a real, measured
    problem (8/200 test-origin images) before this fix."""
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_split[r["source_split"]].append(r)

    kept: list[dict[str, Any]] = []
    for split_records in by_split.values():
        kept_hashes: list[int] = []
        for r in split_records:
            h = r["dhash"]
            if any(hamming_distance(h, kh) <= _NEAR_DUP_THRESHOLD for kh in kept_hashes):
                r["excluded_as_near_duplicate"] = True
                continue
            kept.append(r)
            kept_hashes.append(h)
    return kept


def _tercile_bin(value: float, edges: tuple[float, float]) -> str:
    lo, hi = edges
    if value <= lo:
        return "low"
    if value <= hi:
        return "med"
    return "high"


def _stratify(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature in ("mean_brightness", "mean_saturation", "edge_density"):
        values = sorted(r[feature] for r in records)
        n = len(values)
        edges = (values[n // 3], values[2 * n // 3])
        for r in records:
            r.setdefault("_bins", {})[feature] = _tercile_bin(r[feature], edges)

    for r in records:
        key = (
            f"class={r['source_class']}"
            f"|brightness={r['_bins']['mean_brightness']}"
            f"|saturation={r['_bins']['mean_saturation']}"
            f"|edge_density={r['_bins']['edge_density']}"
        )
        r["stratum"] = key
        strata[key].append(r)
    return strata


def _allocate_budget(strata: dict[str, list[dict[str, Any]]], total_budget: int) -> dict[str, int]:
    """Largest-remainder apportionment, floor of 1 per non-empty stratum
    (unless that would exceed the budget, in which case as many strata
    as fit each get 1 and the rest get 0 - reported, never silently
    dropped)."""
    n_strata = len(strata)
    total_pool = sum(len(v) for v in strata.values())
    if n_strata >= total_budget:
        # More diversity groups than budget: give one slot each to the
        # `total_budget` largest strata (deterministic tie-break: stratum
        # name, ascending) - documented, not hidden.
        ordered = sorted(strata.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        return {k: (1 if i < total_budget else 0) for i, (k, _v) in enumerate(ordered)}

    raw = {k: total_budget * len(v) / total_pool for k, v in strata.items()}
    floors = {k: max(1, int(np.floor(v))) for k, v in raw.items()}
    # Cap each floor at the stratum's own size (can't allocate more than exists).
    floors = {k: min(floors[k], len(strata[k])) for k in floors}
    allocated = sum(floors.values())
    remainder = total_budget - allocated
    # Distribute remaining slots by largest fractional part, skipping
    # strata already exhausted.
    fractional_order = sorted(
        strata.keys(), key=lambda k: (raw[k] - int(np.floor(raw[k]))), reverse=True
    )
    i = 0
    guard = 0
    while remainder > 0 and guard < 10 * len(strata):
        k = fractional_order[i % len(fractional_order)]
        if floors[k] < len(strata[k]):
            floors[k] += 1
            remainder -= 1
        i += 1
        guard += 1
    return floors


def _allocate_split_budgets(deduped: list[dict[str, Any]], total_budget: int) -> dict[str, int]:
    """Top-level allocation: the 200-image budget is first split
    proportionally across source_split (train/val/test), so annotation
    effort - and the eventual held-out ground-level test set Phase 6
    will draw from this same pool - isn't accidentally starved of
    val/test-origin images. Visual/class diversity (the fine-grained
    stratification) is applied WITHIN each split's own sub-budget
    afterward, kept as a separate concern for clarity."""
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in deduped:
        by_split[r["source_split"]].append(r)
    # Reuse the same largest-remainder logic, treating each split as a
    # "stratum" of its own for this top-level pass.
    return _allocate_budget(by_split, total_budget)


def select_sample(
    prepared_root: Path, total_budget: int = 200
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Returns (full_inventory, selected_records, selection_report)."""
    full_inventory = inventory(prepared_root, with_features=True)
    readable = [r for r in full_inventory if r["readable"]]

    deduped = _dedupe_near_duplicates(readable)

    split_budgets = _allocate_split_budgets(deduped, total_budget)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in deduped:
        by_split[r["source_split"]].append(r)

    selected: list[dict[str, Any]] = []
    all_strata: dict[str, list[dict[str, Any]]] = {}
    all_allocations: dict[str, int] = {}
    for split_name, split_budget in split_budgets.items():
        split_pool = by_split[split_name]
        strata = _stratify(split_pool)
        budget = _allocate_budget(strata, split_budget)
        for stratum_key, quota in budget.items():
            full_key = f"split={split_name}|{stratum_key}"
            all_strata[full_key] = strata[stratum_key]
            all_allocations[full_key] = quota
            candidates = sorted(strata[stratum_key], key=lambda r: r["sha256"])
            chosen = candidates[:quota]
            for r in chosen:
                r["stratum"] = full_key
                r["selection_reason"] = (
                    f"Representative of stratum '{full_key}' "
                    f"({len(strata[stratum_key])} candidate image(s) in this visual "
                    f"profile within the {split_name} split; {quota} selected here) - "
                    f"chosen deterministically by SHA-256 order, not visual judgment "
                    f"of correctness. Split-level budget for '{split_name}' "
                    f"({split_budget}/{total_budget}) was fixed before this stratification, "
                    f"proportional to its share of the deduplicated pool, so val/test "
                    f"representation is guaranteed rather than left to chance."
                )
            selected.extend(chosen)

    report = {
        "total_source_images": len(full_inventory),
        "unreadable_images": len(full_inventory) - len(readable),
        "readable_images": len(readable),
        "near_duplicates_collapsed": len(readable) - len(deduped),
        "deduplicated_pool": len(deduped),
        "split_budgets": split_budgets,
        "num_strata": len(all_strata),
        "budget_requested": total_budget,
        "budget_allocated": len(selected),
        "strata_sizes": {k: len(v) for k, v in all_strata.items()},
        "strata_allocations": all_allocations,
        "selected_count": len(selected),
        "selected_by_source_class": {
            cls: sum(1 for r in selected if r["source_class"] == cls) for cls in SOURCE_CLASSES
        },
        "selected_by_source_split": {
            split: sum(1 for r in selected if r["source_split"] == split) for split in SPLITS
        },
    }
    return full_inventory, selected, report


_CSV_FIELDS = [
    "filename", "absolute_path", "sha256", "width", "height", "format",
    "source_dataset", "source_class", "source_split", "license",
    "aspect_ratio", "mean_brightness", "std_brightness", "mean_saturation",
    "edge_density", "stratum", "selection_reason",
]


def write_inventory_csv(records: list[dict[str, Any]], out_path: Path) -> None:
    fields = [
        "filename", "absolute_path", "sha256", "width", "height", "format",
        "source_dataset", "source_class", "source_split", "license",
        "readable", "read_error",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def write_selection_csv(records: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(records, key=lambda r: r["sha256"]):
            writer.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared-root", type=Path, default=Path("E:/Solar AI Training Images/prepared"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=200)
    args = parser.parse_args()

    full_inventory, selected, report = select_sample(args.prepared_root, args.budget)

    inv_path = args.output_dir / "source_manifest_1348.csv"
    sel_path = args.output_dir / "selected_200.csv"
    report_path = args.output_dir / "selection_report.json"

    write_inventory_csv(full_inventory, inv_path)
    write_selection_csv(selected, sel_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print(f"Inventoried {report['total_source_images']} images "
          f"({report['readable_images']} readable, {report['unreadable_images']} unreadable).")
    print(f"Collapsed {report['near_duplicates_collapsed']} near-duplicates -> "
          f"{report['deduplicated_pool']} candidate pool across {report['num_strata']} strata.")
    print(f"Selected {report['selected_count']} images "
          f"(class breakdown: {report['selected_by_source_class']}, "
          f"split breakdown: {report['selected_by_source_split']}).")
    print(f"Wrote: {inv_path}")
    print(f"Wrote: {sel_path}")
    print(f"Wrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
