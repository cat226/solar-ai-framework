#!/usr/bin/env python3
"""training/detection/prepare_closeup_dataset.py — Reusable ingestion
pipeline for a close-up/ground-level solar-panel YOLO detection dataset,
built so the project can immediately consume a legitimate dataset the
moment one is acquired (see docs/ML_DOMAIN_REMEDIATION.md, Phase 3/4).

As of this writing NO dataset has been fed through this script for
training - Phase 6C's and this phase's own dataset search found no
ACCEPT-tier close-up/ground-level RGB bounding-box dataset with acceptable
licensing and provenance. This script exists so that gap is an acquisition
problem, not also an engineering problem, once a legitimate source appears.

Expected input layout::

    SOURCE_DIR/
        images/<id>.<jpg|jpeg|png|webp>
        labels/<id>.txt            # YOLO format, class 0 ("solar_panel") only.
                                    # An empty file is a legitimate negative
                                    # (no panel in frame) - not "missing".
        provenance.json            # {"<id>": {"source_url": ..., "license": ...,
                                    #            "rights_holder": ...,
                                    #            "group_key": ... (optional)}}

Every image MUST have a provenance.json entry with a license from
_ALLOWED_LICENSES and a matching (possibly empty) label file - anything
else is rejected and reported, never silently dropped or silently
accepted. This is a hard gate, not a warning, because "dataset merely
found on Kaggle/Roboflow" and "license claimed by a re-uploader" are
exactly the failure modes this project has already been burned by once
(see training/classification/DATASET_SOURCES.md's Bird-Drop correction)
and was explicitly told never to repeat.

Guarantees this script provides:
- Deterministic train/val/test split from a fixed seed (default 20260905,
  the date this script was written, overridable via --seed) - re-running
  with the same inputs and seed reproduces byte-identical split assignment.
- Exact-duplicate detection via SHA-256 (a true duplicate is kept once,
  reported, and never doubly counted).
- Near-duplicate detection via difference-hash (reusing
  training.evaluation.common.dhash/hamming_distance - the same, already
  audited technique training/evaluation/leakage_audit.py uses), clustered
  with a union-find so a whole near-duplicate cluster (and any explicit
  provenance "group_key", e.g. images from the same capture session) is
  assigned to exactly one split - no near-duplicate/same-scene leakage
  across train/val/test.
- A dataset-level content hash (SHA-256 of the sorted list of every
  included image's own SHA-256), so two runs can be compared for exact
  content equality without diffing every file.
- Single detection class enforced: "solar_panel" (class id 0) - matches
  the existing v1 YOLO taxonomy exactly (see
  training/detection/train_yolo.py). Any other class id in a label file
  is rejected, never silently remapped.

Never fabricates a label, a box, or a license. Never downloads anything -
operates only on already-local files, matching this project's existing
data-handling policy (training/detection/prepare_dataset.py).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image

from training.evaluation.common import dhash, hamming_distance, sha256_file

CLASS_ID = 0
CLASS_NAME = "solar_panel"
SPLITS = ["train", "val", "test"]
_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Licenses this project has determined actually permit ML training /
# derivative-model use (see docs/ML_HARDENING_PHASE6C.md Task 2, criterion
# 4). "explicit-written-grant" covers a case where the rights holder gave
# direct written permission outside a standard license template - the
# provenance record's "notes" field must then describe that grant; this
# script cannot verify a written grant's authenticity, only that one was
# declared, so any use of this category should be corroborated by a human
# before the dataset gate (Phase 10) is passed.
_ALLOWED_LICENSES = {
    "cc0", "cc0-1.0", "public-domain",
    "cc-by", "cc-by-3.0", "cc-by-4.0",
    "cc-by-sa", "cc-by-sa-3.0", "cc-by-sa-4.0",
    "mit",
    "explicit-written-grant",
}
_NEAR_DUP_THRESHOLD = 5  # Hamming distance on a 64-bit dHash; matches leakage_audit.py's default.


def _normalize_license(raw: str) -> str:
    return raw.strip().lower().replace("_", "-")


def _load_provenance(source_dir: Path) -> dict[str, dict[str, Any]]:
    prov_path = source_dir / "provenance.json"
    if not prov_path.is_file():
        raise FileNotFoundError(
            f"No provenance.json found at {prov_path}. Every image in a "
            "training dataset must have a declared source/license - this "
            "script refuses to guess or default one (see "
            "docs/ML_HARDENING_PHASE6C.md's licensing discipline)."
        )
    return json.loads(prov_path.read_text())


def _parse_yolo_label_file(path: Path) -> tuple[list[tuple[float, float, float, float]], list[str]]:
    """Returns (boxes, errors). An empty file (no lines) is valid - it means
    a legitimate negative image with zero panels, not a parsing failure."""
    boxes: list[tuple[float, float, float, float]] = []
    errors: list[str] = []
    text = path.read_text().strip()
    if not text:
        return boxes, errors
    for lineno, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {lineno}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"line {lineno}: could not parse fields as int/float")
            continue
        if cls_id != CLASS_ID:
            errors.append(
                f"line {lineno}: class id {cls_id} is not {CLASS_ID} ('{CLASS_NAME}') - "
                "this dataset must not introduce new detection classes "
                "(Clean/Dusty/Hotspot are MobileNet classification labels only)."
            )
            continue
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)):
            errors.append(f"line {lineno}: box coordinates must be normalized to [0,1]: {(cx, cy, bw, bh)}")
            continue
        if bw <= 0.0 or bh <= 0.0:
            errors.append(f"line {lineno}: degenerate zero/negative-area box: w={bw}, h={bh}")
            continue
        boxes.append((cx, cy, bw, bh))
    return boxes, errors


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self._parent = {item: item for item in items}

    def find(self, x: str) -> str:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def _assign_group_split(group_id: str, seed: int, ratios: tuple[float, float, float]) -> str:
    """Deterministic split assignment: hash(seed, group_id) -> [0, 100),
    bucketed by cumulative ratio. Same seed + same group_id always maps to
    the same split, independent of processing order."""
    import hashlib
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 10_000 / 10_000.0
    train_r, val_r, _test_r = ratios
    if bucket < train_r:
        return "train"
    if bucket < train_r + val_r:
        return "val"
    return "test"


def prepare_closeup_dataset(
    source_dir: Path,
    output_root: Path,
    seed: int = 20260905,
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Path:
    if abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0, got {split_ratios}")

    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Expected an images/ directory at {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Expected a labels/ directory at {labels_dir}")

    provenance = _load_provenance(source_dir)

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in _SUPPORTED_EXTENSIONS)

    rejected: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    seen_sha256: dict[str, str] = {}  # sha256 -> id first seen at

    for img_path in image_paths:
        image_id = img_path.stem
        entry = provenance.get(image_id)
        if entry is None:
            rejected.append({"id": image_id, "reason": "no provenance.json entry"})
            continue
        license_raw = entry.get("license", "")
        license_norm = _normalize_license(license_raw)
        if license_norm not in _ALLOWED_LICENSES:
            rejected.append({
                "id": image_id,
                "reason": f"license '{license_raw}' is not in the allowed set {sorted(_ALLOWED_LICENSES)}",
            })
            continue
        if not entry.get("source_url") and not entry.get("rights_holder"):
            rejected.append({"id": image_id, "reason": "provenance entry has neither source_url nor rights_holder"})
            continue

        label_path = labels_dir / f"{image_id}.txt"
        if not label_path.is_file():
            rejected.append({"id": image_id, "reason": "no matching label file (use an empty file for a genuine negative)"})
            continue
        boxes, label_errors = _parse_yolo_label_file(label_path)
        if label_errors:
            rejected.append({"id": image_id, "reason": f"invalid label file: {'; '.join(label_errors)}"})
            continue

        try:
            img_sha = sha256_file(img_path)
        except OSError as exc:
            rejected.append({"id": image_id, "reason": f"could not read image file: {exc}"})
            continue

        if img_sha in seen_sha256:
            rejected.append({
                "id": image_id,
                "reason": f"exact duplicate (SHA-256) of already-accepted image '{seen_sha256[img_sha]}'",
            })
            continue
        seen_sha256[img_sha] = image_id

        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                width, height = im.size
                phash = dhash(im.convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            rejected.append({"id": image_id, "reason": f"image decode failed: {exc}"})
            continue

        accepted.append({
            "id": image_id,
            "image_path": img_path,
            "label_path": label_path,
            "sha256": img_sha,
            "dhash": phash,
            "width": width,
            "height": height,
            "num_boxes": len(boxes),
            "boxes": boxes,
            "license": license_norm,
            "source_url": entry.get("source_url"),
            "rights_holder": entry.get("rights_holder"),
            "group_key": entry.get("group_key"),
        })

    # --- Near-duplicate clustering (union-find over dHash Hamming distance,
    # merged further by any shared explicit group_key) ---
    uf = _UnionFind([rec["id"] for rec in accepted])
    for i, a in enumerate(accepted):
        for b in accepted[i + 1:]:
            if hamming_distance(a["dhash"], b["dhash"]) <= _NEAR_DUP_THRESHOLD:
                uf.union(a["id"], b["id"])
    group_key_to_ids: dict[str, list[str]] = defaultdict(list)
    for rec in accepted:
        if rec["group_key"]:
            group_key_to_ids[rec["group_key"]].append(rec["id"])
    for ids in group_key_to_ids.values():
        for other in ids[1:]:
            uf.union(ids[0], other)

    cluster_of: dict[str, str] = {rec["id"]: uf.find(rec["id"]) for rec in accepted}
    clusters: dict[str, list[str]] = defaultdict(list)
    for image_id, cluster_id in cluster_of.items():
        clusters[cluster_id].append(image_id)
    near_duplicate_clusters = {cid: ids for cid, ids in clusters.items() if len(ids) > 1}

    # --- Deterministic, cluster-aware split ---
    split_of_cluster = {cid: _assign_group_split(cid, seed, split_ratios) for cid in clusters}

    for split in SPLITS:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, dict[str, int]] = {s: {"images": 0, "positive": 0, "negative": 0, "instances": 0} for s in SPLITS}
    records: list[dict[str, Any]] = []
    for rec in accepted:
        split = split_of_cluster[cluster_of[rec["id"]]]
        dest_img = output_root / split / "images" / rec["image_path"].name
        dest_label = output_root / split / "labels" / f"{rec['id']}.txt"
        shutil.copy2(rec["image_path"], dest_img)
        shutil.copy2(rec["label_path"], dest_label)

        split_counts[split]["images"] += 1
        split_counts[split]["instances"] += rec["num_boxes"]
        split_counts[split][("positive" if rec["num_boxes"] else "negative")] += 1
        records.append({
            "id": rec["id"], "split": split, "sha256": rec["sha256"],
            "width": rec["width"], "height": rec["height"],
            "num_boxes": rec["num_boxes"], "license": rec["license"],
            "source_url": rec["source_url"], "rights_holder": rec["rights_holder"],
            "near_duplicate_cluster": cluster_of[rec["id"]] if cluster_of[rec["id"]] in near_duplicate_clusters else None,
        })

    import hashlib
    dataset_hash = hashlib.sha256("".join(sorted(r["sha256"] for r in records)).encode()).hexdigest()

    manifest = {
        "class_names": [CLASS_NAME],
        "seed": seed,
        "split_ratios": {"train": split_ratios[0], "val": split_ratios[1], "test": split_ratios[2]},
        "split_policy": (
            "Deterministic hash(seed, near-duplicate-cluster-id) bucketing - "
            "a whole near-duplicate cluster (dHash Hamming<=5) and any images "
            "sharing an explicit provenance group_key are always assigned to "
            "the same split, so no same-scene/near-duplicate content leaks "
            "across train/val/test."
        ),
        "dataset_content_hash_sha256": dataset_hash,
        "counts": split_counts,
        "total_accepted": len(records),
        "total_rejected": len(rejected),
        "rejected": rejected,
        "near_duplicate_clusters_found": len(near_duplicate_clusters),
        "near_duplicate_cluster_sizes": {cid: len(ids) for cid, ids in near_duplicate_clusters.items()},
        "records": records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory with images/, labels/, provenance.json.")
    parser.add_argument("--output", type=Path, required=True, help="Output root for the prepared YOLO-format dataset.")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    manifest_path = prepare_closeup_dataset(
        args.source_dir, args.output, seed=args.seed,
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
    )
    manifest = json.loads(manifest_path.read_text())
    print(f"Accepted {manifest['total_accepted']}, rejected {manifest['total_rejected']} images.")
    print(f"Split counts: {manifest['counts']}")
    print(f"Near-duplicate clusters found: {manifest['near_duplicate_clusters_found']}")
    print(f"Dataset content hash: {manifest['dataset_content_hash_sha256']}")
    print(f"manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
