"""Prepare the BDAPPV IGN dataset for YOLO detection training.

Reads local parquet shards (downloaded from the Hugging Face mirror of the
BDAPPV dataset, IGN config only - see PROVENANCE_VERIFICATION.md), converts
each row's binary segmentation mask into per-instance YOLO-format bounding
boxes via connected-component analysis, and writes:

    OUTPUT_ROOT/
        train/images/<id>.png   train/labels/<id>.txt
        val/images/<id>.png     val/labels/<id>.txt
        test/images/<id>.png    test/labels/<id>.txt
        manifest.json

A single class is used throughout ("solar panel", class id 0) - this
production pipeline's detector localizes panels; fault classification is a
separate downstream stage (MobileNetV2). Rows with has_mask=False are
genuine validated negative samples per the dataset's own documentation
(not missing annotations) and are written with an empty label file, which
is the standard YOLO convention for a background image with no objects.

Split assignment uses the dataset's own `split` column (train/val/test),
NOT a fresh re-shuffle. This is a deliberate decision, not an oversight:
the BDAPPV authors' README explicitly states the split uses a spatial
holdout by French department specifically to prevent geographic leakage,
and instructs "do not re-split to ensure comparability with published
results." Re-deriving our own split would both discard a more rigorous
leakage-prevention scheme than an ad hoc reshuffle could provide, and
contradict the source's explicit guidance.

This script does not download anything - it operates only on already-local
parquet files, matching this project's data-handling policy.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy import ndimage

CLASS_ID = 0
CLASS_NAME = "solar panel"
SPLITS = ["train", "val", "test"]
# The parquet 'split' column uses "validation", not "val" - map to our directory name.
_SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mask_to_yolo_boxes(mask_bytes: bytes) -> tuple[list[tuple[float, float, float, float]], dict[str, Any]]:
    """Connected-component-label a binary mask and return per-instance YOLO
    boxes (x_center, y_center, w, h, all normalized to [0,1]) plus audit info.
    """
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
    w, h = mask_img.size
    arr = np.array(mask_img)
    uniq = set(np.unique(arr).tolist())
    labeled, n_components = ndimage.label(arr > 0)

    boxes: list[tuple[float, float, float, float]] = []
    zero_area = 0
    for comp_id in range(1, n_components + 1):
        ys, xs = np.where(labeled == comp_id)
        if len(xs) == 0:
            zero_area += 1
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        if x2 <= x1 or y2 <= y1:
            zero_area += 1
            continue
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        boxes.append((cx, cy, bw, bh))

    audit = {
        "width": w,
        "height": h,
        "non_binary_mask": not uniq.issubset({0, 255}),
        "n_components_found": n_components,
        "zero_area_components": zero_area,
    }
    return boxes, audit


def prepare_dataset(shard_dir: Path, output_root: Path) -> Path:
    """Convert all parquet shards under shard_dir into a YOLO-format dataset.

    Returns the path to the written manifest.json.
    """
    shard_paths = sorted(shard_dir.glob("*.parquet"))
    if not shard_paths:
        raise RuntimeError(f"no parquet shards found under {shard_dir}")

    for split in SPLITS:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

    seen_image_hashes: dict[str, str] = {}  # sha256 -> "split/id" first seen at
    records: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, int]] = {s: {"images": 0, "positive": 0, "negative": 0, "instances": 0} for s in SPLITS}

    for shard_path in shard_paths:
        print(f"processing {shard_path.name} ...", file=sys.stderr)
        table = pq.read_table(shard_path)
        df = table.to_pandas()
        for _, row in df.iterrows():
            raw_split = row["split"]
            split = _SPLIT_MAP.get(raw_split)
            if split is None:
                anomalies.append({"id": row["identifiant"], "issue": f"unknown split value: {raw_split!r}"})
                continue

            image_id = row["identifiant"]
            img_bytes = row["image"]["bytes"]
            img_sha = _sha256_bytes(img_bytes)

            if img_sha in seen_image_hashes:
                anomalies.append({
                    "id": image_id, "issue": "duplicate image (SHA-256)",
                    "duplicate_of": seen_image_hashes[img_sha],
                })
                continue
            seen_image_hashes[img_sha] = f"{split}/{image_id}"

            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                anomalies.append({"id": image_id, "issue": f"image decode failed: {exc}"})
                continue
            if img.size != (400, 400):
                anomalies.append({"id": image_id, "issue": f"unexpected image size: {img.size}"})

            boxes: list[tuple[float, float, float, float]] = []
            if bool(row["has_mask"]):
                mask_bytes = row["mask"]["bytes"]
                if mask_bytes is None:
                    anomalies.append({"id": image_id, "issue": "has_mask=True but mask bytes are None"})
                else:
                    try:
                        boxes, mask_audit = _mask_to_yolo_boxes(mask_bytes)
                    except Exception as exc:  # noqa: BLE001
                        anomalies.append({"id": image_id, "issue": f"mask decode failed: {exc}"})
                        mask_audit = {}
                    if mask_audit.get("non_binary_mask"):
                        anomalies.append({"id": image_id, "issue": "mask has non-binary pixel values"})
                    if mask_audit.get("zero_area_components"):
                        anomalies.append({
                            "id": image_id,
                            "issue": f"{mask_audit['zero_area_components']} zero-area component(s) discarded",
                        })
                    if not boxes:
                        anomalies.append({"id": image_id, "issue": "has_mask=True but zero usable boxes extracted"})

            img_path = output_root / split / "images" / f"{image_id}.png"
            label_path = output_root / split / "labels" / f"{image_id}.txt"
            img.save(img_path)
            with label_path.open("w") as f:
                for cx, cy, bw, bh in boxes:
                    f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            split_counts[split]["images"] += 1
            split_counts[split]["instances"] += len(boxes)
            if boxes:
                split_counts[split]["positive"] += 1
            else:
                split_counts[split]["negative"] += 1

            records.append({
                "id": image_id,
                "split": split,
                "sha256": img_sha,
                "num_instances": len(boxes),
                "has_mask": bool(row["has_mask"]),
                "image_path": str(img_path),
                "label_path": str(label_path),
                "source_shard": shard_path.name,
            })

    manifest = {
        "source": "BDAPPV IGN config (Hugging Face mirror of Zenodo 10.5281/zenodo.7358126)",
        "class_names": [CLASS_NAME],
        "split_policy": "dataset's own geographic (department-level) split column - not re-shuffled",
        "counts": split_counts,
        "total_records": len(records),
        "total_anomalies": len(anomalies),
        "anomalies": anomalies,
        "records": records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True, help="Directory containing the downloaded IGN parquet shards.")
    parser.add_argument("--output", type=Path, required=True, help="Output root for the YOLO-format dataset.")
    args = parser.parse_args()

    manifest_path = prepare_dataset(args.shard_dir, args.output)
    print(f"manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
