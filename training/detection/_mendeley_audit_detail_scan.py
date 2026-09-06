#!/usr/bin/env python3
"""One-off detail scan used to identify exactly which files in the
Mendeley audit have a malformed line or a high-IoU panel-box pair, for
citing in the ingestion report. Not part of the reusable pipeline -
prepare_mendeley_panel_dataset.py's audit_raw_label_file() already
computes this per-file; this script just prints the specific hits."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training.detection.prepare_mendeley_panel_dataset import audit_raw_label_file

raw_root = Path("E:/SolarAI_Datasets/domain_adaptation_v1/mendeley_raw")

for split in ("train", "val", "test"):
    labels_dir = raw_root / split / "labels"
    for label_path in sorted(labels_dir.glob("*.txt")):
        result = audit_raw_label_file(label_path)
        if result["errors"]:
            print(f"MALFORMED [{split}] {label_path.name}: {result['errors']}")
        if result["high_iou_panel_pairs"]:
            print(f"HIGH-IOU [{split}] {label_path.name}: boxes={result['panel_boxes']} pairs={result['high_iou_panel_pairs']}")
