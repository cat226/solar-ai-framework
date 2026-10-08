#!/usr/bin/env python3
"""One-off scan: classify every panel-containing image in the raw
Mendeley release by filename-origin category (native numeric ID vs
a/b/OIP/r-prefixed), to quantify a confirmed provenance contamination
(scraped/watermarked third-party images mixed into the release under
non-numeric filenames - see the ingestion report for full evidence)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training.detection.prepare_mendeley_panel_dataset import audit_raw_dataset

raw_root = Path("E:/Solar AI Project Data/other_sources/domain_adaptation_v1/mendeley_raw")
audit = audit_raw_dataset(raw_root)

_NUMERIC_RE = re.compile(r"^\d+$")

by_category: dict[str, list[str]] = {"numeric_native": [], "suspect_prefixed": []}
prefix_counts: dict[str, int] = {}

for rec in audit["images_with_panel_box"]:
    stem = rec["stem"]
    if _NUMERIC_RE.match(stem):
        by_category["numeric_native"].append(stem)
    else:
        by_category["suspect_prefixed"].append(stem)
        prefix = re.match(r"^[A-Za-z]+", stem)
        key = prefix.group(0) if prefix else "other"
        prefix_counts[key] = prefix_counts.get(key, 0) + 1

print(f"Total panel-containing images: {len(audit['images_with_panel_box'])}")
print(f"  numeric_native (confirmed-genuine filename pattern): {len(by_category['numeric_native'])}")
print(f"  suspect_prefixed (a/b/OIP/r/etc.): {len(by_category['suspect_prefixed'])}")
print(f"  suspect prefix breakdown: {prefix_counts}")

# Recompute panel box totals for each category.
stem_to_boxcount = {r["stem"]: r["panel_box_count"] for r in audit["images_with_panel_box"]}
numeric_boxes = sum(stem_to_boxcount[s] for s in by_category["numeric_native"])
suspect_boxes = sum(stem_to_boxcount[s] for s in by_category["suspect_prefixed"])
print(f"  panel boxes in numeric_native images: {numeric_boxes}")
print(f"  panel boxes in suspect_prefixed images: {suspect_boxes}")

# Per-split breakdown of the numeric_native (clean) subset.
stem_to_split = {r["stem"]: r["original_split"] for r in audit["images_with_panel_box"]}
from collections import Counter
print(f"  numeric_native by original split: {dict(Counter(stem_to_split[s] for s in by_category['numeric_native']))}")
print(f"  suspect_prefixed by original split: {dict(Counter(stem_to_split[s] for s in by_category['suspect_prefixed']))}")

import json
out = Path("E:/Solar AI Project Data/other_sources/domain_adaptation_v1/reports/mendeley_provenance_scan.json")
out.write_text(json.dumps({
    "numeric_native_stems": sorted(by_category["numeric_native"]),
    "suspect_prefixed_stems": sorted(by_category["suspect_prefixed"]),
    "suspect_prefix_counts": prefix_counts,
    "numeric_native_panel_boxes": numeric_boxes,
    "suspect_prefixed_panel_boxes": suspect_boxes,
}, indent=2))
print(f"Wrote: {out}")
