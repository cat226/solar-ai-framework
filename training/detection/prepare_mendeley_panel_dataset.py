#!/usr/bin/env python3
"""training/detection/prepare_mendeley_panel_dataset.py — Ingests the
Mendeley "Solar panel anomaly dataset" (Xing Yang et al., DOI
10.17632/8x7wjsr94v.1, CC BY 4.0 - see
E:\\SolarAI_Datasets\\acquisition_manifest\\ground_level_bbox_candidates.md
for the full rights/provenance audit) and produces a Solar AI-compatible
single-class (`solar_panel` = 0) YOLO detection dataset.

This is Step A of a two-step pipeline:

  Step A (this script): read the raw 4-class Mendeley release
  (classes: bird_drop, cracked, dusty, panel), audit every label file for
  mechanical validity, keep ONLY images that have at least one real
  `panel`-class box, remap that class to `solar_panel` = 0, and discard
  every bird_drop/cracked/dusty box - those fault-type annotations are
  never used as YOLO detection classes in this project (they would
  belong to a separately-scoped, separately-provenance-checked MobileNet
  taxonomy effort, if ever - not here). Writes an intermediate
  images/+labels/+provenance.json tree.

  Step B (training/detection/prepare_closeup_dataset.py, unmodified,
  reused as-is): consumes Step A's output to perform near-duplicate
  clustering, deterministic group-aware train/val/test splitting, and
  final manifest/provenance generation - exactly the same, already-
  tested pipeline built for the SolNET annotation campaign. Not
  reimplemented here.

Never modifies the raw source directory. Never fabricates a box, a
label, or a license claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image

# The raw dataset's own class list (classes.txt / data.yaml, verified by
# direct inspection - never assumed).
RAW_CLASS_NAMES = {0: "bird_drop", 1: "cracked", 2: "dusty", 3: "panel"}
PANEL_CLASS_ID = 3

TARGET_CLASS_ID = 0
TARGET_CLASS_NAME = "solar_panel"

# CRITICAL PROVENANCE FINDING (see docs/reports/mendeley_ingestion_report.md
# for the full evidence): this release's images use two visually distinct
# filename families. Purely-numeric filenames (e.g. "10.jpg", "1108.jpg")
# match the paper's own stated methodology (SIL-IoT node cameras + mobile
# devices) and were directly, visually confirmed clean on every sample
# checked - no watermark, consistent real-world field context. Filenames
# prefixed with a letter ("a406.jpg", "b1.jpg", "OIP-1.jpg", ...) include
# CONFIRMED third-party stock/scraped content: multiple "a"-prefixed files
# are the exact same watermarked Shutterstock preview image (visible
# "shutterstock" tiling + a printed Image ID), and "OIP"-prefixed files
# use Bing Image Search's own cache-thumbnail naming convention and carry
# a different third-party website's watermark. A Mendeley-level CC BY 4.0
# tag cannot legitimately extend to content the uploader does not own -
# this is exactly the "re-uploader's/platform's license claim does not
# cure an unverified original license" principle this project applies
# consistently. "b"-prefixed files show mixed evidence (one sample
# checked, b1.jpg, showed no red flags) and are NOT exhaustively
# verified - treated as CONDITIONAL/unverified, not promoted into the
# trustworthy default, pending individual review.
_NUMERIC_STEM_RE = re.compile(r"^\d+$")


def provenance_category(stem: str) -> str:
    """Classifies one image's provenance trust level from its filename
    alone (a real, disclosed proxy - not a guess about content, an
    directly observed correlation: every checked numeric-filename image
    was clean, every checked "a"/"OIP" image was confirmed third-party
    stock/scraped content)."""
    if _NUMERIC_STEM_RE.match(stem):
        return "numeric_native"
    if stem.startswith("a") or stem.startswith("OIP"):
        return "confirmed_contaminated"
    return "unverified_prefixed"

# MANUAL SITE-GROUP CORRECTION (confirmed by direct visual review, not by
# dHash alone): Step B's automated near-duplicate clustering (dHash,
# Hamming<=5) split the numeric_native subset into 6 clusters, but visual
# inspection of one representative image per cluster showed that 4 of
# those "clusters" (1583, 1605, 1628, 1650) are actually the SAME physical
# installation/session photographed at different zoom levels - the same
# pole, the same fence, the same road, the same buildings in the
# background - just similar enough in different crops that the dHash
# Hamming distance exceeded the near-dup threshold. Left uncorrected, this
# would leak the same physical site across train/val/test (Step B's own
# stated no-leakage guarantee only covers what its dHash actually
# clusters). These 3 stem lists give the real, human-confirmed site
# grouping for the 80 numeric_native images that made it through Step A's
# provenance filter and Step B's exact-duplicate rejection, and are fed
# into provenance.json's group_key so Step B's union-find merges them
# correctly regardless of dHash distance. Any numeric_native stem NOT
# listed here falls back to dHash-only clustering (group_key=None).
_MANUAL_SITE_GROUPS: dict[str, str] = {}
for _stem in (
    "1108 1109 1110 1111 1113 1115 1116 1117 1118 1119 1120 1121 1122 1123 "
    "1124 1126 1127 1128 1132 1140 1144 1148 1152 1156 1160 1164 1168 1172 "
    "1176 1184 1188 1192 1206 1228 1239 1250 1261 1272 1283 1295 1306 1317"
).split():
    _MANUAL_SITE_GROUPS[_stem] = "mendeley_site_A"
for _stem in (
    "1439 1450 1461 1472 1483 1494 1505 1517 1528 1539 1550 1561 1572 1583 "
    "1594 1605 1616 1628 1639 1650 1661"
).split():
    _MANUAL_SITE_GROUPS[_stem] = "mendeley_site_B"
for _stem in (
    "1203 1204 1205 1207 1208 1209 1210 1211 1212 1213 1214 1215 1216 1218 "
    "1219 1220 1221"
).split():
    _MANUAL_SITE_GROUPS[_stem] = "mendeley_site_C"
del _stem

_SOURCE_LICENSE = "CC-BY-4.0"
_SOURCE_URL = "https://doi.org/10.17632/8x7wjsr94v.1"
_SOURCE_RIGHTS_HOLDER = (
    "Xing Yang et al. - College of Intelligent Manufacturing, Anhui Science "
    "and Technology University; Nanjing Agricultural University"
)

# Sentinel license string for images whose provenance_category is NOT
# numeric_native, used ONLY when a caller explicitly opts into including
# them (allowed_provenance_categories widened beyond the default). This is
# deliberately NOT a real license name (not "CC-BY-4.0", not any string in
# prepare_closeup_dataset.py's _ALLOWED_LICENSES) so it can never be
# silently accepted by that script's production license gate, and never be
# mistaken for a verified permissive license downstream. It exists so a
# personal/non-commercial/experimental run can include these images
# without EVER claiming they are license-cleared - see
# E:\SolarAI_Datasets\domain_adaptation_v1\reports\mendeley_ingestion_report.md
# and the experimental-run addendum in that same reports directory.
_EXPERIMENTAL_UNVERIFIED_LICENSE = "personal-experimental-unverified-provenance"
_UNVERIFIED_RIGHTS_HOLDER_NOTE = (
    "UNKNOWN/UNVERIFIED - this specific image's filename pattern was flagged "
    "as provenance-contaminated or unverified during ingestion audit "
    "(see mendeley_ingestion_report.md); the Mendeley uploader (Xing Yang et "
    "al.) is NOT confirmed to be the rights holder for this specific image "
    "and may not have had the right to license it. Included ONLY in an "
    "explicitly-labelled personal/non-commercial/experimental training pool, "
    "never represented as commercially cleared or redistributable."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """a, b are (cx, cy, w, h) normalized boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def parse_raw_label_line(line: str, lineno: int) -> tuple[tuple[int, float, float, float, float] | None, str | None]:
    """Validates one line of a RAW (4-class) Mendeley label file. Returns
    (parsed_box_or_None, error_or_None). A parsed box is
    (class_id, cx, cy, w, h)."""
    parts = line.split()
    if len(parts) != 5:
        return None, f"line {lineno}: expected 5 fields, got {len(parts)}"
    try:
        cid = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:])
    except ValueError:
        return None, f"line {lineno}: could not parse fields as int/float"
    if cid not in RAW_CLASS_NAMES:
        return None, f"line {lineno}: unknown class id {cid} (expected one of {sorted(RAW_CLASS_NAMES)})"
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return None, f"line {lineno}: center out of [0,1]: ({cx},{cy})"
    if w <= 0.0 or h <= 0.0:
        return None, f"line {lineno}: zero/negative-area box: w={w}, h={h}"
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    tol = 1e-6
    if x1 < -tol or y1 < -tol or x2 > 1 + tol or y2 > 1 + tol:
        return None, f"line {lineno}: box edges outside image: ({x1:.4f},{y1:.4f})-({x2:.4f},{y2:.4f})"
    return (cid, cx, cy, w, h), None


def audit_raw_label_file(label_path: Path, overlap_iou_threshold: float = 0.3) -> dict[str, Any]:
    """Full mechanical audit of one raw 4-class label file: parses every
    line, classifies errors, extracts panel-class boxes specifically, and
    flags any pair of panel boxes with high mutual IoU (a candidate
    duplicate/erroneous double-annotation - reported for human review,
    never auto-resolved)."""
    text = label_path.read_text(encoding="utf-8").strip()
    errors: list[str] = []
    class_counts: dict[int, int] = {cid: 0 for cid in RAW_CLASS_NAMES}
    panel_boxes: list[tuple[float, float, float, float]] = []

    if text:
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            box, error = parse_raw_label_line(line, lineno)
            if error:
                errors.append(error)
                continue
            cid, cx, cy, w, h = box
            class_counts[cid] += 1
            if cid == PANEL_CLASS_ID:
                panel_boxes.append((cx, cy, w, h))

    overlaps: list[dict[str, Any]] = []
    for i in range(len(panel_boxes)):
        for j in range(i + 1, len(panel_boxes)):
            v = _iou(panel_boxes[i], panel_boxes[j])
            if v >= overlap_iou_threshold:
                overlaps.append({"box_i": i, "box_j": j, "iou": round(v, 4)})

    return {
        "filename": label_path.name,
        "errors": errors,
        "class_counts": class_counts,
        "panel_box_count": len(panel_boxes),
        "panel_boxes": panel_boxes,
        "high_iou_panel_pairs": overlaps,
    }


def audit_raw_dataset(raw_root: Path) -> dict[str, Any]:
    """Full-dataset audit across all 3 original splits shipped by
    Mendeley (train/val/test) - read-only, modifies nothing. This is the
    evidence base for the ingestion report's annotation-format and
    malformed-entry sections."""
    per_split: dict[str, dict[str, Any]] = {}
    total_class_counts: dict[int, int] = {cid: 0 for cid in RAW_CLASS_NAMES}
    total_malformed = 0
    total_high_iou_pairs = 0
    images_with_panel: list[dict[str, Any]] = []

    for split in ("train", "val", "test"):
        labels_dir = raw_root / split / "labels"
        images_dir = raw_root / split / "images"
        if not labels_dir.is_dir():
            continue
        split_summary = {
            "total_label_files": 0, "malformed_lines": 0,
            "images_with_panel_box": 0, "panel_box_total": 0,
            "high_iou_panel_pairs": 0,
        }
        for label_path in sorted(labels_dir.glob("*.txt")):
            split_summary["total_label_files"] += 1
            result = audit_raw_label_file(label_path)
            for cid, n in result["class_counts"].items():
                total_class_counts[cid] += n
            total_malformed += len(result["errors"])
            split_summary["malformed_lines"] += len(result["errors"])
            total_high_iou_pairs += len(result["high_iou_panel_pairs"])
            split_summary["high_iou_panel_pairs"] += len(result["high_iou_panel_pairs"])
            if result["panel_box_count"] > 0:
                split_summary["images_with_panel_box"] += 1
                split_summary["panel_box_total"] += result["panel_box_count"]
                img_path = images_dir / f"{label_path.stem}.jpg"
                if not img_path.is_file():
                    # A small number of raw images use other extensions -
                    # find the actual one rather than assuming .jpg.
                    matches = list(images_dir.glob(f"{label_path.stem}.*"))
                    img_path = matches[0] if matches else img_path
                images_with_panel.append({
                    "stem": label_path.stem,
                    "original_split": split,
                    "image_path": str(img_path),
                    "label_path": str(label_path),
                    "panel_box_count": result["panel_box_count"],
                    "high_iou_panel_pairs": result["high_iou_panel_pairs"],
                    "provenance_category": provenance_category(label_path.stem),
                })
        per_split[split] = split_summary

    provenance_counts: dict[str, int] = {}
    provenance_box_counts: dict[str, int] = {}
    for rec in images_with_panel:
        cat = rec["provenance_category"]
        provenance_counts[cat] = provenance_counts.get(cat, 0) + 1
        provenance_box_counts[cat] = provenance_box_counts.get(cat, 0) + rec["panel_box_count"]

    return {
        "per_split": per_split,
        "total_class_counts": {RAW_CLASS_NAMES[k]: v for k, v in total_class_counts.items()},
        "total_malformed_lines": total_malformed,
        "total_high_iou_panel_pairs": total_high_iou_pairs,
        "images_with_panel_box_count": len(images_with_panel),
        "images_with_panel_box": images_with_panel,
        "provenance_category_image_counts": provenance_counts,
        "provenance_category_panel_box_counts": provenance_box_counts,
    }


def convert_to_single_class(
    raw_root: Path,
    output_dir: Path,
    audit: dict[str, Any] | None = None,
    allowed_provenance_categories: "frozenset[str] | set[str] | None" = frozenset({"numeric_native"}),
) -> dict[str, Any]:
    """Step A: writes output_dir/images/, output_dir/labels/,
    output_dir/provenance.json containing ONLY the images that have a
    real panel-class box, with labels remapped to class 0 (solar_panel)
    and every bird_drop/cracked/dusty box dropped. Never modifies
    raw_root. Returns a conversion-statistics dict for the report.

    allowed_provenance_categories restricts conversion to images whose
    filename-derived provenance_category (see provenance_category())
    is in this set. Defaults to {"numeric_native"} only - the confirmed-
    clean subset - because direct visual/forensic inspection confirmed
    that a large share of non-numeric-filename images in this release
    are third-party stock/scraped content wrongly bundled under the
    release's CC BY 4.0 tag (see ingestion report). Pass None to disable
    this filter entirely (NOT recommended for the trustworthy training
    dataset; only for exploratory/diagnostic runs)."""
    if audit is None:
        audit = audit_raw_dataset(raw_root)

    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    provenance: dict[str, dict[str, Any]] = {}
    converted_records: list[dict[str, Any]] = []
    skipped_unreadable: list[str] = []
    skipped_provenance: list[str] = []

    for rec in audit["images_with_panel_box"]:
        cat = rec.get("provenance_category") or provenance_category(rec["stem"])
        if allowed_provenance_categories is not None and cat not in allowed_provenance_categories:
            skipped_provenance.append(rec["stem"])
            continue
        img_src = Path(rec["image_path"])
        label_src = Path(rec["label_path"])
        if not img_src.is_file():
            skipped_unreadable.append(rec["stem"])
            continue
        try:
            with Image.open(img_src) as im:
                im.verify()
            with Image.open(img_src) as im:
                width, height = im.size
        except Exception:  # noqa: BLE001
            skipped_unreadable.append(rec["stem"])
            continue

        dest_img = images_out / img_src.name
        shutil.copy2(img_src, dest_img)

        # Re-parse and filter to panel-only, remapped to class 0. Re-parsing
        # here (rather than trusting the earlier audit's cached boxes)
        # keeps this function correct even if called independently of
        # audit_raw_dataset with a fresh audit dict.
        panel_boxes: list[tuple[float, float, float, float]] = []
        for line in label_src.read_text(encoding="utf-8").strip().splitlines():
            box, error = parse_raw_label_line(line, 0)
            if error or box is None:
                continue
            cid, cx, cy, w, h = box
            if cid == PANEL_CLASS_ID:
                panel_boxes.append((cx, cy, w, h))

        label_dest = labels_out / f"{img_src.stem}.txt"
        lines = [f"{TARGET_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in panel_boxes]
        label_dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        img_sha256 = _sha256_file(dest_img)
        is_confirmed_clean = cat == "numeric_native"
        provenance[img_src.stem] = {
            "source_url": _SOURCE_URL,
            "license": _SOURCE_LICENSE if is_confirmed_clean else _EXPERIMENTAL_UNVERIFIED_LICENSE,
            "rights_holder": _SOURCE_RIGHTS_HOLDER if is_confirmed_clean else _UNVERIFIED_RIGHTS_HOLDER_NOTE,
            "group_key": _MANUAL_SITE_GROUPS.get(img_src.stem),
            "notes": (
                f"original_split={rec['original_split']}; "
                f"provenance_category={cat}; "
                f"retained_panel_boxes={len(panel_boxes)}; "
                f"excluded_classes_present={[c for c in ('bird_drop','cracked','dusty')]}"
            ),
        }
        converted_records.append({
            "filename": img_src.name,
            "stem": img_src.stem,
            "sha256": img_sha256,
            "width": width,
            "height": height,
            "original_split": rec["original_split"],
            "retained_panel_box_count": len(panel_boxes),
            "source_image_path": str(img_src),
            "source_label_path": str(label_src),
        })

    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2))

    stats = {
        "source_dataset_doi": "10.17632/8x7wjsr94v.1",
        "source_license": _SOURCE_LICENSE,
        "allowed_provenance_categories": (
            sorted(allowed_provenance_categories) if allowed_provenance_categories is not None else None
        ),
        "images_considered": len(audit["images_with_panel_box"]),
        "images_converted": len(converted_records),
        "images_skipped_provenance_filter": len(skipped_provenance),
        "skipped_provenance_filter_stems": skipped_provenance,
        "images_skipped_unreadable": len(skipped_unreadable),
        "skipped_unreadable_stems": skipped_unreadable,
        "total_retained_panel_boxes": sum(r["retained_panel_box_count"] for r in converted_records),
        "excluded_class_totals": {
            k: v for k, v in audit["total_class_counts"].items() if k != "panel"
        },
        "records": converted_records,
    }
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, required=True,
                         help="Root of the raw, unmodified Mendeley release (contains train/val/test/{images,labels}).")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Where to write images/, labels/, provenance.json (Step A output).")
    parser.add_argument("--audit-output", type=Path, default=None,
                         help="Optional path to write the full raw-dataset audit JSON.")
    parser.add_argument("--allowed-provenance-categories", type=str, default="numeric_native",
                         help="Comma-separated provenance_category values to include in the conversion "
                              "(default: numeric_native only - the confirmed-clean subset). "
                              "Pass 'all' to disable the filter (NOT recommended).")
    args = parser.parse_args()

    print(f"Auditing raw dataset at {args.raw_root} ...")
    audit = audit_raw_dataset(args.raw_root)
    print(f"  Total label files: {sum(s['total_label_files'] for s in audit['per_split'].values())}")
    print(f"  Class instance totals: {audit['total_class_counts']}")
    print(f"  Malformed lines found: {audit['total_malformed_lines']}")
    print(f"  Images with >=1 panel box: {audit['images_with_panel_box_count']}")
    print(f"  High-IoU (candidate duplicate) panel-box pairs: {audit['total_high_iou_panel_pairs']}")
    print(f"  Provenance-category image counts: {audit['provenance_category_image_counts']}")
    print(f"  Provenance-category panel-box counts: {audit['provenance_category_panel_box_counts']}")

    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in audit.items() if k != "images_with_panel_box"}
        payload["images_with_panel_box_count"] = audit["images_with_panel_box_count"]
        args.audit_output.write_text(json.dumps(payload, indent=2))
        print(f"  Wrote audit: {args.audit_output}")

    if args.allowed_provenance_categories.strip().lower() == "all":
        allowed_categories = None
    else:
        allowed_categories = {c.strip() for c in args.allowed_provenance_categories.split(",") if c.strip()}

    print(f"\nConverting to single-class (solar_panel=0) at {args.output_dir} ...")
    print(f"  Allowed provenance categories: {allowed_categories if allowed_categories is not None else 'ALL (filter disabled)'}")
    stats = convert_to_single_class(
        args.raw_root, args.output_dir, audit=audit, allowed_provenance_categories=allowed_categories
    )
    print(f"  Images converted: {stats['images_converted']} / {stats['images_considered']}")
    print(f"  Images skipped by provenance filter: {stats['images_skipped_provenance_filter']}")
    print(f"  Total retained panel boxes: {stats['total_retained_panel_boxes']}")
    print(f"  Excluded (not converted) class totals: {stats['excluded_class_totals']}")
    if stats["images_skipped_unreadable"]:
        print(f"  WARNING: {stats['images_skipped_unreadable']} image(s) skipped as unreadable: "
              f"{stats['skipped_unreadable_stems'][:10]}")

    stats_path = args.output_dir / "conversion_stats.json"
    stats_path.write_text(json.dumps({k: v for k, v in stats.items() if k != "records"}, indent=2))
    print(f"  Wrote: {stats_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
