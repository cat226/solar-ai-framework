"""training/detection/ground_pv/audit.py — Full dataset audit for the
ground-level PV defect research dataset (Roboflow "panel solar" v1).

Implements every check required before this dataset may be used for
training. Each check is reported separately and never conflated. The raw
dataset is READ-ONLY here: nothing in this module writes into the raw
directory.

Checks (numbered to match docs/GROUND_PV_RESEARCH.md and the task spec):

     1  file structure (train/valid/test, images/, labels/, data.yaml)
     2  YOLO annotation validity (parse, field count, numeric)
     3  missing images (label present, image absent)
     4  missing labels (image present, label absent)
     5  empty labels (present but zero boxes)
     6  invalid class IDs (outside the source names: range)
     7  invalid normalized coordinates (outside [0,1])
     8  zero / negative bounding boxes
     9  image corruption (cannot be opened / verified)
    10  image dimensions (min/max/distribution)
    11  aspect-ratio distribution
    12  class instance counts
    13  images containing each class
    14  images containing multiple (distinct) classes
    15  exact duplicate files (SHA-256)
    16  perceptual near-duplicates (dHash, Hamming <= threshold)
    17  cross-split duplicate leakage (exact + near)
    18  same-image augmentations (near-dup families spanning obvious
        augmentation-suffix filrenames)
    19  extremely similar frames (near-dup families, size)
    20  annotation size distribution (box area as fraction of image)
    21  tiny-object distribution (box area below a small threshold)
    22  images with only defects and no panel annotation
    23  images with a panel annotation but no defect annotation

Depends only on PIL + numpy (already project deps) plus the repo's own
training.evaluation.common perceptual-hash helpers - no new dependency.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from training.detection.ground_pv.class_map import (
    RESEARCH_CLASS_ORDER,
    build_class_id_remap,
    canonical_name,
)
from training.evaluation.common import dhash, hamming_distance

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_SPLIT_ALIASES = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}

# A near-duplicate is a dHash Hamming distance <= this. 5/64 bits is the
# value used elsewhere in this repo's leakage audit for the same purpose.
NEAR_DUP_HAMMING = 5
# "Tiny object": normalized box area below this fraction of the image.
TINY_OBJECT_AREA_FRAC = 0.001

# Filename fragments Roboflow / common tools append to augmented copies.
_AUG_SUFFIX_MARKERS = (
    "_jpg.rf.", ".rf.", "_aug", "-aug", "_flip", "_rot", "_bright", "_mosaic",
)


@dataclass
class LabelIssue:
    split: str
    label_file: str
    line: int
    kind: str
    detail: str


@dataclass
class AuditResult:
    dataset_dir: str
    source_names: dict[int, str] = field(default_factory=dict)
    class_id_remap: dict[int, int] = field(default_factory=dict)
    remap_is_identity: bool = True
    structure: dict[str, Any] = field(default_factory=dict)
    per_split: dict[str, Any] = field(default_factory=dict)
    label_issues: list[LabelIssue] = field(default_factory=list)
    corrupt_images: list[str] = field(default_factory=list)
    image_dimensions: dict[str, Any] = field(default_factory=dict)
    aspect_ratios: dict[str, Any] = field(default_factory=dict)
    class_instance_counts: dict[str, int] = field(default_factory=dict)
    images_per_class: dict[str, int] = field(default_factory=dict)
    multi_class_images: int = 0
    multi_class_combos: dict[str, int] = field(default_factory=dict)
    exact_duplicate_groups: list[list[str]] = field(default_factory=list)
    near_duplicate_pairs: list[dict[str, Any]] = field(default_factory=list)
    cross_split_exact_leakage: list[dict[str, Any]] = field(default_factory=list)
    cross_split_near_leakage: list[dict[str, Any]] = field(default_factory=list)
    likely_augmentation_families: list[dict[str, Any]] = field(default_factory=list)
    box_area_distribution: dict[str, Any] = field(default_factory=dict)
    tiny_object_count: int = 0
    images_defect_only_no_panel: list[str] = field(default_factory=list)
    images_panel_no_defect: list[str] = field(default_factory=list)
    fail_closed_reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["label_issues"] = [i.__dict__ for i in self.label_issues]
        d["label_issue_counts"] = dict(Counter(i.kind for i in self.label_issues))
        d["totals"] = {
            "images": sum(s["num_images"] for s in self.per_split.values()),
            "labels": sum(s["num_label_files"] for s in self.per_split.values()),
            "boxes": sum(self.class_instance_counts.values()),
            "corrupt_images": len(self.corrupt_images),
            "exact_duplicate_groups": len(self.exact_duplicate_groups),
            "near_duplicate_pairs": len(self.near_duplicate_pairs),
            "cross_split_exact_leakage": len(self.cross_split_exact_leakage),
            "cross_split_near_leakage": len(self.cross_split_near_leakage),
        }
        d["passed"] = not self.fail_closed_reasons
        return d


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_splits(dataset_dir: Path) -> dict[str, Path]:
    """Map canonical split name -> its directory, for whatever exists."""
    out: dict[str, Path] = {}
    for entry in sorted(dataset_dir.iterdir()) if dataset_dir.is_dir() else []:
        if entry.is_dir() and entry.name in _SPLIT_ALIASES:
            out[_SPLIT_ALIASES[entry.name]] = entry
    return out


def _images_in(split_dir: Path) -> list[Path]:
    images_dir = split_dir / "images"
    if not images_dir.is_dir():
        return []
    return sorted(p for p in images_dir.glob("*") if p.suffix.lower() in _IMAGE_EXT)


def _parse_label_file(
    path: Path, valid_source_ids: set[int]
) -> tuple[list[tuple[int, float, float, float, float]], list[tuple[int, str, str]]]:
    """Returns (boxes, issues). issues are (lineno, kind, detail)."""
    boxes: list[tuple[int, float, float, float, float]] = []
    issues: list[tuple[int, str, str]] = []
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return boxes, issues
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append((lineno, "malformed_line", f"expected 5 fields, got {len(parts)}"))
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            issues.append((lineno, "non_numeric_field", line))
            continue
        if cls not in valid_source_ids:
            issues.append((lineno, "invalid_class_id", f"class {cls} not in {sorted(valid_source_ids)}"))
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            issues.append((lineno, "coord_out_of_range", f"{(cx, cy, w, h)}"))
        if w <= 0.0 or h <= 0.0:
            issues.append((lineno, "zero_or_negative_box", f"w={w} h={h}"))
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        if x1 < -1e-6 or y1 < -1e-6 or x2 > 1 + 1e-6 or y2 > 1 + 1e-6:
            issues.append((lineno, "box_edge_outside_image", f"({x1:.4f},{y1:.4f})-({x2:.4f},{y2:.4f})"))
        boxes.append((cls, cx, cy, w, h))
    return boxes, issues


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    s = sorted(values)
    n = len(s)
    return {
        "count": n,
        "min": s[0],
        "max": s[-1],
        "mean": sum(s) / n,
        "median": _median(s),
        "p05": s[max(0, int(0.05 * n) - 1)],
        "p95": s[min(n - 1, int(0.95 * n))],
    }


def _read_source_names(dataset_dir: Path) -> dict[int, str]:
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.is_file():
        return {}
    import yaml
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def run_audit(
    dataset_dir: Path,
    *,
    near_dup_hamming: int = NEAR_DUP_HAMMING,
    tiny_object_area_frac: float = TINY_OBJECT_AREA_FRAC,
    max_corruption_fraction: float = 0.05,
) -> AuditResult:
    dataset_dir = Path(dataset_dir)
    result = AuditResult(dataset_dir=str(dataset_dir))

    # --- 1. structure ---
    splits = _find_splits(dataset_dir)
    has_yaml = (dataset_dir / "data.yaml").is_file()
    result.structure = {
        "data_yaml_present": has_yaml,
        "splits_found": sorted(splits.keys()),
        "expected_splits": ["train", "valid", "test"],
    }
    if not dataset_dir.is_dir():
        result.fail_closed_reasons.append(f"dataset directory does not exist: {dataset_dir}")
        return result
    if not has_yaml:
        result.fail_closed_reasons.append("data.yaml is missing")
    for expected in ("train", "valid", "test"):
        if expected not in splits:
            result.fail_closed_reasons.append(f"missing split: {expected}")

    # --- source names + class-id remap ---
    result.source_names = _read_source_names(dataset_dir)
    if result.source_names:
        try:
            remap = build_class_id_remap(result.source_names)
            result.class_id_remap = remap
            result.remap_is_identity = all(k == v for k, v in remap.items())
            if len(remap) != len(RESEARCH_CLASS_ORDER):
                result.fail_closed_reasons.append(
                    f"source data.yaml has {len(remap)} classes; research taxonomy needs "
                    f"{len(RESEARCH_CLASS_ORDER)} ({RESEARCH_CLASS_ORDER})"
                )
        except (KeyError, ValueError) as exc:
            result.fail_closed_reasons.append(f"class mapping failed: {exc}")
    else:
        result.fail_closed_reasons.append("could not read names: from data.yaml")

    valid_source_ids = set(result.source_names.keys())
    id_to_canon = {}
    for sid, sname in result.source_names.items():
        try:
            id_to_canon[sid] = canonical_name(sname)
        except KeyError:
            pass  # already recorded as a fail-closed reason above

    # --- per-split pass: labels, images, geometry, dims, dup hashing ---
    all_sha: dict[str, list[str]] = defaultdict(list)          # sha256 -> ["split/name", ...]
    all_dhash: list[tuple[str, str, int]] = []                 # (split, name, dhash)
    class_instances: Counter[str] = Counter()
    images_with_class: defaultdict[str, set[str]] = defaultdict(set)
    multi_combo: Counter[str] = Counter()
    box_area_fracs: list[float] = []
    widths: list[int] = []
    heights: list[int] = []
    aspects: list[float] = []
    tiny = 0
    total_images = 0

    for split, split_dir in splits.items():
        images = _images_in(split_dir)
        labels_dir = split_dir / "labels"
        image_stems = {p.stem for p in images}
        label_files = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
        label_stems = {p.stem for p in label_files}

        missing_labels = sorted(image_stems - label_stems)      # 4
        missing_images = sorted(label_stems - image_stems)      # 3
        empty_labels: list[str] = []
        corrupt_here: list[str] = []
        defect_only: list[str] = []
        panel_no_defect: list[str] = []

        for img in images:
            total_images += 1
            try:
                with Image.open(img) as im:
                    im.verify()
                with Image.open(img) as im2:
                    w, h = im2.size
                    im2.convert("RGB")
            except Exception as exc:  # noqa: BLE001 - any decode failure counts
                corrupt_here.append(img.name)
                result.corrupt_images.append(f"{split}/{img.name}")
                continue
            widths.append(w)
            heights.append(h)
            if h:
                aspects.append(w / h)
            sha = _sha256_file(img)
            all_sha[sha].append(f"{split}/{img.name}")
            try:
                with Image.open(img) as im3:
                    all_dhash.append((split, img.name, dhash(im3.convert("RGB"))))
            except Exception:  # noqa: BLE001
                pass

            label_path = labels_dir / f"{img.stem}.txt"
            if not label_path.is_file():
                continue
            boxes, issues = _parse_label_file(label_path, valid_source_ids)
            for lineno, kind, detail in issues:
                result.label_issues.append(LabelIssue(split, label_path.name, lineno, kind, detail))
            if not boxes:
                empty_labels.append(img.name)
                continue
            canon_here: set[str] = set()
            has_panel = False
            has_defect = False
            for (cls, cx, cy, bw, bh) in boxes:
                canon = id_to_canon.get(cls)
                if canon is None:
                    continue
                class_instances[canon] += 1
                canon_here.add(canon)
                images_with_class[canon].add(f"{split}/{img.name}")
                area = max(0.0, bw) * max(0.0, bh)
                box_area_fracs.append(area)
                if area < tiny_object_area_frac:
                    tiny += 1
                if canon == "panel":
                    has_panel = True
                else:
                    has_defect = True
            if len(canon_here) > 1:
                multi_combo["+".join(sorted(canon_here))] += 1
            if has_defect and not has_panel:
                defect_only.append(img.name)
            if has_panel and not has_defect:
                panel_no_defect.append(img.name)

        # orphan label files with content but no image -> also parse for issues
        for lf in label_files:
            if lf.stem in image_stems:
                continue
            _, issues = _parse_label_file(lf, valid_source_ids)
            for lineno, kind, detail in issues:
                result.label_issues.append(LabelIssue(split, lf.name, lineno, kind, detail))

        result.per_split[split] = {
            "num_images": len(images),
            "num_label_files": len(label_files),
            "missing_label_files": missing_labels,
            "missing_label_count": len(missing_labels),
            "label_files_without_image": missing_images,
            "label_files_without_image_count": len(missing_images),
            "empty_label_files": empty_labels,
            "empty_label_count": len(empty_labels),
            "corrupt_images": corrupt_here,
            "corrupt_image_count": len(corrupt_here),
            "images_defect_only_no_panel_count": len(defect_only),
            "images_panel_no_defect_count": len(panel_no_defect),
        }
        result.images_defect_only_no_panel.extend(f"{split}/{n}" for n in defect_only)
        result.images_panel_no_defect.extend(f"{split}/{n}" for n in panel_no_defect)

    # --- 9. corruption gate ---
    if total_images and (len(result.corrupt_images) / total_images) > max_corruption_fraction:
        result.fail_closed_reasons.append(
            f"{len(result.corrupt_images)}/{total_images} images corrupt "
            f"(> {max_corruption_fraction:.0%} threshold)"
        )

    # --- 10-11. dimensions / aspect ratio ---
    result.image_dimensions = {"width": _distribution([float(x) for x in widths]),
                               "height": _distribution([float(x) for x in heights])}
    result.aspect_ratios = _distribution(aspects)

    # --- 12-14. class stats ---
    result.class_instance_counts = {c: class_instances.get(c, 0) for c in RESEARCH_CLASS_ORDER}
    result.images_per_class = {c: len(images_with_class.get(c, set())) for c in RESEARCH_CLASS_ORDER}
    result.multi_class_images = sum(multi_combo.values())
    result.multi_class_combos = dict(multi_combo)

    # --- 20-21. box area / tiny objects ---
    result.box_area_distribution = _distribution(box_area_fracs)
    result.tiny_object_count = tiny

    # --- 15,17(exact). exact duplicates + cross-split exact leakage ---
    for sha, members in sorted(all_sha.items()):
        if len(members) < 2:
            continue
        result.exact_duplicate_groups.append(members)
        splits_involved = {m.split("/", 1)[0] for m in members}
        if len(splits_involved) > 1:
            result.cross_split_exact_leakage.append({"sha256": sha, "members": members})

    # --- 16,17(near),18,19. perceptual near-duplicates ---
    near_pairs: list[dict[str, Any]] = []
    n = len(all_dhash)
    for i in range(n):
        si, ni, hi = all_dhash[i]
        for j in range(i + 1, n):
            sj, nj, hj = all_dhash[j]
            dist = hamming_distance(hi, hj)
            if dist <= near_dup_hamming:
                near_pairs.append({
                    "a": f"{si}/{ni}", "b": f"{sj}/{nj}", "hamming": dist,
                    "cross_split": si != sj,
                })
    result.near_duplicate_pairs = near_pairs
    result.cross_split_near_leakage = [p for p in near_pairs if p["cross_split"] and p["hamming"] > 0]

    # --- 18. likely same-image augmentation families ---
    #     group near-dup members whose filenames share a base once known
    #     augmentation suffixes are stripped.
    fam_map: defaultdict[str, set[str]] = defaultdict(set)
    for p in near_pairs:
        for member in (p["a"], p["b"]):
            base = member
            for marker in _AUG_SUFFIX_MARKERS:
                if marker in base:
                    base = base.split(marker, 1)[0]
            fam_map[base].add(member)
    result.likely_augmentation_families = [
        {"base": base, "members": sorted(members), "size": len(members)}
        for base, members in sorted(fam_map.items()) if len(members) > 1
    ]

    # --- fail-closed: strong cross-split leakage ---
    if result.cross_split_exact_leakage:
        result.fail_closed_reasons.append(
            f"{len(result.cross_split_exact_leakage)} exact-duplicate image(s) span multiple splits "
            "(train/val/test leakage) - a leakage-clean split is required before training"
        )

    # any hard label issue (malformed, non-numeric, invalid class id, out-of-range) is fail-closed
    hard_kinds = {"malformed_line", "non_numeric_field", "invalid_class_id", "coord_out_of_range",
                  "zero_or_negative_box"}
    hard = [i for i in result.label_issues if i.kind in hard_kinds]
    if hard:
        result.fail_closed_reasons.append(f"{len(hard)} hard annotation error(s) (see label_issues)")

    return result


def write_reports(result: AuditResult, reports_dir: Path) -> tuple[Path, Path]:
    """Write dataset_audit.json + dataset_audit.md. Returns (json_path, md_path)."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()

    json_path = reports_dir / "dataset_audit.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    md_path = reports_dir / "dataset_audit.md"
    md_path.write_text(_render_markdown(result, payload), encoding="utf-8")
    return json_path, md_path


def _render_markdown(result: AuditResult, payload: dict[str, Any]) -> str:
    t = payload["totals"]
    lines: list[str] = []
    lines.append("# Ground-level PV research dataset — audit report\n")
    lines.append(f"Dataset directory: `{result.dataset_dir}`\n")
    lines.append(f"**Overall: {'PASS' if payload['passed'] else 'FAIL (see fail-closed reasons)'}**\n")
    if result.fail_closed_reasons:
        lines.append("## Fail-closed reasons\n")
        for r in result.fail_closed_reasons:
            lines.append(f"- {r}")
        lines.append("")

    lines.append("## 1. Structure\n")
    lines.append(f"- data.yaml present: {result.structure.get('data_yaml_present')}")
    lines.append(f"- splits found: {result.structure.get('splits_found')}")
    lines.append(f"- source `names:` = {result.source_names}")
    lines.append(f"- class-id remap (source→research): {result.class_id_remap} "
                 f"(identity: {result.remap_is_identity})\n")

    lines.append("## Totals\n")
    lines.append(f"- images: {t['images']}   label files: {t['labels']}   boxes: {t['boxes']}")
    lines.append(f"- corrupt images: {t['corrupt_images']}")
    lines.append(f"- exact-duplicate groups: {t['exact_duplicate_groups']}")
    lines.append(f"- near-duplicate pairs (Hamming ≤ {NEAR_DUP_HAMMING}): {t['near_duplicate_pairs']}")
    lines.append(f"- cross-split exact leakage: {t['cross_split_exact_leakage']}")
    lines.append(f"- cross-split near leakage: {t['cross_split_near_leakage']}\n")

    lines.append("## Per-split\n")
    lines.append("| split | images | labels | missing labels | empty labels | label w/o image | corrupt | defect-only (no panel) | panel (no defect) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s, d in result.per_split.items():
        lines.append(
            f"| {s} | {d['num_images']} | {d['num_label_files']} | {d['missing_label_count']} | "
            f"{d['empty_label_count']} | {d['label_files_without_image_count']} | {d['corrupt_image_count']} | "
            f"{d['images_defect_only_no_panel_count']} | {d['images_panel_no_defect_count']} |"
        )
    lines.append("")

    lines.append("## Class distribution\n")
    lines.append("| class | instances | images containing |")
    lines.append("|---|---|---|")
    for c in RESEARCH_CLASS_ORDER:
        lines.append(f"| {c} | {result.class_instance_counts.get(c, 0)} | {result.images_per_class.get(c, 0)} |")
    lines.append(f"\n- images with >1 distinct class: {result.multi_class_images}")
    lines.append(f"- combinations: {result.multi_class_combos}\n")

    lines.append("## Geometry\n")
    lines.append(f"- image width dist: {result.image_dimensions.get('width')}")
    lines.append(f"- image height dist: {result.image_dimensions.get('height')}")
    lines.append(f"- aspect ratio dist: {result.aspect_ratios}")
    lines.append(f"- box area (frac of image) dist: {result.box_area_distribution}")
    lines.append(f"- tiny objects (area < {TINY_OBJECT_AREA_FRAC}): {result.tiny_object_count}\n")

    lines.append("## Duplication & leakage\n")
    lines.append(f"- exact-duplicate groups: {len(result.exact_duplicate_groups)}")
    lines.append(f"- near-duplicate pairs: {len(result.near_duplicate_pairs)}")
    lines.append(f"- cross-split exact leakage: {len(result.cross_split_exact_leakage)}")
    lines.append(f"- cross-split near leakage: {len(result.cross_split_near_leakage)}")
    lines.append(f"- likely augmentation families: {len(result.likely_augmentation_families)}")
    for grp in result.cross_split_exact_leakage[:20]:
        lines.append(f"  - EXACT: {grp['members']}")
    for p in result.cross_split_near_leakage[:20]:
        lines.append(f"  - NEAR (h={p['hamming']}): {p['a']}  <->  {p['b']}")
    lines.append("")

    lines.append("## Annotation issues\n")
    lines.append(f"- issue counts by kind: {payload['label_issue_counts']}")
    for i in result.label_issues[:40]:
        lines.append(f"  - {i.split}/{i.label_file}:{i.line}  {i.kind}  {i.detail}")
    lines.append("")
    return "\n".join(lines) + "\n"
