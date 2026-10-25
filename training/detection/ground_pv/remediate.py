"""training/detection/ground_pv/remediate.py — Build a derived, research-
training dataset from the audited-but-BLOCKED raw Roboflow "panel solar" v1
dataset.

The raw dataset at `datasets/ground_pv_research/raw/panel_solar_bw945_v1/`
is IMMUTABLE and is only ever read here.

What this does (see docs/GROUND_PV_RESEARCH.md and the remediation report):

1. Polygon -> bounding box: every valid YOLO-segmentation polygon line is
   converted to its axis-aligned box (reuses
   normalize._remap_label_text - the "existing normalize.py implementation").
2. Class filter: the FIRST experiment uses only panel / cracked / dusty.
   `bird_drop` annotations are dropped (audit: 9 images / ~3 unique photos).
   Derived taxonomy is contiguous: 0=panel, 1=cracked, 2=dusty.
3. Exact-duplicate removal: the one image whose bytes appear in two splits
   is de-duplicated (one copy kept, the rest dropped).
4. Leakage-clean split: images are grouped into near-duplicate FAMILIES
   BEFORE any split assignment, using
       exact SHA-256  U  shared source basename (Roboflow .rf. suffix
       stripped)  U  16x16 dHash Hamming <= threshold
   then whole families are assigned to train/valid/test (~70/20/10 by image
   count, deterministic). A family never spans two splits.
5. The original official Roboflow split is recorded separately for
   provenance (splits/official_split/manifest.csv) - never overwritten.
6. Full traceability: every derived image records its source path, source
   official split, family id, SHA-256 and class set.

Depends on PIL + numpy (already project deps).
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from training.detection.ground_pv.class_map import (
    EXCLUDED_FROM_FIRST_EXPERIMENT,
    FIRST_EXPERIMENT_CLASS_NAMES,
    FIRST_EXPERIMENT_CLASS_ORDER,
)
from training.detection.ground_pv.normalize import _remap_label_text

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_SPLIT_ALIASES = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}

# Roboflow export suffixes that mark a re-export / augmentation of one
# source image. Everything before the first marker is the source basename.
# Checked longest-first so "pv_crack-1-_JPG_jpg.rf.<h>" resolves to
# "pv_crack-1-" (original stem), not "pv_crack-1-_JPG".
_RF_MARKERS = tuple(sorted((
    "_jpg.rf.", "_jpeg.rf.", "_png.rf.", "_png_jpg.rf.", "_jpeg_jpg.rf.",
    "_JPG_jpg.rf.", "_JPEG_jpg.rf.", "_PNG_jpg.rf.", ".rf.",
), key=len, reverse=True))

# 16x16 dHash (256-bit). Hamming <= this between two DIFFERENT-basename
# images merges their families. 8/256 bits ~ 3%. Chosen from a threshold
# sweep on this dataset: the value at which the residual cross-split
# near-duplicate count reaches 0 without collapsing everything into one
# giant family (8x8 dHash does collapse - too coarse for low-texture
# panel close-ups - so a finer 16x16 hash is used here).
DEFAULT_DHASH16_MERGE_HAMMING = 8

# Source dataset taxonomy (from its real data.yaml, alphabetical):
_SOURCE_NAMES = {0: "bird_drop", 1: "cracked", 2: "dusty", 3: "panel"}
# source id -> first-experiment id, or None to DROP that class entirely.
_SOURCE_TO_EXPERIMENT: dict[int, int | None] = {
    0: None,  # bird_drop - excluded
    1: FIRST_EXPERIMENT_CLASS_ORDER.index("cracked"),   # 1 -> 1
    2: FIRST_EXPERIMENT_CLASS_ORDER.index("dusty"),     # 2 -> 2
    3: FIRST_EXPERIMENT_CLASS_ORDER.index("panel"),     # 3 -> 0
}

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def source_basename(filename: str) -> str:
    """The uploaded source image's name, with the Roboflow export suffix
    stripped. Two files with the same basename are re-exports / augmentations
    of one source photograph."""
    name = Path(filename).name
    for marker in _RF_MARKERS:
        if marker in name:
            return name.split(marker, 1)[0]
    return Path(name).stem


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dhash16_bytes(image: Image.Image) -> bytes:
    """256-bit difference hash as 32 bytes (for fast vectorised Hamming)."""
    g = image.convert("L").resize((17, 16), Image.LANCZOS)
    px = np.asarray(g, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten().astype(np.uint8)
    return np.packbits(bits).tobytes()


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


@dataclass
class _SrcImage:
    key: str            # "<official_split>/<filename>"
    official_split: str
    filename: str
    path: Path
    label_path: Path
    sha256: str
    basename: str
    dhash16: bytes
    classes: set[str] = field(default_factory=set)   # experiment classes present
    had_bird_drop: bool = False


def _scan_raw(raw_dir: Path) -> list[_SrcImage]:
    raw_dir = Path(raw_dir)
    out: list[_SrcImage] = []
    for entry in sorted(raw_dir.iterdir()):
        if not entry.is_dir() or entry.name not in _SPLIT_ALIASES:
            continue
        split = _SPLIT_ALIASES[entry.name]
        images_dir, labels_dir = entry / "images", entry / "labels"
        if not images_dir.is_dir():
            continue
        for img in sorted(images_dir.glob("*")):
            if img.suffix.lower() not in _IMAGE_EXT:
                continue
            with Image.open(img) as im:
                im.verify()
            with Image.open(img) as im:
                dh = _dhash16_bytes(im.convert("RGB"))
            lbl = labels_dir / f"{img.stem}.txt"
            classes: set[str] = set()
            had_bd = False
            if lbl.is_file():
                for line in lbl.read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cid = int(parts[0])
                    except ValueError:
                        continue
                    if cid == 0:
                        had_bd = True
                    elif cid in _SOURCE_NAMES:
                        classes.add(_SOURCE_NAMES[cid])
            out.append(_SrcImage(
                key=f"{split}/{img.name}", official_split=split, filename=img.name,
                path=img, label_path=lbl, sha256=_sha256_file(img),
                basename=source_basename(img.name), dhash16=dh,
                classes=classes, had_bird_drop=had_bd,
            ))
    return out


def compute_families(
    images: list[_SrcImage], *, dhash16_merge_hamming: int = DEFAULT_DHASH16_MERGE_HAMMING
) -> tuple[dict[str, str], list[dict[str, Any]], list[list[str]]]:
    """Returns (image_key -> family_id, near_dup_pair_records, exact_dup_groups).

    Family = connected component over three edge types:
      * identical SHA-256
      * identical source basename
      * 16x16 dHash Hamming <= threshold between DIFFERENT-basename images
    """
    uf = _UnionFind()
    for im in images:
        uf.find(im.key)

    # exact-SHA groups
    by_sha: dict[str, list[str]] = defaultdict(list)
    for im in images:
        by_sha[im.sha256].append(im.key)
    exact_groups = [sorted(v) for v in by_sha.values() if len(v) > 1]
    for grp in exact_groups:
        for other in grp[1:]:
            uf.union(grp[0], other)

    # basename groups
    by_base: dict[str, list[str]] = defaultdict(list)
    for im in images:
        by_base[im.basename].append(im.key)
    for members in by_base.values():
        for other in members[1:]:
            uf.union(members[0], other)

    # 16x16 dHash near-duplicates between different-basename images
    hashes = np.frombuffer(b"".join(im.dhash16 for im in images), dtype=np.uint8).reshape(len(images), 32)
    base_arr = [im.basename for im in images]
    keys = [im.key for im in images]
    near_pairs: list[dict[str, Any]] = []
    for i in range(len(images)):
        xor = np.bitwise_xor(hashes[i + 1:], hashes[i])
        dist = _POPCOUNT[xor].sum(axis=1)
        for off in np.where(dist <= dhash16_merge_hamming)[0]:
            j = i + 1 + int(off)
            if base_arr[i] == base_arr[j]:
                continue  # already unioned by basename
            h = int(dist[off])
            near_pairs.append({"a": keys[i], "b": keys[j], "dhash16_hamming": h})
            uf.union(keys[i], keys[j])

    fam_of = {im.key: uf.find(im.key) for im in images}
    # rename family roots to stable f0000-style ids ordered by (size desc, root)
    sizes = Counter(fam_of.values())
    ordered = sorted(sizes, key=lambda r: (-sizes[r], r))
    rename = {root: f"fam{idx:04d}" for idx, root in enumerate(ordered)}
    fam_of = {k: rename[v] for k, v in fam_of.items()}
    return fam_of, near_pairs, exact_groups


def assign_leakage_clean_splits(
    family_sizes: dict[str, int], *, ratios: dict[str, float], seed: int = 42
) -> dict[str, str]:
    """Assign whole families to splits, greedily filling the most
    under-target split first. Deterministic (sorted by size desc, id).
    `seed` only tie-breaks equal ratio-fills via a fixed permutation."""
    rng = np.random.default_rng(seed)
    order = sorted(family_sizes, key=lambda f: (-family_sizes[f], f))
    cur = {s: 0 for s in ratios}
    assign: dict[str, str] = {}
    for fam in order:
        size = family_sizes[fam]
        # pick split minimising projected (count/target); random tie-break
        scored = sorted(
            ratios,
            key=lambda s: ((cur[s] + size) / ratios[s], rng.random()),
        )
        assign[fam] = scored[0]
        cur[scored[0]] += size
    return assign


@dataclass
class RemediationReport:
    raw_dir: str
    derived_dir: str
    raw_image_count: int = 0
    derived_image_count: int = 0
    images_removed_total: int = 0
    exact_duplicates_removed: int = 0
    near_duplicate_groups: int = 0
    near_duplicate_pairs_16x16: int = 0
    family_grouping: dict[str, Any] = field(default_factory=dict)
    final_split_counts: dict[str, int] = field(default_factory=dict)
    class_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    unique_groups_per_split: dict[str, int] = field(default_factory=dict)
    no_group_crosses_splits: bool = False
    polygon_to_box_conversions: int = 0
    unparseable_lines_dropped: int = 0
    bird_drop_annotations_excluded: int = 0
    bird_drop_images_affected: int = 0
    other_exclusions: list[dict[str, Any]] = field(default_factory=list)
    derived_taxonomy: dict[int, str] = field(default_factory=dict)
    official_split_counts: dict[str, int] = field(default_factory=dict)
    seed: int = 42
    filename_collisions_renamed: int = 0
    residual_cross_split_near_dup_pairs: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def remediate(
    raw_dir: Path,
    derived_dir: Path,
    splits_dir: Path,
    reports_dir: Path,
    *,
    ratios: dict[str, float] | None = None,
    seed: int = 42,
    dhash16_merge_hamming: int = DEFAULT_DHASH16_MERGE_HAMMING,
    overwrite: bool = False,
) -> RemediationReport:
    raw_dir, derived_dir = Path(raw_dir), Path(derived_dir)
    splits_dir, reports_dir = Path(splits_dir), Path(reports_dir)
    ratios = ratios or {"train": 0.70, "valid": 0.20, "test": 0.10}

    if not (raw_dir / "data.yaml").is_file():
        raise FileNotFoundError(f"no raw data.yaml under {raw_dir}")
    if derived_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{derived_dir} exists (pass overwrite=True)")
        shutil.rmtree(derived_dir)

    raw_sha_before = _sha256_file(raw_dir / "data.yaml")

    images = _scan_raw(raw_dir)
    rep = RemediationReport(raw_dir=str(raw_dir), derived_dir=str(derived_dir), seed=seed)
    rep.raw_image_count = len(images)
    rep.derived_taxonomy = dict(FIRST_EXPERIMENT_CLASS_NAMES)
    rep.official_split_counts = dict(Counter(im.official_split for im in images))

    fam_of, near_pairs, exact_groups = compute_families(
        images, dhash16_merge_hamming=dhash16_merge_hamming
    )
    rep.near_duplicate_pairs_16x16 = len(near_pairs)

    # --- exact-duplicate removal (keep the lexicographically-first copy) ---
    drop_keys: set[str] = set()
    exact_removed = 0
    for grp in exact_groups:
        for extra in sorted(grp)[1:]:
            drop_keys.add(extra)
            exact_removed += 1
    rep.exact_duplicates_removed = exact_removed
    rep.other_exclusions.append({
        "reason": "exact byte-duplicate (kept one copy)",
        "count": exact_removed,
        "groups": [{"sha_group": g} for g in exact_groups],
    })

    kept = [im for im in images if im.key not in drop_keys]

    # --- family sizes over KEPT images, then split assignment ---
    fam_members: dict[str, list[_SrcImage]] = defaultdict(list)
    for im in kept:
        fam_members[fam_of[im.key]].append(im)
    fam_sizes = {f: len(v) for f, v in fam_members.items()}
    fam_split = assign_leakage_clean_splits(fam_sizes, ratios=ratios, seed=seed)

    # --- materialise derived dataset ---
    for split in ("train", "valid", "test"):
        (derived_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (derived_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    remap = _SOURCE_TO_EXPERIMENT
    trace_rows: list[dict[str, Any]] = []
    class_counts: dict[str, Counter] = {s: Counter() for s in ("train", "valid", "test")}
    used_names: dict[str, set[str]] = {s: set() for s in ("train", "valid", "test")}
    total_poly = total_unparse = total_bd_ann = 0
    bd_images = 0
    collisions = 0
    invalid_box_files: list[str] = []

    for im in kept:
        split = fam_split[fam_of[im.key]]
        text = im.label_path.read_text(encoding="utf-8", errors="replace") if im.label_path.is_file() else ""
        new_text, polys, dropped, bd_excluded = _remap_label_text(text, remap)
        total_poly += polys
        total_unparse += dropped
        total_bd_ann += bd_excluded
        if im.had_bird_drop:
            bd_images += 1

        # validate every derived box
        for ln in new_text.splitlines():
            p = ln.split()
            if len(p) != 5:
                invalid_box_files.append(f"{split}/{im.filename}:{ln!r}")
                continue
            cid = int(p[0]); vals = [float(x) for x in p[1:]]
            if not (0 <= cid < len(FIRST_EXPERIMENT_CLASS_ORDER)) or not all(0 <= v <= 1 for v in vals) \
               or vals[2] <= 0 or vals[3] <= 0:
                invalid_box_files.append(f"{split}/{im.filename}:{ln!r}")
            else:
                class_counts[split][FIRST_EXPERIMENT_CLASS_NAMES[cid]] += 1

        out_name = im.filename
        if out_name in used_names[split]:
            out_name = f"{im.official_split}__{im.filename}"
            collisions += 1
        used_names[split].add(out_name)

        shutil.copy2(im.path, derived_dir / split / "images" / out_name)
        (derived_dir / split / "labels" / f"{Path(out_name).stem}.txt").write_text(new_text, encoding="utf-8")

        trace_rows.append({
            "derived_filename": out_name,
            "derived_split": split,
            "family_id": fam_of[im.key],
            "source_path": str(im.path.relative_to(raw_dir)).replace("\\", "/"),
            "source_official_split": im.official_split,
            "source_filename": im.filename,
            "source_basename": im.basename,
            "sha256": im.sha256,
            "classes": ";".join(sorted(im.classes)) or "(none)",
            "had_bird_drop": im.had_bird_drop,
        })

    if invalid_box_files:
        raise RuntimeError(
            f"{len(invalid_box_files)} derived label line(s) are not valid YOLO boxes - "
            f"refusing to finish. First few: {invalid_box_files[:5]}"
        )

    # --- data.yaml for the derived 3-class dataset ---
    import yaml
    (derived_dir / "data.yaml").write_text(yaml.safe_dump({
        "path": str(derived_dir.resolve()),
        "train": "train/images", "val": "valid/images", "test": "test/images",
        "names": {i: FIRST_EXPERIMENT_CLASS_NAMES[i] for i in range(len(FIRST_EXPERIMENT_CLASS_ORDER))},
    }, sort_keys=False), encoding="utf-8")

    # --- manifests ---
    splits_dir.mkdir(parents=True, exist_ok=True)
    official_dir = splits_dir / "official_split"
    clean_dir = splits_dir / "leakage_clean_split"
    official_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    with (official_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source_path", "official_split", "source_basename", "sha256"])
        for im in images:
            w.writerow([str(im.path.relative_to(raw_dir)).replace("\\", "/"),
                        im.official_split, im.basename, im.sha256])
    (official_dir / "README.md").write_text(
        "# Official Roboflow split (provenance reference)\n\n"
        "This records the source dataset's own train/valid/test assignment for every raw "
        "image. It is NOT overwritten by remediation and is NOT used for training - the "
        "leakage-clean split is. Derived from the immutable raw dataset only.\n",
        encoding="utf-8")

    fieldnames = list(trace_rows[0].keys()) if trace_rows else []
    for target in (clean_dir / "manifest.csv", derived_dir / "TRACEABILITY.csv"):
        with target.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(trace_rows)

    # --- verify: no family crosses splits ---
    fam_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in trace_rows:
        fam_to_splits[row["family_id"]].add(row["derived_split"])
    crossing = {f: sorted(s) for f, s in fam_to_splits.items() if len(s) > 1}
    rep.no_group_crosses_splits = not crossing

    # residual near-dup pairs that still cross splits (should be 0 by construction)
    key_to_split = {row["family_id"]: row["derived_split"] for row in trace_rows}
    residual = 0
    for p in near_pairs:
        fa, fb = fam_of.get(p["a"]), fam_of.get(p["b"])
        if fa in key_to_split and fb in key_to_split and key_to_split[fa] != key_to_split[fb]:
            residual += 1
    rep.residual_cross_split_near_dup_pairs = residual

    # --- fill report ---
    rep.derived_image_count = len(trace_rows)
    rep.images_removed_total = rep.raw_image_count - rep.derived_image_count
    rep.near_duplicate_groups = len({fam_of[im.key] for im in images})
    rep.family_grouping = {
        "definition": "exact SHA-256 U source basename U 16x16 dHash Hamming <= "
                      f"{dhash16_merge_hamming} (different-basename only)",
        "total_families_over_raw": len({fam_of[im.key] for im in images}),
        "total_families_over_kept": len(fam_sizes),
        "largest_family_images": max(fam_sizes.values()) if fam_sizes else 0,
        "singleton_families": sum(1 for v in fam_sizes.values() if v == 1),
        "dhash16_merge_hamming": dhash16_merge_hamming,
    }
    rep.final_split_counts = {s: sum(1 for r in trace_rows if r["derived_split"] == s)
                              for s in ("train", "valid", "test")}
    rep.class_counts = {s: dict(class_counts[s]) for s in ("train", "valid", "test")}
    rep.class_counts["total"] = dict(sum((Counter(v) for v in class_counts.values()), Counter()))
    rep.unique_groups_per_split = {s: len(v) for s, v in
                                   ((s, {r["family_id"] for r in trace_rows if r["derived_split"] == s})
                                    for s in ("train", "valid", "test"))}
    rep.polygon_to_box_conversions = total_poly
    rep.unparseable_lines_dropped = total_unparse
    rep.bird_drop_annotations_excluded = total_bd_ann
    rep.bird_drop_images_affected = bd_images
    rep.filename_collisions_renamed = collisions
    rep.other_exclusions.append({
        "reason": "bird_drop class excluded from the first experiment",
        "annotations_dropped": total_bd_ann,
        "images_affected": bd_images,
        "note": EXCLUDED_FROM_FIRST_EXPERIMENT["bird_drop"],
    })

    # --- raw untouched check ---
    raw_sha_after = _sha256_file(raw_dir / "data.yaml")
    if raw_sha_before != raw_sha_after:
        raise RuntimeError("raw data.yaml hash changed during remediation - aborting")

    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = rep.as_dict()
    payload["crossing_families"] = crossing
    (reports_dir / "remediation_report.json").write_text(json.dumps(payload, indent=2, default=str),
                                                         encoding="utf-8")
    (reports_dir / "remediation_report.md").write_text(_render_md(rep, crossing), encoding="utf-8")
    (derived_dir / "REMEDIATION.json").write_text(json.dumps(payload, indent=2, default=str),
                                                  encoding="utf-8")
    return rep


def _render_md(rep: RemediationReport, crossing: dict[str, list[str]]) -> str:
    L: list[str] = []
    L.append("# Ground-level PV research dataset — remediation report\n")
    L.append(f"- raw dir (immutable): `{rep.raw_dir}`")
    L.append(f"- derived dir: `{rep.derived_dir}`")
    L.append(f"- seed: {rep.seed}\n")
    verdict = "APPROVED WITH CAVEATS" if rep.no_group_crosses_splits and not crossing else "BLOCKED"
    L.append(f"**Remediation verdict: {verdict}**\n")

    L.append("## Counts\n")
    L.append(f"- raw images: {rep.raw_image_count}")
    L.append(f"- derived images: {rep.derived_image_count}")
    L.append(f"- images removed: {rep.images_removed_total} "
             f"(exact byte-duplicates: {rep.exact_duplicates_removed})")
    L.append(f"- filename collisions renamed: {rep.filename_collisions_renamed}\n")

    L.append("## Near-duplicate families\n")
    fg = rep.family_grouping
    L.append(f"- grouping: {fg['definition']}")
    L.append(f"- families over raw: {fg['total_families_over_raw']}   "
             f"over kept: {fg['total_families_over_kept']}")
    L.append(f"- largest family: {fg['largest_family_images']} images   "
             f"singletons: {fg['singleton_families']}")
    L.append(f"- 16x16 dHash near-duplicate pairs merged: {rep.near_duplicate_pairs_16x16}\n")

    L.append("## Final split (leakage-clean)\n")
    L.append("| split | images | unique families | target |")
    L.append("|---|---|---|---|")
    tgt = {"train": "~70%", "valid": "~20%", "test": "~10%"}
    for s in ("train", "valid", "test"):
        L.append(f"| {s} | {rep.final_split_counts[s]} | {rep.unique_groups_per_split[s]} | {tgt[s]} |")
    L.append(f"\n- **no family crosses splits: {rep.no_group_crosses_splits}**")
    L.append(f"- residual cross-split near-duplicate pairs: {rep.residual_cross_split_near_dup_pairs}")
    if crossing:
        L.append(f"- CROSSING FAMILIES (should be empty): {crossing}")
    L.append(f"\nOfficial Roboflow split (reference only): {rep.official_split_counts}\n")

    L.append("## Class counts (derived, per split)\n")
    L.append("| class | train | valid | test | total |")
    L.append("|---|---|---|---|---|")
    for c in FIRST_EXPERIMENT_CLASS_ORDER:
        L.append(f"| {c} | {rep.class_counts['train'].get(c,0)} | {rep.class_counts['valid'].get(c,0)} "
                 f"| {rep.class_counts['test'].get(c,0)} | {rep.class_counts['total'].get(c,0)} |")
    L.append("")

    L.append("## Conversions & exclusions\n")
    L.append(f"- polygon → bounding box conversions: {rep.polygon_to_box_conversions}")
    L.append(f"- unparseable label lines dropped: {rep.unparseable_lines_dropped}")
    L.append(f"- **bird_drop excluded**: {rep.bird_drop_annotations_excluded} annotations "
             f"across {rep.bird_drop_images_affected} images")
    L.append(f"  - reason: {EXCLUDED_FROM_FIRST_EXPERIMENT['bird_drop']}")
    L.append(f"  - the affected images are kept (their `panel` boxes remain); only bird_drop boxes are dropped")
    L.append(f"- derived taxonomy: {rep.derived_taxonomy}\n")

    L.append("## Traceability\n")
    L.append("- `<derived>/TRACEABILITY.csv` and `splits/leakage_clean_split/manifest.csv`: "
             "every derived image → source_path, source_official_split, family_id, sha256, classes.")
    L.append("- `splits/official_split/manifest.csv`: every raw image → its official split "
             "(provenance; not overwritten, not used for training).\n")
    return "\n".join(L) + "\n"
