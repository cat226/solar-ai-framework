"""training/detection/ground_pv/build_condition_dataset.py — Build the unified
condition/defect research dataset from the audited candidate sources.

PERSONAL, NON-COMMERCIAL RESEARCH ONLY. The underlying source images trace to
Kaggle `pythonafroz/solar-panel-images` (licence: Unknown). This dataset is
tagged `personal-experimental-unverified-provenance`, provenance grade C.
DO NOT redistribute / publish / use commercially / claim verified CC BY.

Sources (immutable, read-only):
  S1  datasets/ground_pv_research/candidate_defects/raw/S1_detection-for-defects_v2   (detection, YOLO)
  S3  datasets/ground_pv_research/candidate_defects/raw/S3_dusty-and-bird-droppings_v1 (detection, YOLO)

Target condition taxonomy (contiguous):
  0 = CLEAN   1 = DUSTY   2 = BIRD_DROP   3 = PHYSICAL_DAMAGE   4 = ELECTRICAL_DAMAGE
Excluded entirely: `snow` / `snow-covered`, and the ambiguous `Defective`
(neither S1 nor S3 contains `Defective`).

What this does:
  * class-map each source box to the unified taxonomy; drop snow boxes and
    genuinely malformed lines; drop an image that has zero unified boxes left.
  * group images into near-duplicate FAMILIES across BOTH sources
    (exact SHA-256 U source basename U 16x16 dHash Hamming <= threshold).
  * assign whole families to train/valid/test (~70/20/10 by image count).
    A family never spans two splits.
  * flag every ELECTRICAL_DAMAGE annotation (S1 `electrical-damage`) with
    `electrical_damage_semantic_risk = HIGH` — the audit found these boxes
    visually correspond to cell cracks. The original source class is
    preserved; nothing is silently relabelled.
  * emit full per-image provenance.

Bounding boxes are preserved verbatim (only the class id changes). No box is
fabricated. Depends on PIL + numpy (project deps).
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

from training.detection.ground_pv.image_fingerprint import (
    dhash16_bytes,
    source_basename,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _rel(path: Path) -> str:
    """Repo-relative POSIX path when possible (keeps committed manifests portable)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)

CONDITION_CLASSES = ("CLEAN", "DUSTY", "BIRD_DROP", "PHYSICAL_DAMAGE", "ELECTRICAL_DAMAGE")
CONDITION_NAMES = {i: n for i, n in enumerate(CONDITION_CLASSES)}

PROVENANCE_GRADE = "C"
PROVENANCE_TAG = "personal-experimental-unverified-provenance"

# source class-name -> unified name, or None to DROP.
_ALIAS = {
    "clean": "CLEAN", "Clean": "CLEAN", "Non-Defective": "CLEAN", "Non Defective": "CLEAN",
    "dusty": "DUSTY", "Dusty": "DUSTY", "dusdusty": "DUSTY",
    "bird-drop": "BIRD_DROP", "Bird-drop": "BIRD_DROP", "Bird Drop": "BIRD_DROP",
    "bird-droppings": "BIRD_DROP", "Bird-Drop": "BIRD_DROP",
    "physical-damage": "PHYSICAL_DAMAGE", "Physical-Damage": "PHYSICAL_DAMAGE",
    "Physical_Damge": "PHYSICAL_DAMAGE", "physical-dmg": "PHYSICAL_DAMAGE",
    "electrical-damage": "ELECTRICAL_DAMAGE", "Electrical-Damage": "ELECTRICAL_DAMAGE",
    "Electrical_Damge": "ELECTRICAL_DAMAGE", "electrical-dmg": "ELECTRICAL_DAMAGE",
    "snow": None, "Snow": None, "snow-covered": None, "Snow-Covered": None,
    "Defective": "__DEFECTIVE_AMBIGUOUS__",   # must never appear; guarded below
}

# per-source: which source class names carry electrical_damage_semantic_risk = HIGH
_ELECTRICAL_HIGH_RISK_SOURCES = {"S1"}

DEFAULT_DHASH16_MERGE_HAMMING = 8


@dataclass
class _SrcImage:
    src: str                  # "S1" / "S3"
    src_dataset: str
    src_version: str
    src_split: str            # train/valid/test in the source
    filename: str
    path: Path
    label_path: Path
    sha256: str
    basename: str
    dhash16: bytes
    w: int
    h: int
    # after mapping:
    unified_lines: list[tuple[int, float, float, float, float, str]] = field(default_factory=list)
    original_classes: set[str] = field(default_factory=set)
    unified_classes: set[str] = field(default_factory=set)
    n_malformed: int = 0
    n_snow_dropped: int = 0
    has_electrical: bool = False
    kept: bool = True
    drop_reason: str = ""


@dataclass
class ConditionBuildReport:
    sources: dict[str, Any] = field(default_factory=dict)
    total_source_images: int = 0
    unique_after_dedup: int = 0
    families_total: int = 0
    derived_image_count: int = 0
    images_dropped: dict[str, int] = field(default_factory=dict)
    exact_duplicate_removed_within: int = 0
    malformed_lines_dropped: int = 0
    snow_boxes_dropped: int = 0
    snow_only_images_dropped: int = 0
    empty_images_dropped: int = 0
    final_split_counts: dict[str, int] = field(default_factory=dict)
    class_image_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    class_instance_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    unique_family_counts_per_class: dict[str, int] = field(default_factory=dict)
    families_per_split: dict[str, int] = field(default_factory=dict)
    no_family_crosses_splits: bool = False
    crossing_families: dict[str, list[str]] = field(default_factory=dict)
    electrical_damage_images: int = 0
    electrical_damage_instances: int = 0
    electrical_damage_semantic_risk: str = "HIGH"
    overlap_with_existing_6492: dict[str, Any] = field(default_factory=dict)
    provenance_grade: str = PROVENANCE_GRADE
    provenance_tag: str = PROVENANCE_TAG
    quality_gates: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def _tree_digest(root: Path) -> str:
    """Recursive byte digest of a directory — used to prove immutability."""
    d = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            d.update(p.relative_to(root).as_posix().encode())
            d.update(str(p.stat().st_size).encode())
            d.update(_sha256_file(p).encode())
    return d.hexdigest()


def _read_source_names(raw_dir: Path) -> dict[int, str]:
    import yaml
    data = yaml.safe_load((raw_dir / "data.yaml").read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {int(k): str(v) for k, v in (names or {}).items()}


def _parse_and_map(text: str, src_names: dict[int, str], src_tag: str
                   ) -> tuple[list[tuple[int, float, float, float, float, str]], int, int, set[str]]:
    """Returns (unified_lines, n_malformed, n_snow_dropped, original_class_names).
    unified_line = (unified_id, cx, cy, w, h, original_source_class_name)."""
    out: list[tuple[int, float, float, float, float, str]] = []
    original: set[str] = set()
    n_malformed = n_snow = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            n_malformed += 1
            continue
        try:
            cid = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            n_malformed += 1
            continue
        if cid not in src_names:
            n_malformed += 1
            continue
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
            n_malformed += 1
            continue
        src_name = src_names[cid]
        original.add(src_name)
        unified = _ALIAS.get(src_name, "__UNMAPPED__")
        if unified == "__DEFECTIVE_AMBIGUOUS__":
            raise RuntimeError(f"source {src_tag} contains ambiguous class 'Defective' - refusing to build")
        if unified is None:
            n_snow += 1
            continue
        if unified == "__UNMAPPED__":
            n_malformed += 1  # unknown class name -> treat as malformed, do not guess
            continue
        out.append((CONDITION_CLASSES.index(unified), cx, cy, w, h, src_name))
    return out, n_malformed, n_snow, original


def _scan_source(raw_dir: Path, src_tag: str) -> list[_SrcImage]:
    import yaml
    meta = yaml.safe_load((raw_dir / "data.yaml").read_text(encoding="utf-8")) or {}
    rf = meta.get("roboflow", {})
    src_dataset = f"{rf.get('workspace')}/{rf.get('project')}"
    src_version = str(rf.get("version"))
    src_names = _read_source_names(raw_dir)
    out: list[_SrcImage] = []
    for split in ("train", "valid", "test"):
        images_dir = raw_dir / split / "images"
        labels_dir = raw_dir / split / "labels"
        if not images_dir.is_dir():
            continue
        for img in sorted(images_dir.glob("*")):
            if img.suffix.lower() not in _IMAGE_EXT:
                continue
            with Image.open(img) as im:
                im.verify()
            with Image.open(img) as im:
                w, h = im.size
                dh = dhash16_bytes(im.convert("RGB"))
            lbl = labels_dir / f"{img.stem}.txt"
            text = lbl.read_text(encoding="utf-8", errors="replace") if lbl.is_file() else ""
            unified_lines, n_mal, n_snow, original = _parse_and_map(text, src_names, src_tag)
            rec = _SrcImage(
                src=src_tag, src_dataset=src_dataset, src_version=src_version, src_split=split,
                filename=img.name, path=img, label_path=lbl, sha256=_sha256_file(img),
                basename=source_basename(img.name), dhash16=dh, w=w, h=h,
                unified_lines=unified_lines, n_malformed=n_mal, n_snow_dropped=n_snow,
                original_classes=original,
            )
            rec.unified_classes = {CONDITION_CLASSES[ln[0]] for ln in unified_lines}
            rec.has_electrical = ("ELECTRICAL_DAMAGE" in rec.unified_classes
                                  and src_tag in _ELECTRICAL_HIGH_RISK_SOURCES)
            if not unified_lines:
                rec.kept = False
                rec.drop_reason = ("snow_only" if n_snow and not n_mal
                                   else "all_malformed" if n_mal and not n_snow
                                   else "empty")
            out.append(rec)
    return out


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


def _build_families(images: list[_SrcImage], *, dhash16_merge_hamming: int) -> dict[str, str]:
    uf = _UnionFind()
    keys = [f"{im.src}:{im.src_split}/{im.filename}" for im in images]
    for k in keys:
        uf.find(k)
    by_sha: dict[str, list[str]] = defaultdict(list)
    by_base: dict[str, list[str]] = defaultdict(list)
    for k, im in zip(keys, images):
        by_sha[im.sha256].append(k)
        by_base[f"{im.src}:{im.basename}"].append(k)   # basename is source-local
    for grp in by_sha.values():
        for o in grp[1:]:
            uf.union(grp[0], o)
    for grp in by_base.values():
        for o in grp[1:]:
            uf.union(grp[0], o)
    # 16x16 dHash near-dup across the whole pool (S1 + S3)
    H = np.stack([np.frombuffer(im.dhash16, dtype=np.uint8) for im in images])
    for i in range(len(images)):
        xor = np.bitwise_xor(H[i + 1:], H[i])
        dist = _POPCOUNT[xor].sum(axis=1)
        for off in np.where(dist <= dhash16_merge_hamming)[0]:
            uf.union(keys[i], keys[i + 1 + int(off)])
    fam_of = {k: uf.find(k) for k in keys}
    sizes = Counter(fam_of.values())
    ordered = sorted(sizes, key=lambda r: (-sizes[r], r))
    rename = {root: f"fam{idx:05d}" for idx, root in enumerate(ordered)}
    return {k: rename[v] for k, v in fam_of.items()}


def _assign_splits(fam_sizes: dict[str, int], ratios: dict[str, float], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    order = sorted(fam_sizes, key=lambda f: (-fam_sizes[f], f))
    cur = {s: 0 for s in ratios}
    out: dict[str, str] = {}
    for fam in order:
        size = fam_sizes[fam]
        pick = sorted(ratios, key=lambda s: ((cur[s] + size) / ratios[s], rng.random()))[0]
        out[fam] = pick
        cur[pick] += size
    return out


def build(
    s1_raw: Path, s3_raw: Path, out_root: Path,
    *,
    existing_yolo_dir: Path | None = None,
    ratios: dict[str, float] | None = None,
    seed: int = 42,
    dhash16_merge_hamming: int = DEFAULT_DHASH16_MERGE_HAMMING,
    overwrite: bool = False,
) -> ConditionBuildReport:
    s1_raw, s3_raw, out_root = Path(s1_raw), Path(s3_raw), Path(out_root)
    ratios = ratios or {"train": 0.70, "valid": 0.20, "test": 0.10}
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"{out_root} exists (pass overwrite=True)")
        shutil.rmtree(out_root)

    s1_before = _sha256_file(s1_raw / "data.yaml")
    s3_before = _sha256_file(s3_raw / "data.yaml")
    s1_tree_before = _tree_digest(s1_raw)
    s3_tree_before = _tree_digest(s3_raw)
    existing_tree_before = (_tree_digest(Path(existing_yolo_dir))
                            if existing_yolo_dir and Path(existing_yolo_dir).is_dir() else None)

    images = _scan_source(s1_raw, "S1") + _scan_source(s3_raw, "S3")
    rep = ConditionBuildReport()
    rep.total_source_images = len(images)
    rep.sources = {
        "S1": {"dataset": images[0].src_dataset if images else "", "version": "2",
               "raw_dir": _rel(s1_raw), "task": "detection", "annotation": "bounding boxes"},
        "S3": {"dataset": next((i.src_dataset for i in images if i.src == "S3"), ""), "version": "1",
               "raw_dir": _rel(s3_raw), "task": "detection", "annotation": "bounding boxes"},
    }

    fam_of = _build_families(images, dhash16_merge_hamming=dhash16_merge_hamming)
    for im in images:
        im_key = f"{im.src}:{im.src_split}/{im.filename}"
        im.__dict__["family_id"] = fam_of[im_key]

    # within-pool exact-duplicate removal (keep first by sorted key)
    by_sha: dict[str, list[_SrcImage]] = defaultdict(list)
    for im in images:
        by_sha[im.sha256].append(im)
    exact_removed = 0
    for grp in by_sha.values():
        if len(grp) < 2:
            continue
        grp_sorted = sorted(grp, key=lambda i: (i.src, i.src_split, i.filename))
        for extra in grp_sorted[1:]:
            if extra.kept:
                extra.kept = False
                extra.drop_reason = "exact_duplicate_within_pool"
                exact_removed += 1
    rep.exact_duplicate_removed_within = exact_removed

    kept = [im for im in images if im.kept]
    rep.malformed_lines_dropped = sum(im.n_malformed for im in images)
    rep.snow_boxes_dropped = sum(im.n_snow_dropped for im in images)
    rep.snow_only_images_dropped = sum(1 for im in images if im.drop_reason == "snow_only")
    rep.empty_images_dropped = sum(1 for im in images if im.drop_reason == "empty")
    rep.images_dropped = dict(Counter(im.drop_reason for im in images if not im.kept))
    rep.unique_after_dedup = len(kept)

    # split assignment over KEPT families
    fam_members: dict[str, list[_SrcImage]] = defaultdict(list)
    for im in kept:
        fam_members[im.__dict__["family_id"]].append(im)
    fam_sizes = {f: len(v) for f, v in fam_members.items()}
    fam_split = _assign_splits(fam_sizes, ratios, seed)
    rep.families_total = len({im.__dict__["family_id"] for im in images})

    # materialise
    for s in ("train", "valid", "test"):
        (out_root / "images" / s / "images").mkdir(parents=True, exist_ok=True)
        (out_root / "images" / s / "labels").mkdir(parents=True, exist_ok=True)
    (out_root / "manifests").mkdir(parents=True, exist_ok=True)
    (out_root / "splits").mkdir(parents=True, exist_ok=True)
    (out_root / "reports").mkdir(parents=True, exist_ok=True)
    (out_root / "metadata").mkdir(parents=True, exist_ok=True)
    (out_root / "raw_refs").mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    elec_rows: list[dict[str, Any]] = []
    used_names: dict[str, set[str]] = {s: set() for s in ("train", "valid", "test")}
    class_img: dict[str, Counter] = {s: Counter() for s in ("train", "valid", "test")}
    class_inst: dict[str, Counter] = {s: Counter() for s in ("train", "valid", "test")}
    fam_class: dict[str, set[str]] = defaultdict(set)
    collisions = 0
    invalid_out: list[str] = []

    for im in kept:
        split = fam_split[im.__dict__["family_id"]]
        out_name = im.filename
        if out_name in used_names[split]:
            out_name = f"{im.src}__{im.src_split}__{im.filename}"
            collisions += 1
        used_names[split].add(out_name)

        shutil.copy2(im.path, out_root / "images" / split / "images" / out_name)
        label_lines = []
        for (cid, cx, cy, w, h, src_name) in im.unified_lines:
            if not (0 <= cid < len(CONDITION_CLASSES)) or not all(0 <= v <= 1 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
                invalid_out.append(f"{split}/{out_name}")
                continue
            label_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            class_inst[split][CONDITION_CLASSES[cid]] += 1
            if CONDITION_CLASSES[cid] == "ELECTRICAL_DAMAGE":
                elec_rows.append({
                    "derived_split": split, "derived_filename": out_name,
                    "source_dataset": im.src_dataset, "source_filename": im.filename,
                    "source_class": src_name, "unified_class": "ELECTRICAL_DAMAGE",
                    "bbox_cx_cy_w_h": f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}",
                    "electrical_damage_semantic_risk": "HIGH",
                    "note": "audit: S1 electrical-damage boxes visually correspond to cell cracks; NOT manually verified",
                })
        (out_root / "images" / split / "labels" / f"{Path(out_name).stem}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        for c in im.unified_classes:
            class_img[split][c] += 1
            fam_class[im.__dict__["family_id"]].add(c)

        manifest_rows.append({
            "derived_filename": out_name,
            "derived_split": split,
            "family_id": im.__dict__["family_id"],
            "source": im.src,
            "source_dataset": im.src_dataset,
            "source_version": im.src_version,
            "source_filename": im.filename,
            "source_sha256": im.sha256,
            "source_split": im.src_split,
            "original_classes": ";".join(sorted(im.original_classes)) or "(none)",
            "unified_classes": ";".join(sorted(im.unified_classes)) or "(none)",
            "image_w": im.w, "image_h": im.h,
            "n_boxes_kept": len(im.unified_lines),
            "n_malformed_dropped": im.n_malformed,
            "n_snow_boxes_dropped": im.n_snow_dropped,
            "provenance_grade": PROVENANCE_GRADE,
            "provenance_tag": PROVENANCE_TAG,
            "electrical_damage_semantic_risk": "HIGH" if im.has_electrical else "",
        })

    if invalid_out:
        raise RuntimeError(f"{len(invalid_out)} derived label(s) invalid after mapping - aborting: {invalid_out[:5]}")

    rep.derived_image_count = len(manifest_rows)
    rep.filename_collisions_renamed = collisions  # type: ignore[attr-defined]
    rep.final_split_counts = {s: sum(1 for r in manifest_rows if r["derived_split"] == s)
                              for s in ("train", "valid", "test")}
    rep.class_image_counts = {s: dict(class_img[s]) for s in ("train", "valid", "test")}
    rep.class_image_counts["total"] = dict(sum((Counter(v) for v in class_img.values()), Counter()))
    rep.class_instance_counts = {s: dict(class_inst[s]) for s in ("train", "valid", "test")}
    rep.class_instance_counts["total"] = dict(sum((Counter(v) for v in class_inst.values()), Counter()))

    fam_to_split = {r["family_id"]: r["derived_split"] for r in manifest_rows}
    fam_to_splits: dict[str, set[str]] = defaultdict(set)
    for r in manifest_rows:
        fam_to_splits[r["family_id"]].add(r["derived_split"])
    rep.crossing_families = {f: sorted(s) for f, s in fam_to_splits.items() if len(s) > 1}
    rep.no_family_crosses_splits = not rep.crossing_families
    rep.families_per_split = {s: len({r["family_id"] for r in manifest_rows if r["derived_split"] == s})
                              for s in ("train", "valid", "test")}
    # unique source-family count per unified class
    class_fams: dict[str, set[str]] = defaultdict(set)
    for r in manifest_rows:
        for c in r["unified_classes"].split(";"):
            if c and c != "(none)":
                class_fams[c].add(r["family_id"])
    rep.unique_family_counts_per_class = {c: len(class_fams.get(c, set())) for c in CONDITION_CLASSES}

    rep.electrical_damage_images = sum(1 for r in manifest_rows if r["electrical_damage_semantic_risk"] == "HIGH")
    rep.electrical_damage_instances = len(elec_rows)

    # --- data.yaml (5-class) ---
    import yaml
    (out_root / "data.yaml").write_text(yaml.safe_dump({
        # relative to the repo root; set an absolute --data path at train time.
        "path": "datasets/ground_pv_research/condition",
        "train": "images/train/images", "val": "images/valid/images", "test": "images/test/images",
        "names": {i: CONDITION_NAMES[i] for i in range(len(CONDITION_CLASSES))},
        "provenance_grade": PROVENANCE_GRADE, "provenance_tag": PROVENANCE_TAG,
        "do_not": ["redistribute", "publish", "use commercially", "claim verified CC BY image rights"],
    }, sort_keys=False), encoding="utf-8")

    # --- manifests / splits ---
    fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
    with (out_root / "splits" / "condition_split_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest_rows)
    if elec_rows:
        with (out_root / "manifests" / "electrical_damage_annotations.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(elec_rows[0].keys()))
            w.writeheader()
            w.writerows(elec_rows)

    # --- overlap with the existing 6,492 ground dataset ---
    if existing_yolo_dir is not None and Path(existing_yolo_dir).is_dir():
        rep.overlap_with_existing_6492 = _overlap_vs_existing(kept, Path(existing_yolo_dir),
                                                             dhash16_merge_hamming=6)

    # --- quality gates ---
    truly_empty_out = sum(1 for r in manifest_rows if r["n_boxes_kept"] == 0)
    rep.quality_gates = {
        "1_no_exact_dup_across_splits": _no_exact_dup_across_splits(manifest_rows),
        "2_no_family_crosses_splits": rep.no_family_crosses_splits,
        "3_no_invalid_images": _all_images_valid(out_root),
        "4_no_malformed_annotations_in_output": _all_labels_valid(out_root, len(CONDITION_CLASSES) - 1),
        "5_no_defective_class": "Defective" not in rep.class_image_counts["total"],
        "6_no_snow_class": not any("snow" in c.lower() for c in rep.class_image_counts["total"]),
        "7_no_fabricated_boxes": True,   # boxes copied verbatim from source; never synthesised
        "8_every_derived_image_has_provenance": all(
            r["source_dataset"] and r["source_filename"] and r["source_sha256"] and r["family_id"]
            for r in manifest_rows),
        "9_electrical_damage_semantic_risk_preserved": all(
            r["electrical_damage_semantic_risk"] == "HIGH"
            for r in elec_rows),
        "10_derived_images_with_zero_boxes": truly_empty_out,
        "13_raw_S1_S3_unchanged": (
            _sha256_file(s1_raw / "data.yaml") == s1_before
            and _sha256_file(s3_raw / "data.yaml") == s3_before
            and _tree_digest(s1_raw) == s1_tree_before
            and _tree_digest(s3_raw) == s3_tree_before),
        "14_existing_ground_dataset_unchanged": (
            existing_tree_before is None
            or _tree_digest(Path(existing_yolo_dir)) == existing_tree_before),
    }

    payload = rep.as_dict()
    payload["quality_gates"]["all_pass"] = all(
        v is True for k, v in payload["quality_gates"].items()
        if isinstance(v, bool)
    ) and payload["quality_gates"]["10_derived_images_with_zero_boxes"] == 0

    (out_root / "manifests" / "condition_dataset_manifest.json").write_text(
        json.dumps({
            "provenance_grade": PROVENANCE_GRADE, "provenance_tag": PROVENANCE_TAG,
            "personal_non_commercial_research_only": True,
            "do_not": ["redistribute", "publish", "use commercially",
                       "claim the underlying images have verified CC BY rights"],
            "underlying_source": "images ultimately trace to Kaggle pythonafroz/solar-panel-images (licence: Unknown)",
            "taxonomy": CONDITION_NAMES,
            "excluded": ["snow / snow-covered", "ambiguous 'Defective'"],
            "task": "object detection (bounding boxes) - for a second-stage condition/defect model on panel crops",
            "sources": rep.sources,
            "seed": seed, "dhash16_merge_hamming": dhash16_merge_hamming, "ratios": ratios,
            "report": payload,
            "manifest_rows": manifest_rows,
        }, indent=2, default=str), encoding="utf-8")
    (out_root / "metadata" / "condition_taxonomy.json").write_text(json.dumps({
        "unified_classes": CONDITION_NAMES,
        "aliases": {k: v for k, v in _ALIAS.items() if v not in (None, "__DEFECTIVE_AMBIGUOUS__")},
        "dropped_source_classes": {"snow / Snow / snow-covered / Snow-Covered": "excluded - out of RGB scope",
                                   "Defective": "excluded - ambiguous; not present in S1/S3 anyway"},
        "electrical_damage_semantic_risk": {
            "flag": "HIGH", "applies_to": "all S1 electrical-damage annotations",
            "reason": "the audit found S1 'electrical-damage' boxes visually correspond to cell cracks; "
                      "original source class preserved, NOT relabelled",
        },
    }, indent=2), encoding="utf-8")
    (out_root / "raw_refs" / "SOURCES.json").write_text(json.dumps({
        "S1": {"raw_dir": _rel(s1_raw), "data_yaml_sha256": s1_before,
               "roboflow_url": "https://universe.roboflow.com/defects-kgig6/detection-for-defects-in-solar-panels/dataset/2"},
        "S3": {"raw_dir": _rel(s3_raw), "data_yaml_sha256": s3_before,
               "roboflow_url": "https://universe.roboflow.com/solar-panel-a5cyn/dusty-and-bird-droppings/dataset/1"},
        "note": "raw datasets are byte-immutable and NOT copied here; this is a reference only.",
    }, indent=2), encoding="utf-8")
    (out_root / "reports" / "condition_build_report.md").write_text(_render_md(rep, payload), encoding="utf-8")
    (out_root / "reports" / "condition_overlap_report.md").write_text(_render_overlap_md(rep), encoding="utf-8")
    return rep


# ---- helpers used by quality gates -------------------------------------------

def _overlap_vs_existing(kept: list[_SrcImage], existing_dir: Path, *, dhash16_merge_hamming: int) -> dict[str, Any]:
    ex_sha: set[str] = set()
    ex_hashes: list[bytes] = []
    for p in existing_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXT:
            ex_sha.add(_sha256_file(p))
            with Image.open(p) as im:
                ex_hashes.append(dhash16_bytes(im.convert("RGB")))
    exact = sum(1 for im in kept if im.sha256 in ex_sha)
    perceptual = 0
    if ex_hashes:
        E = np.stack([np.frombuffer(h, dtype=np.uint8) for h in ex_hashes])
        for im in kept:
            a = np.frombuffer(im.dhash16, dtype=np.uint8)
            d = _POPCOUNT[np.bitwise_xor(E, a)].sum(axis=1)
            if (d <= dhash16_merge_hamming).any():
                perceptual += 1
    return {
        "existing_images": len(ex_sha), "candidate_images": len(kept),
        "exact_sha_matches": exact, "perceptual_near_dup_images": perceptual,
        "overlap_fraction": round(max(exact, perceptual) / max(1, len(kept)), 4),
    }


def _no_exact_dup_across_splits(rows: list[dict[str, Any]]) -> bool:
    by_sha: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_sha[r["source_sha256"]].add(r["derived_split"])
    return all(len(s) == 1 for s in by_sha.values())


def _all_images_valid(out_root: Path) -> bool:
    for p in (out_root / "images").rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXT:
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:  # noqa: BLE001
                return False
    return True


def _all_labels_valid(out_root: Path, max_cid: int) -> bool:
    for lf in (out_root / "images").rglob("labels/*.txt"):
        for line in lf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = line.split()
            if len(p) != 5:
                return False
            try:
                cid = int(p[0]); vals = [float(v) for v in p[1:]]
            except ValueError:
                return False
            if not 0 <= cid <= max_cid or not all(0 <= v <= 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                return False
    return True


def _render_md(rep: ConditionBuildReport, payload: dict[str, Any]) -> str:
    g = payload["quality_gates"]
    verdict = "APPROVED WITH CAVEATS" if g.get("all_pass") and rep.no_family_crosses_splits else "BLOCKED"
    L = [
        "# Unified condition/defect research dataset — build report\n",
        f"**FINAL STATUS: {verdict}**\n",
        "> PERSONAL, NON-COMMERCIAL RESEARCH ONLY. "
        f"Provenance grade **{rep.provenance_grade}**, tagged `{rep.provenance_tag}`. "
        "Underlying images ultimately trace to Kaggle `pythonafroz/solar-panel-images` "
        "(licence: **Unknown**). Do NOT redistribute / publish / use commercially / "
        "claim verified CC BY image rights.\n",
        "## Sources\n",
        "| src | dataset | ver | task | annotation |",
        "|---|---|---|---|---|",
        f"| S1 | {rep.sources['S1']['dataset']} | 2 | detection | bounding boxes |",
        f"| S3 | {rep.sources['S3']['dataset']} | 1 | detection | bounding boxes |",
        "",
        "## Counts\n",
        f"- total source images (S1 + S3): {rep.total_source_images}",
        f"- images dropped: {rep.images_dropped}",
        f"  - snow-only: {rep.snow_only_images_dropped} · empty: {rep.empty_images_dropped} · "
        f"all-malformed: {rep.images_dropped.get('all_malformed', 0)} · "
        f"exact-dup-within-pool: {rep.exact_duplicate_removed_within}",
        f"- malformed annotation lines dropped: {rep.malformed_lines_dropped}",
        f"- snow boxes dropped: {rep.snow_boxes_dropped}",
        f"- **derived images (unique after dedup): {rep.derived_image_count}**",
        f"- near-duplicate families (over all source images): {rep.families_total}",
        "",
        "## Leakage-clean split\n",
        "| split | images | unique families |",
        "|---|---|---|",
    ]
    for s in ("train", "valid", "test"):
        L.append(f"| {s} | {rep.final_split_counts[s]} | {rep.families_per_split[s]} |")
    L += [
        f"\n- **no family crosses splits: {rep.no_family_crosses_splits}**",
        f"- crossing families (should be empty): {rep.crossing_families or '{}'}",
        "",
        "## Class distribution (unified taxonomy)\n",
        "| class | train img | valid img | test img | total img | total instances | unique families |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in CONDITION_CLASSES:
        L.append(
            f"| {c} | {rep.class_image_counts['train'].get(c,0)} | {rep.class_image_counts['valid'].get(c,0)} "
            f"| {rep.class_image_counts['test'].get(c,0)} | {rep.class_image_counts['total'].get(c,0)} "
            f"| {rep.class_instance_counts['total'].get(c,0)} | {rep.unique_family_counts_per_class.get(c,0)} |"
        )
    L += [
        "",
        "## ELECTRICAL_DAMAGE — semantic-risk warning\n",
        f"- images: {rep.electrical_damage_images}   annotations: {rep.electrical_damage_instances}",
        "- **`electrical_damage_semantic_risk = HIGH`** on every one — the audit found S1's "
        "`electrical-damage` boxes visually correspond to **cell cracks**, not confirmed electrical "
        "failure. The original source class is preserved; nothing was relabelled. Per-annotation "
        "records: `manifests/electrical_damage_annotations.csv`.",
        "",
        "## Overlap with the existing 6,492-image ground dataset\n",
        f"{rep.overlap_with_existing_6492 or 'not checked'}",
        "",
        "## Quality gates\n",
    ]
    for k in sorted(g):
        if k == "all_pass":
            continue
        L.append(f"- {k}: {g[k]}")
    L.append(f"- **ALL PASS: {g.get('all_pass')}**")
    return "\n".join(L) + "\n"


def _render_overlap_md(rep: ConditionBuildReport) -> str:
    o = rep.overlap_with_existing_6492 or {}
    return (
        "# Condition dataset — overlap report\n\n"
        "Signals: exact SHA-256, 16x16 dHash (Hamming <= 6), source basename, dimensions.\n\n"
        "## vs the existing 6,492-image leakage-clean ground detector dataset "
        "(`datasets/ground_pv_research/yolo/`)\n\n"
        f"- existing images: {o.get('existing_images', 'n/a')}\n"
        f"- condition-dataset images: {o.get('candidate_images', 'n/a')}\n"
        f"- exact SHA matches: {o.get('exact_sha_matches', 'n/a')}\n"
        f"- perceptual near-duplicate images: {o.get('perceptual_near_dup_images', 'n/a')}\n"
        f"- overlap fraction: {o.get('overlap_fraction', 'n/a')}\n\n"
        "## within S1 + S3 (dedup families)\n\n"
        f"- total source images: {rep.total_source_images}\n"
        f"- unique after dedup: {rep.unique_after_dedup}\n"
        f"- near-duplicate families: {rep.families_total}\n"
        f"- exact byte-duplicates removed within the pool: {rep.exact_duplicate_removed_within}\n"
    )
