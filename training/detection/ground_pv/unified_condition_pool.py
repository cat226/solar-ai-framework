"""training/detection/ground_pv/unified_condition_pool.py — pure helpers for
building ONE deduplicated, family-split 4-class condition pool out of several
public RGB sources that are known to re-ship the same photos
(Faulty_solar_panel / the Roboflow ahmed-tamer re-cut / the suhasmaddali
GitHub repo — see the Oct-2026 dataset analyses).

The unit of independence is the content FAMILY: a set of images whose
orientation-invariant perceptual hash is within ``max_dist`` bits of each
other (flip / rot90 augmentation copies, byte-identical re-packs, and the same
photo shipped by two sources all collapse into one family). Splits are made
per family so a photo and its near-twins can never straddle train/valid/test.

No torch / ultralytics dependency (PIL + numpy only), matching the project's
convention for taxonomy / pool helpers.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from training.detection.ground_pv.condition_taxonomy import (
    CONDITION_4CLASS_IDS,
    REJECTED_CLASSES,
)

SCHEMA_VERSION = "1.0"
MIN_SIDE_PX = 128  # below this a crop carries too little signal for a 224px classifier
DEFAULT_MAX_DIST = 4  # bits of 64 — matches the Oct-2026 source analyses

# Lineages (coarse provenance of an image). phone2021_macro is the Roboflow
# single-panel macro photo session: a different domain from a rooftop camera.
LINEAGE_FAULTY = "faulty_solar_panel"
LINEAGE_STOCK_WEB = "imgclean_imgdirty_web"
LINEAGE_PHONE_MACRO = "phone2021_macro"
LINEAGE_OTHER = "other"
DEFAULT_EXCLUDED_LINEAGES = frozenset({LINEAGE_PHONE_MACRO})

# Faulty_solar_panel folder -> active class (None = deliberately excluded).
FAULTY_FOLDER_TO_CLASS: dict[str, str | None] = {
    "Clean": "CLEAN",
    "Dusty": "DUSTY",
    "Bird-drop": "BIRD_DROP",
    "Physical-Damage": "PHYSICAL_DAMAGE",
    "Electrical-damage": None,  # REJECTED taxonomy class (semantic risk)
    "Snow-Covered": None,       # REJECTED taxonomy class (out of scope)
}
BINARY_FOLDER_TO_CLASS = {"Clean": "CLEAN", "Dusty": "DUSTY"}

# Representative preference: lower wins. Original-looking sources first.
SOURCE_PRIORITY = {"faulty": 0, "github_raw": 1, "github_split": 2, "roboflow_eval": 3, "roboflow_train": 4}


@dataclass
class Rec:
    path: str
    source: str            # faulty | github_raw | github_split | roboflow_eval | roboflow_train
    source_class: str      # folder label in the source
    label: str | None      # active class, or None if deliberately excluded
    lineage: str
    w: int = 0
    h: int = 0
    sha256: str = ""
    chash: int = 0
    family: int = -1
    exclude_reason: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def min_side(self) -> int:
        return min(self.w, self.h)


# ---------------------------------------------------------------------------
# hashing / families
# ---------------------------------------------------------------------------

def canon_hash(image: Image.Image) -> int:
    """64-bit difference hash, invariant to the 8 flip/rot90 variants (min over
    orientations). Photos that differ only by Roboflow-style flips / 90-degree
    rotations therefore hash identically."""
    g = ImageOps.exif_transpose(image).convert("L").resize((9, 9), Image.LANCZOS)
    a = np.asarray(g, dtype=np.float32)
    best: int | None = None
    for k in range(4):
        for flip in (False, True):
            b = np.rot90(a, k)
            b = b[:, ::-1] if flip else b
            bits = (b[:8, 1:9] > b[:8, :8]).flatten()
            v = int("".join("1" if x else "0" for x in bits), 2)
            best = v if best is None or v < best else best
    return int(best)  # type: ignore[arg-type]


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def group_families(hashes: list[int], max_dist: int = DEFAULT_MAX_DIST) -> list[int]:
    """Union-find clustering: items whose hashes are within ``max_dist`` bits
    are one family. Returns a family id per input index (ids are the smallest
    member index of the cluster, so they are deterministic)."""
    n = len(hashes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if n:
        # unique hashes first: identical hashes are free unions
        first_idx: dict[int, int] = {}
        for i, h in enumerate(hashes):
            if h in first_idx:
                parent[find(i)] = find(first_idx[h])
            else:
                first_idx[h] = i
        uniq = list(first_idx.items())  # (hash, idx)
        arr = np.array([h for h, _ in uniq], dtype=np.uint64)
        lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        for i in range(len(uniq)):
            xor = np.bitwise_xor(arr[i + 1:], arr[i])
            if xor.size == 0:
                continue
            pop = lut[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
            for j in np.nonzero(pop <= max_dist)[0]:
                parent[find(uniq[i][1])] = find(uniq[i + 1 + int(j)][1])
    roots: dict[int, int] = {}
    out = []
    for i in range(n):
        out.append(roots.setdefault(find(i), len(roots)))
    return out


# ---------------------------------------------------------------------------
# labels / lineage
# ---------------------------------------------------------------------------

def faulty_label(folder: str) -> str | None:
    if folder not in FAULTY_FOLDER_TO_CLASS:
        raise KeyError(f"unknown Faulty_solar_panel folder: {folder!r}")
    label = FAULTY_FOLDER_TO_CLASS[folder]
    assert label is None or label in CONDITION_4CLASS_IDS
    return label


def binary_label(folder: str) -> str | None:
    return BINARY_FOLDER_TO_CLASS.get(folder)


_PHONE_RE = re.compile(r"^\d{8}_\d{6}")


def lineage_of(source: str, filename: str) -> str:
    if source == "faulty":
        return LINEAGE_FAULTY
    n = filename.lower()
    if n.startswith(("imgclean", "imgdirty")):
        return LINEAGE_STOCK_WEB
    if _PHONE_RE.match(filename):
        return LINEAGE_PHONE_MACRO
    return LINEAGE_OTHER


def roboflow_base(filename: str) -> str:
    m = re.match(r"(.+?)_(?:jpg|jpeg|png)\.rf\.[0-9a-f]+\.", filename, re.I)
    return m.group(1) if m else Path(filename).stem


# ---------------------------------------------------------------------------
# family resolution
# ---------------------------------------------------------------------------

def _rep_key(r: Rec) -> tuple:
    return (SOURCE_PRIORITY.get(r.source, 9), -(r.w * r.h), r.path)


_FINE_CLASSES = frozenset({"BIRD_DROP", "PHYSICAL_DAMAGE"})


def _coarse_to_fine(members: list[Rec], labels: set[str]) -> str | None:
    """The binary Clean/Dusty sources only say "dirty"; the 4-class source
    (Faulty_solar_panel) says WHAT kind. A photo labelled DUSTY by a binary
    source and BIRD_DROP / PHYSICAL_DAMAGE by Faulty is not a contradiction -
    the finer label wins. Anything else (incl. every conflict involving
    CLEAN, or two fine classes disagreeing) stays a real conflict -> None."""
    if len(labels) != 2 or "DUSTY" not in labels:
        return None
    (fine,) = labels - {"DUSTY"}
    if fine not in _FINE_CLASSES:
        return None
    fine_srcs = {m.source for m in members if m.label == fine}
    dusty_srcs = {m.source for m in members if m.label == "DUSTY"}
    if fine_srcs == {"faulty"} and "faulty" not in dusty_srcs:
        return fine
    return None


def resolve_families(recs: list[Rec], excluded_lineages: frozenset[str] = DEFAULT_EXCLUDED_LINEAGES):
    """Group recs (all must have .family set) and decide, per family, the
    outcome. Returns (families, dropped) where ``families`` maps family id ->
    dict(label, rep, members, lineage) for kept families and ``dropped`` maps
    family id -> reason. Rules, in order:

    1. any member deliberately excluded (Electrical / Snow) AND any member in an
       active class                      -> 'conflict_with_excluded_class'
    2. members disagree on the active class -> 'label_conflict', EXCEPT the
       coarse-to-fine case (binary-source DUSTY vs Faulty BIRD_DROP /
       PHYSICAL_DAMAGE) which resolves to the finer Faulty label
    3. family has only excluded-class members -> 'excluded_class'
    4. no member >= MIN_SIDE_PX            -> 'too_small'
    5. representative's lineage excluded    -> 'excluded_lineage:<name>'
    """
    by_fam: dict[int, list[Rec]] = defaultdict(list)
    for r in recs:
        by_fam[r.family].append(r)
    kept: dict[int, dict] = {}
    dropped: dict[int, str] = {}
    for fam, members in by_fam.items():
        labels = {m.label for m in members if m.label is not None}
        has_excluded = any(m.label is None for m in members)
        if labels and has_excluded:
            dropped[fam] = "conflict_with_excluded_class"
            continue
        override = False
        if len(labels) > 1:
            fine = _coarse_to_fine(members, labels)
            if fine is None:
                dropped[fam] = "label_conflict"
                continue
            labels, override = {fine}, True
        if not labels:
            dropped[fam] = "excluded_class"
            continue
        big = [m for m in members if m.min_side >= MIN_SIDE_PX]
        if not big:
            dropped[fam] = "too_small"
            continue
        rep = sorted(big, key=_rep_key)[0]
        if rep.lineage in excluded_lineages:
            dropped[fam] = f"excluded_lineage:{rep.lineage}"
            continue
        kept[fam] = {"label": next(iter(labels)), "rep": rep, "members": members, "lineage": rep.lineage,
                     "coarse_to_fine_override": override}
    return kept, dropped


def assign_splits(fam_labels: dict[int, str], *, seed: int = 42,
                  fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> dict[int, str]:
    """Family-level stratified split. Within each class the families are
    shuffled (seeded) and cut by ``fractions``; a class with >= 3 families
    always gets at least one family in every split."""
    assert abs(sum(fractions) - 1.0) < 1e-9
    rng = random.Random(seed)
    out: dict[int, str] = {}
    by_cls: dict[str, list[int]] = defaultdict(list)
    for fam, cls in sorted(fam_labels.items()):
        by_cls[cls].append(fam)
    for cls in sorted(by_cls):
        fams = by_cls[cls]
        rng.shuffle(fams)
        n = len(fams)
        n_val = round(n * fractions[1])
        n_test = round(n * fractions[2])
        if n >= 3:
            n_val, n_test = max(1, n_val), max(1, n_test)
        n_train = n - n_val - n_test
        for i, fam in enumerate(fams):
            out[fam] = "train" if i < n_train else ("valid" if i < n_train + n_val else "test")
    return out


def assert_no_family_straddle(rows: list[dict]) -> None:
    """Fail closed if any family appears in more than one split."""
    seen: dict[int, str] = {}
    for r in rows:
        if not r.get("split"):
            continue
        prev = seen.setdefault(r["family"], r["split"])
        if prev != r["split"]:
            raise AssertionError(f"family {r['family']} appears in both {prev} and {r['split']}")


def assert_active_labels(labels) -> None:
    bad = [x for x in set(labels) if x not in CONDITION_4CLASS_IDS or x in REJECTED_CLASSES]
    if bad:
        raise ValueError(f"non-active class labels in pool: {bad}")
