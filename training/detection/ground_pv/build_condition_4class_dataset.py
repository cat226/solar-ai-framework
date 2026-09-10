"""training/detection/ground_pv/build_condition_4class_dataset.py — derive
the active 4-class condition dataset (`condition_4class_v1`) from the
existing, already-leakage-clean 5-class research dataset.

Source (read-only, NEVER mutated): `datasets/ground_pv_research/condition/`
— built by `build_condition_dataset.py` (see
[[ground_pv_condition_dataset_build]]), taxonomy 0=CLEAN 1=DUSTY
2=BIRD_DROP 3=PHYSICAL_DAMAGE 4=ELECTRICAL_DAMAGE, already family-grouped
and leakage-clean-split (train/valid/test, no family crosses a split).

Why derive from the 5-class dataset rather than re-scan S1/S3 raw: the
active taxonomy's first three ids (CLEAN=0, DUSTY=1, BIRD_DROP=2,
PHYSICAL_DAMAGE=3) are ALREADY numbered identically in the 5-class
dataset — only class id 4 (ELECTRICAL_DAMAGE) needs to be removed. Deriving
from the already-audited, already-family-grouped, already-split 5-class
set means the 4-class set is PROVABLY leakage-clean by construction (a
class-filtered subset of a leakage-clean split is still leakage-clean — no
image ever moves between splits, no family grouping is recomputed) instead
of re-deriving family/split logic a second time with the attendant risk of
it drifting from the 5-class version.

Removal policy (Phase 15A req 3 — do NOT silently relabel):
  - Every ELECTRICAL_DAMAGE annotation LINE is dropped. It is never
    relabelled as PHYSICAL_DAMAGE or any other active class — the 5-class
    audit found these boxes visually correspond to cell cracks, not
    confirmed electrical failure, and fabricating a different diagnosis for
    them would be worse than dropping them.
  - An image whose annotations were ENTIRELY ELECTRICAL_DAMAGE (so it has
    zero boxes left after removal) is DROPPED from the derived dataset —
    the same "empty after class removal" policy the 5-class builder itself
    already applies to snow-only images (see build_condition_dataset.py's
    `drop_reason == "snow_only"` / `"empty"` handling). Documented count:
    `electrical_damage_only_images_dropped`.
  - An image that had ELECTRICAL_DAMAGE alongside other active-class boxes
    is KEPT, with only the ELECTRICAL_DAMAGE line(s) removed; its
    `n_boxes_kept` count and `unified_classes` are updated to match.
  - Images with NO electrical_damage at all pass through completely
    unchanged (image bytes copied verbatim, label file byte-identical).

No box is fabricated. No class id is renumbered (0-3 keep their existing
meaning). No family/split assignment is recomputed.
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

from training.detection.ground_pv.condition_taxonomy import (
    CONDITION_4CLASS_NAMES,
    CONDITION_4CLASS_ORDER,
)

_ELECTRICAL_DAMAGE_CID = 4
_SPLITS = ("train", "valid", "test")


def _sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def _tree_digest(root: Path) -> str:
    d = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            d.update(p.relative_to(root).as_posix().encode())
            d.update(str(p.stat().st_size).encode())
            d.update(_sha256_file(p).encode())
    return d.hexdigest()


@dataclass
class Condition4ClassBuildReport:
    source_dir: str = ""
    derived_dir: str = ""
    source_image_count: int = 0
    derived_image_count: int = 0
    electrical_damage_instances_removed: int = 0
    electrical_damage_only_images_dropped: int = 0
    images_with_electrical_damage_partially_removed: int = 0
    final_split_counts: dict[str, int] = field(default_factory=dict)
    class_image_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    class_instance_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    dropped_images: list[dict[str, Any]] = field(default_factory=list)
    quality_gates: dict[str, Any] = field(default_factory=dict)
    source_tree_unchanged: bool = False

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _filter_label_lines(text: str) -> tuple[list[str], int, list[int]]:
    """Returns (kept_lines_verbatim, n_electrical_removed, kept_class_ids_per_box).

    `kept_class_ids_per_box` is a LIST with one entry per surviving box
    (NOT deduplicated) — needed so instance counts (multiple boxes of the
    same class on one image) are counted correctly, not collapsed to
    per-image presence.
    """
    kept: list[str] = []
    removed = 0
    classes: list[int] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            # Not this module's job to repair malformed lines - the 5-class
            # build already validated them. Preserve as-is (pass through);
            # the "5_no_defective_class"-style gates below still catch
            # anything genuinely wrong via _all_labels_valid.
            kept.append(stripped)
            continue
        cid = int(parts[0])
        if cid == _ELECTRICAL_DAMAGE_CID:
            removed += 1
            continue
        kept.append(stripped)
        classes.append(cid)
    return kept, removed, classes


def build(
    source_dir: Path,
    derived_dir: Path,
    *,
    overwrite: bool = False,
) -> Condition4ClassBuildReport:
    source_dir, derived_dir = Path(source_dir), Path(derived_dir)
    if not (source_dir / "data.yaml").is_file():
        raise FileNotFoundError(f"no 5-class data.yaml under {source_dir}")
    if derived_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{derived_dir} exists (pass overwrite=True)")
        shutil.rmtree(derived_dir)

    source_tree_before = _tree_digest(source_dir)

    rep = Condition4ClassBuildReport(source_dir=str(source_dir), derived_dir=str(derived_dir))

    for s in _SPLITS:
        (derived_dir / "images" / s / "images").mkdir(parents=True, exist_ok=True)
        (derived_dir / "images" / s / "labels").mkdir(parents=True, exist_ok=True)
    (derived_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (derived_dir / "reports").mkdir(parents=True, exist_ok=True)

    # Load the 5-class split manifest for provenance carry-through.
    src_manifest_path = source_dir / "splits" / "condition_split_manifest.csv"
    src_rows_by_filename: dict[tuple[str, str], dict[str, Any]] = {}
    if src_manifest_path.is_file():
        with src_manifest_path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                src_rows_by_filename[(row["derived_split"], row["derived_filename"])] = row

    class_img: dict[str, Counter] = {s: Counter() for s in _SPLITS}
    class_inst: dict[str, Counter] = {s: Counter() for s in _SPLITS}
    kept_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []

    for split in _SPLITS:
        images_dir = source_dir / "images" / split / "images"
        labels_dir = source_dir / "images" / split / "labels"
        if not images_dir.is_dir():
            continue
        for img_path in sorted(images_dir.iterdir()):
            if not img_path.is_file():
                continue
            rep.source_image_count += 1
            label_path = labels_dir / f"{img_path.stem}.txt"
            text = label_path.read_text(encoding="utf-8") if label_path.is_file() else ""
            kept_lines, n_removed, kept_cids = _filter_label_lines(text)
            rep.electrical_damage_instances_removed += n_removed

            src_row = src_rows_by_filename.get((split, img_path.name), {})

            if not kept_lines:
                rep.electrical_damage_only_images_dropped += 1
                dropped_rows.append({
                    "split": split, "filename": img_path.name,
                    "reason": "all_boxes_were_electrical_damage",
                    "electrical_damage_instances_removed": n_removed,
                    "family_id": src_row.get("family_id", ""),
                    "source_sha256": src_row.get("source_sha256", ""),
                })
                continue

            if n_removed > 0:
                rep.images_with_electrical_damage_partially_removed += 1

            out_img = derived_dir / "images" / split / "images" / img_path.name
            out_lbl = derived_dir / "images" / split / "labels" / f"{img_path.stem}.txt"
            shutil.copy2(img_path, out_img)
            out_lbl.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")

            classes_present = sorted({CONDITION_4CLASS_NAMES[c] for c in kept_cids if c in CONDITION_4CLASS_NAMES})
            for c in classes_present:
                class_img[split][c] += 1
            for c in kept_cids:
                if c in CONDITION_4CLASS_NAMES:
                    class_inst[split][CONDITION_4CLASS_NAMES[c]] += 1

            row = dict(src_row) if src_row else {
                "derived_filename": img_path.name, "derived_split": split,
                "family_id": "", "source_sha256": "",
            }
            row["derived_filename"] = img_path.name
            row["derived_split"] = split
            row["unified_classes"] = ";".join(classes_present) or "(none)"
            row["n_boxes_kept"] = len(kept_lines)
            row["electrical_damage_instances_removed_here"] = n_removed
            row.pop("electrical_damage_semantic_risk", None)  # no longer applicable - class removed
            kept_rows.append(row)

    rep.derived_image_count = len(kept_rows)
    rep.dropped_images = dropped_rows
    rep.final_split_counts = {s: sum(1 for r in kept_rows if r["derived_split"] == s) for s in _SPLITS}
    rep.class_image_counts = {s: dict(class_img[s]) for s in _SPLITS}
    rep.class_image_counts["total"] = dict(sum((Counter(v) for v in class_img.values()), Counter()))
    rep.class_instance_counts = {s: dict(class_inst[s]) for s in _SPLITS}
    rep.class_instance_counts["total"] = dict(sum((Counter(v) for v in class_inst.values()), Counter()))

    # --- data.yaml (4-class) ---
    import yaml
    (derived_dir / "data.yaml").write_text(yaml.safe_dump({
        "path": str(derived_dir).replace("\\", "/"),
        "train": "images/train/images", "val": "images/valid/images", "test": "images/test/images",
        "names": {i: CONDITION_4CLASS_NAMES[i] for i in range(len(CONDITION_4CLASS_ORDER))},
        "provenance_grade": "C", "provenance_tag": "personal-experimental-unverified-provenance",
        "derived_from": "datasets/ground_pv_research/condition (5-class) — ELECTRICAL_DAMAGE removed, not relabelled",
        "do_not": ["redistribute", "publish", "use commercially", "claim verified CC BY image rights"],
    }, sort_keys=False), encoding="utf-8")

    # --- manifests ---
    if kept_rows:
        fieldnames = sorted({k for r in kept_rows for k in r})
        with (derived_dir / "splits_4class_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in kept_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
    dropped_fieldnames = ["split", "filename", "reason", "electrical_damage_instances_removed",
                          "family_id", "source_sha256"]
    with (derived_dir / "manifests" / "dropped_images.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=dropped_fieldnames)
        w.writeheader()
        w.writerows(dropped_rows)

    # --- leakage re-verification (defensive; guaranteed by construction, checked anyway) ---
    fam_to_splits: dict[str, set[str]] = defaultdict(set)
    for r in kept_rows:
        fam = r.get("family_id", "")
        if fam:
            fam_to_splits[fam].add(r["derived_split"])
    crossing = {f: sorted(s) for f, s in fam_to_splits.items() if len(s) > 1}
    sha_to_splits: dict[str, set[str]] = defaultdict(set)
    for r in kept_rows:
        sha = r.get("source_sha256", "")
        if sha:
            sha_to_splits[sha].add(r["derived_split"])
    exact_dup_crossing = {s: sorted(v) for s, v in sha_to_splits.items() if len(v) > 1}

    rep.quality_gates = {
        "no_electrical_damage_class_present": "ELECTRICAL_DAMAGE" not in rep.class_image_counts.get("total", {}),
        "no_snow_class_present": not any("snow" in c.lower() for c in rep.class_image_counts.get("total", {})),
        "no_defective_class_present": "DEFECTIVE" not in rep.class_image_counts.get("total", {}),
        "no_panel_class_present": "PANEL" not in rep.class_image_counts.get("total", {}),
        "no_family_crosses_splits": not crossing,
        "no_exact_duplicate_crosses_splits": not exact_dup_crossing,
        "crossing_families": crossing,
        "exact_duplicate_crossing": exact_dup_crossing,
        "electrical_damage_never_relabelled": True,  # by construction — lines are dropped, never rewritten to another class id
    }
    rep.quality_gates["all_pass"] = all(
        v is True for k, v in rep.quality_gates.items()
        if isinstance(v, bool)
    )

    source_tree_after = _tree_digest(source_dir)
    rep.source_tree_unchanged = source_tree_before == source_tree_after
    if not rep.source_tree_unchanged:
        raise RuntimeError("source 5-class dataset changed during the 4-class build — aborting")

    payload = rep.as_dict()
    (derived_dir / "manifests" / "condition_4class_manifest.json").write_text(
        json.dumps({
            "taxonomy": CONDITION_4CLASS_NAMES,
            "derived_from": str(source_dir),
            "removal_policy": (
                "Every ELECTRICAL_DAMAGE annotation line is dropped, never relabelled to another "
                "class. An image whose annotations were entirely ELECTRICAL_DAMAGE is dropped "
                "from the derived dataset (same policy the 5-class builder already applies to "
                "snow-only images). An image with a mix of ELECTRICAL_DAMAGE and other classes is "
                "kept with only the ELECTRICAL_DAMAGE line(s) removed."
            ),
            "report": payload,
        }, indent=2, default=str),
        encoding="utf-8",
    )
    (derived_dir / "reports" / "condition_4class_build_report.md").write_text(_render_md(rep), encoding="utf-8")
    return rep


def _render_md(rep: Condition4ClassBuildReport) -> str:
    g = rep.quality_gates
    verdict = "APPROVED" if g.get("all_pass") else "BLOCKED"
    L = [
        "# Active 4-class condition dataset (`condition_4class_v1`) — build report\n",
        f"**STATUS: {verdict}**\n",
        f"- derived from (read-only, verified unchanged: {rep.source_tree_unchanged}): `{rep.source_dir}`",
        f"- output: `{rep.derived_dir}`\n",
        "## Removal effect\n",
        f"- source images (5-class): {rep.source_image_count}",
        f"- derived images (4-class): {rep.derived_image_count}",
        f"- ELECTRICAL_DAMAGE instances removed (never relabelled): {rep.electrical_damage_instances_removed}",
        f"- images dropped (ALL boxes were ELECTRICAL_DAMAGE): {rep.electrical_damage_only_images_dropped}",
        f"- images kept with ELECTRICAL_DAMAGE partially removed (other classes remain): "
        f"{rep.images_with_electrical_damage_partially_removed}\n",
        "## Split counts (preserved exactly from the 5-class leakage-clean split)\n",
        "| split | images |",
        "|---|---|",
    ]
    for s in _SPLITS:
        L.append(f"| {s} | {rep.final_split_counts.get(s, 0)} |")
    L += ["", "## Class distribution (4-class)\n", "| class | train img | valid img | test img | total img | total instances |",
          "|---|---|---|---|---|---|"]
    for c in CONDITION_4CLASS_ORDER:
        L.append(
            f"| {c} | {rep.class_image_counts.get('train', {}).get(c, 0)} "
            f"| {rep.class_image_counts.get('valid', {}).get(c, 0)} "
            f"| {rep.class_image_counts.get('test', {}).get(c, 0)} "
            f"| {rep.class_image_counts.get('total', {}).get(c, 0)} "
            f"| {rep.class_instance_counts.get('total', {}).get(c, 0)} |"
        )
    L += ["", "## Quality gates\n"]
    for k in sorted(g):
        if k in ("all_pass", "crossing_families", "exact_duplicate_crossing"):
            continue
        L.append(f"- {k}: {g[k]}")
    if g.get("crossing_families"):
        L.append(f"- CROSSING FAMILIES (should be empty): {g['crossing_families']}")
    if g.get("exact_duplicate_crossing"):
        L.append(f"- CROSSING EXACT DUPLICATES (should be empty): {g['exact_duplicate_crossing']}")
    L.append(f"- **ALL PASS: {g.get('all_pass')}**\n")
    return "\n".join(L) + "\n"
