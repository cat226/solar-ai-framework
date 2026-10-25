"""training/detection/ground_pv/build_condition_4class_kaggle_package.py —
stage the ACTIVE 4-class condition/defect research dataset
(`datasets/ground_pv_research/condition_4class_v1/`) into
`kaggle/ground_condition_4class_dataset/` for upload as a PRIVATE Kaggle
Dataset.

Local only. Never calls the Kaggle API — it assembles + validates the
staging directory. Uploading is a separate explicit step.

Source: datasets/ground_pv_research/condition_4class_v1/ (built by
build_condition_4class_dataset.py from the existing, unmodified 5-class
research dataset — see [[phase15a_four_class_condition_pipeline]]).
Taxonomy: 0=CLEAN 1=DUSTY 2=BIRD_DROP 3=PHYSICAL_DAMAGE. ELECTRICAL_DAMAGE
has been REMOVED (never relabelled) — see docs/PANEL_CONDITION_PIPELINE.md.

PERSONAL, NON-COMMERCIAL RESEARCH ONLY (unchanged from the 5-class
dataset — the underlying images still trace to the same provenance-grade-C
Kaggle source). Provenance grade C, tag
`personal-experimental-unverified-provenance`.

Validation before staging (fail closed):
  * data.yaml names == {0:CLEAN, 1:DUSTY, 2:BIRD_DROP, 3:PHYSICAL_DAMAGE}
    exactly (via condition_taxonomy.assert_active_taxonomy)
  * every image has a label file; every line parses; class ids 0..3;
    coords in [0,1]; positive w/h
  * split counts match the on-disk image counts
  * no leakage family spans two derived splits (re-checked from
    splits_4class_manifest.csv)
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]

import sys  # noqa: E402
sys.path.insert(0, str(_REPO_ROOT))
from training.detection.ground_pv.condition_taxonomy import (  # noqa: E402
    CONDITION_4CLASS_NAMES,
    assert_active_taxonomy,
)

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

KAGGLE_DATASET_ID = "edithstark/solar-ai-ground-condition-4class-research-v1"
KAGGLE_TITLE = "Solar AI ground PV condition 4-class research v1"


def _sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_labels(images_root: Path) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        idir = images_root / split / "images"
        ldir = images_root / split / "labels"
        if not idir.is_dir():
            errors.append(f"missing split dir: {split}")
            continue
        imgs = [p for p in sorted(idir.glob("*")) if p.suffix.lower() in _IMAGE_EXT]
        counts[split] = len(imgs)
        for img in imgs:
            lp = ldir / f"{img.stem}.txt"
            if not lp.is_file():
                errors.append(f"{split}: no label for {img.name}")
                continue
            for i, line in enumerate(lp.read_text(encoding="utf-8").splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{split}/{lp.name}:{i} not 5 fields")
                    continue
                try:
                    cid = int(parts[0])
                    cx, cy, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    errors.append(f"{split}/{lp.name}:{i} non-numeric")
                    continue
                if cid not in CONDITION_4CLASS_NAMES:
                    errors.append(f"{split}/{lp.name}:{i} class id {cid} out of range")
                if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
                    errors.append(f"{split}/{lp.name}:{i} bad coords")
    return errors, counts


def _check_family_isolation(split_manifest_csv: Path) -> tuple[bool, dict[str, list[str]]]:
    fam_splits: dict[str, set[str]] = defaultdict(set)
    with split_manifest_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fam_splits[row["family_id"]].add(row["derived_split"])
    crossing = {f: sorted(s) for f, s in fam_splits.items() if len(s) > 1}
    return (not crossing), crossing


def _check_exact_dup_isolation(split_manifest_csv: Path) -> tuple[bool, dict[str, list[str]]]:
    sha_splits: dict[str, set[str]] = defaultdict(set)
    with split_manifest_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sha = row.get("source_sha256", "")
            if sha:
                sha_splits[sha].add(row["derived_split"])
    crossing = {s: sorted(v) for s, v in sha_splits.items() if len(v) > 1}
    return (not crossing), crossing


def build(condition_4class_dir: Path, out_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    condition_4class_dir = Path(condition_4class_dir)
    out_dir = Path(out_dir)
    images_root = condition_4class_dir / "images"
    data_yaml_src = condition_4class_dir / "data.yaml"
    manifest_src = condition_4class_dir / "manifests" / "condition_4class_manifest.json"
    split_csv_src = condition_4class_dir / "splits_4class_manifest.csv"
    build_report_src = condition_4class_dir / "reports" / "condition_4class_build_report.md"
    dropped_src = condition_4class_dir / "manifests" / "dropped_images.csv"

    import yaml
    cfg = yaml.safe_load(data_yaml_src.read_text(encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    names = {int(k): str(v) for k, v in names.items()}
    assert_active_taxonomy(names)  # fails closed on anything but the exact 4-class taxonomy

    errors, counts = _validate_labels(images_root)
    if errors:
        raise SystemExit("label validation failed:\n  " + "\n  ".join(errors[:40]))

    if not split_csv_src.is_file():
        raise SystemExit(f"split manifest not found: {split_csv_src}")

    fam_isolated, fam_crossing = _check_family_isolation(split_csv_src)
    if not fam_isolated:
        raise SystemExit(f"leakage family spans splits: {fam_crossing}")

    dup_isolated, dup_crossing = _check_exact_dup_isolation(split_csv_src)
    if not dup_isolated:
        raise SystemExit(f"exact duplicate spans splits: {dup_crossing}")

    if out_dir.exists():
        if not overwrite:
            raise SystemExit(f"{out_dir} exists (pass --overwrite)")
        for child in out_dir.iterdir():
            if child.name == "dataset-metadata.json":
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()

    # ---- materialise the flat YOLO layout -------------------------------
    content_digest = hashlib.sha256()
    per_split_class_instances: dict[str, Counter] = {}
    for split in ("train", "valid", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        src_i = images_root / split / "images"
        src_l = images_root / split / "labels"
        inst: Counter = Counter()
        for img in sorted(src_i.glob("*")):
            if img.suffix.lower() not in _IMAGE_EXT:
                continue
            shutil.copy2(img, out_dir / "images" / split / img.name)
            lp = src_l / f"{img.stem}.txt"
            shutil.copy2(lp, out_dir / "labels" / split / lp.name)
            content_digest.update(f"images/{split}/{img.name}".encode())
            content_digest.update(img.read_bytes())
            content_digest.update(f"labels/{split}/{lp.name}".encode())
            content_digest.update(lp.read_bytes())
            for line in lp.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) == 5:
                    inst[CONDITION_4CLASS_NAMES[int(p[0])]] += 1
        per_split_class_instances[split] = inst

    # ---- data.yaml (path filled at train time on Kaggle) ---------------
    pkg_yaml = {
        "path": ".",
        "train": "images/train", "val": "images/valid", "test": "images/test",
        "names": CONDITION_4CLASS_NAMES,
        "nc": len(CONDITION_4CLASS_NAMES),
        "provenance_grade": "C",
        "provenance_tag": "personal-experimental-unverified-provenance",
        "underlying_source": "images ultimately trace to Kaggle pythonafroz/solar-panel-images (licence: Unknown)",
        "do_not": ["redistribute", "publish", "use commercially", "claim verified CC BY image rights"],
        "derived_from": "the existing 5-class condition dataset — ELECTRICAL_DAMAGE removed, never relabelled",
    }
    data_yaml_text = yaml.safe_dump(pkg_yaml, sort_keys=False)
    (out_dir / "data.yaml").write_text(data_yaml_text, encoding="utf-8")
    content_digest.update(b"data.yaml")
    content_digest.update(data_yaml_text.encode())

    # ---- provenance bundle ------------------------------------------
    prov = out_dir / "provenance"
    prov.mkdir(exist_ok=True)
    copied = []
    for src in (manifest_src, split_csv_src, build_report_src, dropped_src):
        if src.is_file():
            shutil.copy2(src, prov / src.name)
            copied.append(src.name)

    dataset_content_sha256 = content_digest.hexdigest()
    data_yaml_sha256 = _sha256_text(data_yaml_text)
    split_manifest_sha256 = _sha256_file(split_csv_src)

    summary = {
        "staging_dir": str(out_dir),
        "kaggle_dataset_id": KAGGLE_DATASET_ID,
        "taxonomy": CONDITION_4CLASS_NAMES,
        "provenance_grade": "C",
        "provenance_tag": "personal-experimental-unverified-provenance",
        "personal_non_commercial_research_only": True,
        "do_not": ["redistribute", "publish", "use commercially",
                   "claim the underlying images have verified CC BY rights"],
        "electrical_damage_removed": True,
        "counts": counts,
        "total_images": sum(counts.values()),
        "class_instances_per_split": {k: dict(v) for k, v in per_split_class_instances.items()},
        "class_instances_total": dict(sum((Counter(v) for v in per_split_class_instances.values()), Counter())),
        "no_family_crosses_splits": fam_isolated,
        "no_exact_duplicate_crosses_splits": dup_isolated,
        "data_yaml_sha256": data_yaml_sha256,
        "dataset_content_sha256": dataset_content_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "provenance_files_copied": sorted(copied),
        "source_dir": str(condition_4class_dir),
        "derived_from": "datasets/ground_pv_research/condition (5-class, unmodified) — ELECTRICAL_DAMAGE removed, never relabelled",
    }
    (out_dir / "PACKAGE_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- dataset-metadata.json (Kaggle) -----------------------------
    description = (
        "Solar AI ground-level PV CONDITION research dataset — ACTIVE 4-CLASS taxonomy "
        "(experimental).\n\n"
        "PERSONAL, NON-COMMERCIAL RESEARCH ONLY. Provenance grade C, tagged "
        "`personal-experimental-unverified-provenance`. Derived from the project's existing "
        "5-class condition dataset by REMOVING every ELECTRICAL_DAMAGE annotation "
        "(never relabelled as PHYSICAL_DAMAGE or any other class) — the source audit found "
        "those boxes visually correspond to cell cracks, not confirmed electrical failure. "
        "The underlying photographs ultimately trace to Kaggle `pythonafroz/solar-panel-images`, "
        "whose Kaggle licence is UNKNOWN. Do NOT redistribute, publish, use commercially, or "
        "claim verified CC BY image rights for the underlying images.\n\n"
        "Classes (contiguous, ACTIVE taxonomy): 0=CLEAN 1=DUSTY 2=BIRD_DROP 3=PHYSICAL_DAMAGE. "
        "ELECTRICAL_DAMAGE, SNOW, and the ambiguous 'Defective' are excluded. PANEL is not a "
        "class here — this feeds a SECOND-STAGE condition model that runs on panel crops from "
        "the separate ground panel detector.\n\n"
        "PHYSICAL_DAMAGE is the rarest active class (314 images / 725 instances vs. DUSTY's "
        "1,364 images / 8,550 instances) — a known class-imbalance limitation.\n\n"
        f"{sum(counts.values())} images: train {counts['train']} / valid {counts['valid']} "
        f"/ test {counts['test']}. Leakage-safe split preserved EXACTLY from the 5-class "
        "dataset's own leakage-clean family-grouped split (a class-filtered subset of a "
        "leakage-clean split stays leakage-clean by construction). Detection format, bounding "
        "boxes preserved verbatim for the surviving classes (no fabricated boxes). "
        "provenance/ carries the full build report and per-image manifest.\n\n"
        "Not a validated engineering or safety diagnosis."
    )
    metadata = {
        "title": KAGGLE_TITLE,
        "id": KAGGLE_DATASET_ID,
        "licenses": [{"name": "other"}],
        "subtitle": "Active 4-class PV condition set - unverified provenance (grade C)",
        "description": description,
    }
    (out_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    (out_dir / "README.md").write_text(
        "# ground_condition_4class_dataset/ — staging for a PRIVATE Kaggle Dataset\n\n"
        f"Kaggle id: `{KAGGLE_DATASET_ID}` (PRIVATE).\n\n"
        "Built by `scripts/build_condition_4class_kaggle_package.py` from "
        "`datasets/ground_pv_research/condition_4class_v1/`. The materialised image/label tree "
        "is gitignored; `PACKAGE_SUMMARY.json`, `dataset-metadata.json`, this README and the "
        "`provenance/` bundle are committed.\n\n"
        "PERSONAL NON-COMMERCIAL RESEARCH ONLY — provenance grade C, unverified. "
        "ELECTRICAL_DAMAGE has been REMOVED from this dataset (see provenance/ for the exact "
        "removal counts) — never relabelled.\n\n"
        f"- images: {sum(counts.values())} (train {counts['train']} / valid {counts['valid']} / test {counts['test']})\n"
        f"- data_yaml_sha256: `{data_yaml_sha256}`\n"
        f"- dataset_content_sha256: `{dataset_content_sha256}`\n"
        f"- split_manifest_sha256: `{split_manifest_sha256}`\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition-4class-dir", type=Path,
                    default=_REPO_ROOT / "datasets" / "ground_pv_research" / "condition_4class_v1")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "kaggle" / "ground_condition_4class_dataset")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    summary = build(args.condition_4class_dir, args.out, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
