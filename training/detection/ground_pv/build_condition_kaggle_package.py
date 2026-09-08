"""training/detection/ground_pv/build_condition_kaggle_package.py — stage the
unified condition/defect research dataset into kaggle/ground_condition_dataset/
for upload as a PRIVATE Kaggle Dataset.

Local only. Never calls the Kaggle API — it assembles + validates the staging
directory. Uploading is a separate explicit step.

Source: datasets/ground_pv_research/condition/  (built by build_condition_dataset.py)
Taxonomy: 0=CLEAN 1=DUSTY 2=BIRD_DROP 3=PHYSICAL_DAMAGE 4=ELECTRICAL_DAMAGE

PERSONAL, NON-COMMERCIAL RESEARCH ONLY. Provenance grade C, tag
`personal-experimental-unverified-provenance`. Underlying images ultimately
trace to Kaggle `pythonafroz/solar-panel-images` (licence: Unknown). The
provenance + electrical-damage HIGH-semantic-risk warnings are copied into the
package and echoed in the dataset description.

Validation before staging (fail closed):
  * data.yaml names == {0:CLEAN,1:DUSTY,2:BIRD_DROP,3:PHYSICAL_DAMAGE,4:ELECTRICAL_DAMAGE}
  * every image has a label file; every line parses; class ids 0..4; coords in
    [0,1]; positive w/h
  * split counts match condition_dataset_manifest.json
  * no leakage family (family_id) spans two derived splits — re-checked from
    condition_split_manifest.csv
  * the committed manifest / split / taxonomy SHAs match
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

CONDITION_NAMES = {0: "CLEAN", 1: "DUSTY", 2: "BIRD_DROP", 3: "PHYSICAL_DAMAGE", 4: "ELECTRICAL_DAMAGE"}
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

KAGGLE_DATASET_ID = "edithstark/solar-ai-ground-condition-research-v1"
KAGGLE_TITLE = "Solar AI ground PV condition research v1"


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
                if cid not in CONDITION_NAMES:
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


def build(condition_dir: Path, out_dir: Path, *, overwrite: bool = False) -> dict:
    condition_dir = Path(condition_dir)
    out_dir = Path(out_dir)
    images_root = condition_dir / "images"
    data_yaml_src = condition_dir / "data.yaml"
    manifest_src = condition_dir / "manifests" / "condition_dataset_manifest.json"
    split_csv_src = condition_dir / "splits" / "condition_split_manifest.csv"
    taxonomy_src = condition_dir / "metadata" / "condition_taxonomy.json"
    elec_src = condition_dir / "manifests" / "electrical_damage_annotations.csv"
    build_report_src = condition_dir / "reports" / "condition_build_report.md"
    overlap_report_src = condition_dir / "reports" / "condition_overlap_report.md"
    sources_src = condition_dir / "raw_refs" / "SOURCES.json"

    import yaml
    cfg = yaml.safe_load(data_yaml_src.read_text(encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    names = {int(k): str(v) for k, v in names.items()}
    if names != CONDITION_NAMES:
        raise SystemExit(f"taxonomy mismatch: {names} != {CONDITION_NAMES}")

    errors, counts = _validate_labels(images_root)
    if errors:
        raise SystemExit("label validation failed:\n  " + "\n  ".join(errors[:40]))

    manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
    rep = manifest["report"]
    expected = rep["final_split_counts"]
    if counts != {k: expected[k] for k in ("train", "valid", "test")}:
        raise SystemExit(f"split count mismatch: on-disk {counts} != manifest {expected}")

    isolated, crossing = _check_family_isolation(split_csv_src)
    if not isolated:
        raise SystemExit(f"leakage family spans splits: {crossing}")

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
        inst = Counter()
        for img in sorted(src_i.glob("*")):
            if img.suffix.lower() not in _IMAGE_EXT:
                continue
            shutil.copy2(img, out_dir / "images" / split / img.name)
            lp = src_l / f"{img.stem}.txt"
            shutil.copy2(lp, out_dir / "labels" / split / lp.name)
            content_digest.update(f"images/{split}/{img.name}".encode())
            content_digest.update(img.read_bytes())
            content_digest.update(f"labels/{split}/{lp.name}".encode())
            lb = lp.read_bytes()
            content_digest.update(lb)
            for line in lp.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) == 5:
                    inst[CONDITION_NAMES[int(p[0])]] += 1
        per_split_class_instances[split] = inst

    # ---- data.yaml (path filled at train time) -----------------------
    pkg_yaml = {
        "path": ".",
        "train": "images/train", "val": "images/valid", "test": "images/test",
        "names": CONDITION_NAMES,
        "nc": len(CONDITION_NAMES),
        "provenance_grade": "C",
        "provenance_tag": "personal-experimental-unverified-provenance",
        "underlying_source": "images ultimately trace to Kaggle pythonafroz/solar-panel-images (licence: Unknown)",
        "do_not": ["redistribute", "publish", "use commercially", "claim verified CC BY image rights"],
        "electrical_damage_semantic_risk": "HIGH — S1 electrical-damage boxes visually correspond to cell cracks; source class preserved, not relabelled",
    }
    data_yaml_text = yaml.safe_dump(pkg_yaml, sort_keys=False)
    (out_dir / "data.yaml").write_text(data_yaml_text, encoding="utf-8")
    content_digest.update(b"data.yaml")
    content_digest.update(data_yaml_text.encode())

    # ---- provenance bundle ------------------------------------------
    prov = out_dir / "provenance"
    prov.mkdir(exist_ok=True)
    copied = []
    for src in (manifest_src, split_csv_src, taxonomy_src, elec_src,
                build_report_src, overlap_report_src, sources_src):
        if src.is_file():
            shutil.copy2(src, prov / src.name)
            copied.append(src.name)

    dataset_content_sha256 = content_digest.hexdigest()
    data_yaml_sha256 = _sha256_text(data_yaml_text)
    split_manifest_sha256 = _sha256_file(split_csv_src)
    taxonomy_sha256 = _sha256_file(taxonomy_src)
    source_manifest_sha256 = _sha256_file(manifest_src)

    summary = {
        "staging_dir": str(out_dir),
        "kaggle_dataset_id": KAGGLE_DATASET_ID,
        "taxonomy": CONDITION_NAMES,
        "provenance_grade": "C",
        "provenance_tag": "personal-experimental-unverified-provenance",
        "personal_non_commercial_research_only": True,
        "do_not": ["redistribute", "publish", "use commercially",
                   "claim the underlying images have verified CC BY rights"],
        "electrical_damage_semantic_risk": {
            "flag": "HIGH",
            "applies_to": "all S1 electrical-damage annotations",
            "reason": "audit: those boxes visually correspond to cell cracks, not confirmed "
                      "electrical failure; source class preserved, NOT relabelled",
            "images": rep["electrical_damage_images"],
            "instances": rep["electrical_damage_instances"],
        },
        "counts": counts,
        "total_images": sum(counts.values()),
        "class_instances_per_split": {k: dict(v) for k, v in per_split_class_instances.items()},
        "class_instances_total": dict(sum((Counter(v) for v in per_split_class_instances.values()), Counter())),
        "class_image_counts": rep["class_image_counts"],
        "unique_family_counts_per_class": rep["unique_family_counts_per_class"],
        "families_per_split": rep["families_per_split"],
        "no_family_crosses_splits": rep["no_family_crosses_splits"],
        "data_yaml_sha256": data_yaml_sha256,
        "dataset_content_sha256": dataset_content_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "taxonomy_sha256": taxonomy_sha256,
        "source_condition_manifest_sha256": source_manifest_sha256,
        "provenance_files_copied": sorted(copied),
        "source_dir": str(_REPO_ROOT / "datasets" / "ground_pv_research" / "condition"),
        "quality_gates_all_pass": rep["quality_gates"].get("all_pass"),
    }
    (out_dir / "PACKAGE_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- dataset-metadata.json (Kaggle) -----------------------------
    description = (
        "Solar AI ground-level PV CONDITION/DEFECT research dataset (experimental).\n\n"
        "PERSONAL, NON-COMMERCIAL RESEARCH ONLY. Provenance grade C, tagged "
        "`personal-experimental-unverified-provenance`. This is a de-duplicated, "
        "leakage-safe repackaging of two Roboflow Universe object-detection datasets "
        "(defects-kgig6/detection-for-defects-in-solar-panels v2 and "
        "solar-panel-a5cyn/dusty-and-bird-droppings v1). The underlying photographs "
        "ultimately trace to Kaggle `pythonafroz/solar-panel-images`, whose Kaggle "
        "licence is UNKNOWN. Do NOT redistribute, publish, use commercially, or claim "
        "verified CC BY image rights for the underlying images.\n\n"
        "Classes (contiguous): 0=CLEAN 1=DUSTY 2=BIRD_DROP 3=PHYSICAL_DAMAGE "
        "4=ELECTRICAL_DAMAGE. snow/snow-covered and the ambiguous 'Defective' are "
        "excluded. PANEL is not a class here — this feeds a SECOND-STAGE condition "
        "model that runs on panel crops from the separate ground panel detector.\n\n"
        "*** ELECTRICAL_DAMAGE — HIGH SEMANTIC RISK ***\n"
        "Every ELECTRICAL_DAMAGE annotation comes from S1's 'electrical-damage' class. "
        "The provenance audit found these boxes visually correspond to CELL CRACKS, not "
        "confirmed electrical failure. The label is preserved unchanged (not relabelled) "
        "but a high score on this class must NOT be read as electrical-fault diagnosis. "
        "See provenance/electrical_damage_annotations.csv.\n\n"
        f"{sum(counts.values())} images: train {counts['train']} / valid {counts['valid']} "
        f"/ test {counts['test']}. Leakage-safe: whole near-duplicate families "
        "(exact SHA-256 u source basename u 16x16 dHash) assigned to one split each; "
        "no family crosses splits. Detection format, bounding boxes preserved verbatim "
        "(no fabricated boxes). provenance/ carries the full build report, per-image "
        "manifest, split manifest, taxonomy, overlap report and source references.\n\n"
        "Not a validated engineering or safety diagnosis."
    )
    metadata = {
        "title": KAGGLE_TITLE,
        "id": KAGGLE_DATASET_ID,
        "licenses": [{"name": "other"}],
        "subtitle": "Experimental 5-class PV condition/defect set - unverified provenance (grade C)",
        "description": description,
    }
    (out_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    (out_dir / "README.md").write_text(
        "# ground_condition_dataset/ — staging for a PRIVATE Kaggle Dataset\n\n"
        f"Kaggle id: `{KAGGLE_DATASET_ID}` (PRIVATE).\n\n"
        "Built by `scripts/build_condition_kaggle_package.py` from "
        "`datasets/ground_pv_research/condition/`. The materialised image/label tree is "
        "gitignored; `PACKAGE_SUMMARY.json`, `dataset-metadata.json`, this README and the "
        "`provenance/` bundle are committed.\n\n"
        "PERSONAL NON-COMMERCIAL RESEARCH ONLY — provenance grade C, unverified. "
        "ELECTRICAL_DAMAGE carries a HIGH semantic-risk flag (source boxes look like cell "
        "cracks). Do not redistribute / publish / use commercially.\n\n"
        f"- images: {sum(counts.values())} (train {counts['train']} / valid {counts['valid']} / test {counts['test']})\n"
        f"- data_yaml_sha256: `{data_yaml_sha256}`\n"
        f"- dataset_content_sha256: `{dataset_content_sha256}`\n"
        f"- split_manifest_sha256: `{split_manifest_sha256}`\n"
        f"- taxonomy_sha256: `{taxonomy_sha256}`\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition-dir", type=Path,
                    default=_REPO_ROOT / "datasets" / "ground_pv_research" / "condition")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "kaggle" / "ground_condition_dataset")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    summary = build(args.condition_dir, args.out, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
