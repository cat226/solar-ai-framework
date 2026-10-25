"""training/detection/ground_pv/build_kaggle_dataset.py — Stage the audited /
remediated ground-PV dataset into kaggle/ground_pv_dataset/ for upload as a
Kaggle Dataset.

Local only. Never calls the Kaggle API - it only assembles + validates the
staging directory. Uploading is a separate, explicit `kaggle datasets
create/version` step (see kaggle/ground_pv_dataset/README.md).

Two supported inputs (auto-detected from the source data.yaml `names:`):

  * the 4-class NORMALIZED dataset (panel / bird_drop / cracked / dusty) -
    gated on a passing dataset_audit.json.
  * the 3-class REMEDIATED, leakage-clean dataset
    (panel / cracked / dusty, from datasets/ground_pv_research/yolo/) -
    gated on remediation_report.json (no_group_crosses_splits == true).
    The split-provenance manifests and remediation report are copied INTO
    the package so Kaggle carries full traceability.

Validation before staging (fail closed):
  * data.yaml `names:` == a recognised taxonomy
  * every image has a label file; every label line parses; class ids in
    range; coords in [0, 1]; positive width/height
  * for the remediated dataset: image count == leakage-clean manifest row
    count, and no leakage family spans two splits
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.class_map import (  # noqa: E402
    FIRST_EXPERIMENT_CLASS_NAMES,
    RESEARCH_CLASS_NAMES,
)

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

_TAXONOMIES = {
    "research4": RESEARCH_CLASS_NAMES,
    "experiment3": FIRST_EXPERIMENT_CLASS_NAMES,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _detect_taxonomy(names: dict[int, str]) -> str:
    for key, tax in _TAXONOMIES.items():
        if names == tax:
            return key
    raise SystemExit(f"data.yaml names {names} match no known taxonomy {list(_TAXONOMIES)}")


def _validate_labels(dataset_dir: Path, max_cid: int) -> list[str]:
    errors: list[str] = []
    for split in ("train", "valid", "test"):
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if not images_dir.is_dir():
            errors.append(f"missing split dir: {split}/images")
            continue
        for img in sorted(images_dir.glob("*")):
            if img.suffix.lower() not in _IMAGE_EXT:
                continue
            lp = labels_dir / f"{img.stem}.txt"
            if not lp.is_file():
                errors.append(f"{split}: no label file for {img.name}")
                continue
            for i, line in enumerate(lp.read_text(encoding="utf-8").splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{split}/{lp.name}:{i} malformed ({len(parts)} fields)")
                    continue
                try:
                    cid = int(parts[0])
                    vals = [float(v) for v in parts[1:]]
                except ValueError:
                    errors.append(f"{split}/{lp.name}:{i} non-numeric")
                    continue
                if not 0 <= cid <= max_cid:
                    errors.append(f"{split}/{lp.name}:{i} class id {cid} out of 0..{max_cid}")
                if not all(0.0 <= v <= 1.0 for v in vals):
                    errors.append(f"{split}/{lp.name}:{i} coord out of [0,1]")
                if vals[2] <= 0 or vals[3] <= 0:
                    errors.append(f"{split}/{lp.name}:{i} non-positive w/h")
            if len(errors) > 200:
                errors.append("... (truncated)")
                return errors
    return errors


def _check_remediation(dataset_dir: Path, reports_dir: Path, splits_dir: Path) -> dict:
    rep_path = reports_dir / "remediation_report.json"
    if not rep_path.is_file():
        raise SystemExit(f"no remediation report at {rep_path} - run scripts/remediate_ground_pv_dataset.py")
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    if not rep.get("no_group_crosses_splits", False):
        raise SystemExit("remediation_report.json: no_group_crosses_splits is not true - refusing to stage")
    if rep.get("residual_cross_split_near_dup_pairs", 1) != 0:
        raise SystemExit("remediation_report.json: residual cross-split near-duplicate pairs != 0")

    manifest = splits_dir / "leakage_clean_split" / "manifest.csv"
    if not manifest.is_file():
        raise SystemExit(f"no leakage-clean manifest at {manifest}")
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    n_images = sum(1 for s in ("train", "valid", "test") for _ in (dataset_dir / s / "images").glob("*"))
    if len(rows) != n_images:
        raise SystemExit(f"manifest rows ({len(rows)}) != staged images ({n_images})")

    fam_by_split: dict[str, set[str]] = {}
    for r in rows:
        fam_by_split.setdefault(r["derived_split"], set()).add(r["family_id"])
    a, b, c = fam_by_split.get("train", set()), fam_by_split.get("valid", set()), fam_by_split.get("test", set())
    if (a & b) or (a & c) or (b & c):
        raise SystemExit("leakage families overlap between splits - refusing to stage")
    return {
        "remediation_report": rep,
        "leakage_clean_manifest_sha256": _sha256_file(manifest),
        "leakage_families_total": len(a | b | c),
        "leakage_families_per_split": {"train": len(a), "valid": len(b), "test": len(c)},
        "manifest_rows": len(rows),
    }


def build(
    dataset_dir: Path,
    staging_dir: Path,
    reports_dir: Path,
    splits_dir: Path | None = None,
    *,
    overwrite: bool = False,
) -> dict:
    dataset_dir = Path(dataset_dir)
    staging_dir = Path(staging_dir)
    splits_dir = Path(splits_dir) if splits_dir else _REPO_ROOT / "datasets/ground_pv_research/splits"
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(f"no data.yaml at {data_yaml} - run the normalize / remediate step first")

    import yaml
    names = (yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}).get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    names = {int(k): v for k, v in names.items()}
    taxonomy = _detect_taxonomy(names)
    max_cid = len(names) - 1

    provenance: dict = {"taxonomy": taxonomy}
    if taxonomy == "experiment3":
        provenance.update(_check_remediation(dataset_dir, reports_dir, splits_dir))
    else:
        audit_json = reports_dir / "dataset_audit.json"
        if not audit_json.is_file():
            raise SystemExit(f"no audit report at {audit_json}")
        audit = json.loads(audit_json.read_text(encoding="utf-8"))
        if not audit.get("passed", False):
            raise SystemExit("dataset_audit.json says the audit did NOT pass - refusing to stage")
        provenance["dataset_audit_sha256"] = _sha256_file(audit_json)

    errors = _validate_labels(dataset_dir, max_cid)
    if errors:
        raise SystemExit("label validation failed:\n  " + "\n  ".join(errors[:50]))

    for sub in ("images", "labels"):
        target = staging_dir / sub
        if target.exists():
            if not overwrite:
                raise SystemExit(f"{target} exists (pass --overwrite)")
            shutil.rmtree(target)
    for extra in ("provenance",):
        if (staging_dir / extra).exists() and overwrite:
            shutil.rmtree(staging_dir / extra)

    counts: dict[str, dict[str, int]] = {}
    content_lines: list[str] = []
    for split in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            src = dataset_dir / split / sub
            dst = staging_dir / sub / split
            dst.mkdir(parents=True, exist_ok=True)
            n = 0
            for f in sorted(src.glob("*")):
                if f.is_file():
                    shutil.copy2(f, dst / f.name)
                    if sub == "labels":  # hash labels inline; images below in bulk
                        content_lines.append(f"{split}/labels/{f.name}:{_sha256_file(dst / f.name)}")
                    n += 1
            counts.setdefault(split, {})[sub] = n
    # bulk image hashes (kept separate so a label-only change is visible)
    for split in ("train", "valid", "test"):
        for f in sorted((staging_dir / "images" / split).glob("*")):
            content_lines.append(f"{split}/images/{f.name}:{_sha256_file(f)}")

    kaggle_yaml = {
        "path": ".",
        "train": "images/train", "val": "images/valid", "test": "images/test",
        "names": {i: names[i] for i in range(len(names))},
    }
    (staging_dir / "data.yaml").write_text(yaml.safe_dump(kaggle_yaml, sort_keys=False), encoding="utf-8")

    # --- copy provenance INTO the package ---
    prov_dir = staging_dir / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for rel_src in (
        reports_dir / "remediation_report.md", reports_dir / "remediation_report.json",
        reports_dir / "dataset_audit.md", reports_dir / "dataset_audit.json",
        reports_dir / "acquisition_manifest.json",
        dataset_dir / "REMEDIATION.json", dataset_dir / "TRACEABILITY.csv",
        dataset_dir / "NORMALIZATION.json",
        splits_dir / "leakage_clean_split" / "manifest.csv",
        splits_dir / "official_split" / "manifest.csv",
        splits_dir / "official_split" / "README.md",
    ):
        if rel_src.is_file():
            name = rel_src.name
            if rel_src.parent.name in ("leakage_clean_split", "official_split"):
                name = f"{rel_src.parent.name}__{rel_src.name}"
            shutil.copy2(rel_src, prov_dir / name)
            copied.append(name)

    dataset_content_sha256 = _sha256_text("\n".join(sorted(content_lines)))
    summary = {
        "staging_dir": str(staging_dir),
        "taxonomy": taxonomy,
        "class_names": kaggle_yaml["names"],
        "counts": counts,
        "total_images": sum(v["images"] for v in counts.values()),
        "data_yaml": str(staging_dir / "data.yaml"),
        "data_yaml_sha256": _sha256_text((staging_dir / "data.yaml").read_text(encoding="utf-8")),
        "dataset_content_sha256": dataset_content_sha256,
        "provenance_files_copied": sorted(copied),
        "provenance": provenance,
    }
    (staging_dir / "STAGING_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str),
                                                     encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _repo_yolo = _REPO_ROOT / "datasets/ground_pv_research/yolo"
    try:
        from training.detection.ground_pv.paths import NORMALIZED_YOLO_DIR, REPORTS_DIR
    except Exception:  # noqa: BLE001
        NORMALIZED_YOLO_DIR = _repo_yolo
        REPORTS_DIR = _REPO_ROOT / "datasets/ground_pv_research/reports"
    default_dir = _repo_yolo if (_repo_yolo / "data.yaml").is_file() else NORMALIZED_YOLO_DIR
    parser.add_argument("--dataset-dir", "--normalized-dir", type=Path, default=default_dir,
                        dest="dataset_dir")
    parser.add_argument("--staging-dir", type=Path, default=_REPO_ROOT / "kaggle/ground_pv_dataset")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--splits-dir", type=Path,
                        default=_REPO_ROOT / "datasets/ground_pv_research/splits")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = build(args.dataset_dir, args.staging_dir, args.reports_dir, args.splits_dir,
                    overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, default=str))
    print("\nNext: edit dataset-metadata.json 'id', then  kaggle datasets create -p "
          f"{args.staging_dir} --dir-mode zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
