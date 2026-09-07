"""training/detection/ground_pv/build_kaggle_dataset.py — Stage the audited,
class-normalized ground-PV dataset into kaggle/ground_pv_dataset/ for upload
as a Kaggle Dataset.

Local only. Never calls the Kaggle API — it only assembles + validates the
staging directory. Uploading is a separate, explicit `kaggle datasets
create/version` step (see kaggle/ground_pv_dataset/README.md).

Validation before staging (fail closed):
  * normalized data.yaml exists and its names == the research taxonomy
  * every image has a label file (empty label files are allowed = negative)
  * every label line parses, class ids in 0..3, coords in [0,1]
  * dataset_audit.json exists and recorded passed == true
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.class_map import RESEARCH_CLASS_NAMES  # noqa: E402

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _validate_labels(normalized_dir: Path) -> list[str]:
    errors: list[str] = []
    for split in ("train", "valid", "test"):
        images_dir = normalized_dir / split / "images"
        labels_dir = normalized_dir / split / "labels"
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
                    errors.append(f"{split}/{lp.name}:{i} malformed")
                    continue
                try:
                    cid = int(parts[0])
                    vals = [float(v) for v in parts[1:]]
                except ValueError:
                    errors.append(f"{split}/{lp.name}:{i} non-numeric")
                    continue
                if not 0 <= cid <= 3:
                    errors.append(f"{split}/{lp.name}:{i} class id {cid} out of 0..3")
                if not all(0.0 <= v <= 1.0 for v in vals):
                    errors.append(f"{split}/{lp.name}:{i} coord out of [0,1]")
            if len(errors) > 200:
                errors.append("... (truncated)")
                return errors
    return errors


def build(normalized_dir: Path, staging_dir: Path, reports_dir: Path, *, overwrite: bool = False) -> dict:
    normalized_dir = Path(normalized_dir)
    staging_dir = Path(staging_dir)
    data_yaml = normalized_dir / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(f"no normalized data.yaml at {data_yaml} — run the normalize step first")

    import yaml
    names = (yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}).get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    names = {int(k): v for k, v in names.items()}
    if names != RESEARCH_CLASS_NAMES:
        raise SystemExit(f"normalized names {names} != research taxonomy {RESEARCH_CLASS_NAMES}")

    audit_json = reports_dir / "dataset_audit.json"
    if not audit_json.is_file():
        raise SystemExit(f"no audit report at {audit_json} — run scripts/audit_ground_pv_dataset.py first")
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    if not audit.get("passed", False):
        raise SystemExit("dataset_audit.json says the audit did NOT pass — refusing to stage for training")

    errors = _validate_labels(normalized_dir)
    if errors:
        raise SystemExit("label validation failed:\n  " + "\n  ".join(errors[:50]))

    for sub in ("images", "labels"):
        target = staging_dir / sub
        if target.exists():
            if not overwrite:
                raise SystemExit(f"{target} exists (pass --overwrite)")
            shutil.rmtree(target)

    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            src = normalized_dir / split / sub
            dst = staging_dir / sub / split
            dst.mkdir(parents=True, exist_ok=True)
            n = 0
            for f in sorted(src.glob("*")):
                if f.is_file():
                    shutil.copy2(f, dst / f.name)
                    n += 1
            counts.setdefault(split, {})[sub] = n

    # Kaggle data.yaml: paths relative to the dataset root inside /kaggle/input.
    kaggle_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {i: RESEARCH_CLASS_NAMES[i] for i in range(4)},
    }
    (staging_dir / "data.yaml").write_text(yaml.safe_dump(kaggle_yaml, sort_keys=False), encoding="utf-8")

    for name in ("dataset_audit.md", "dataset_audit.json"):
        if (reports_dir / name).is_file():
            shutil.copy2(reports_dir / name, staging_dir / name)
    if (normalized_dir / "NORMALIZATION.json").is_file():
        shutil.copy2(normalized_dir / "NORMALIZATION.json", staging_dir / "NORMALIZATION.json")

    summary = {"staging_dir": str(staging_dir), "counts": counts,
               "data_yaml": str(staging_dir / "data.yaml")}
    (staging_dir / "STAGING_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    try:
        from training.detection.ground_pv.paths import NORMALIZED_YOLO_DIR, REPORTS_DIR
    except Exception:  # noqa: BLE001
        NORMALIZED_YOLO_DIR = _REPO_ROOT / "datasets/ground_pv_research/yolo"
        REPORTS_DIR = _REPO_ROOT / "datasets/ground_pv_research/reports"
    parser.add_argument("--normalized-dir", type=Path, default=NORMALIZED_YOLO_DIR)
    parser.add_argument("--staging-dir", type=Path,
                        default=_REPO_ROOT / "kaggle/ground_pv_dataset")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = build(args.normalized_dir, args.staging_dir, args.reports_dir, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2))
    print("\nNext: edit dataset-metadata.json 'id', then  kaggle datasets create -p "
          f"{args.staging_dir} --dir-mode zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
