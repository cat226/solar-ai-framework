"""tests/test_ground_pv_build_kaggle_dataset.py — staging the remediated
3-class dataset into a Kaggle package, with provenance carried along and the
leakage-clean gate enforced."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from training.detection.ground_pv.build_kaggle_dataset import _detect_taxonomy, build


def _mini_derived(root: Path) -> tuple[Path, Path, Path]:
    """A tiny 3-class derived dataset + matching manifests + remediation report."""
    yolo = root / "yolo"
    reports = root / "reports"
    splits = root / "splits"
    rows = []
    for split, items in {
        "train": [("a", "fam0000", ["0 0.5 0.5 0.4 0.4"]), ("b", "fam0001", ["1 0.5 0.5 0.3 0.3"])],
        "valid": [("c", "fam0002", ["0 0.5 0.5 0.5 0.5"])],
        "test":  [("d", "fam0003", ["2 0.5 0.5 0.2 0.2"])],
    }.items():
        (yolo / split / "images").mkdir(parents=True, exist_ok=True)
        (yolo / split / "labels").mkdir(parents=True, exist_ok=True)
        for stem, fam, lines in items:
            Image.new("RGB", (32, 32), (10, 20, 30)).save(yolo / split / "images" / f"{stem}.jpg")
            (yolo / split / "labels" / f"{stem}.txt").write_text("\n".join(lines))
            rows.append({"derived_filename": f"{stem}.jpg", "derived_split": split,
                         "family_id": fam, "source_path": f"{split}/images/{stem}.jpg",
                         "source_official_split": split, "sha256": "x", "classes": "panel"})
    (yolo / "data.yaml").write_text(yaml.safe_dump(
        {"path": ".", "train": "train/images", "val": "valid/images", "test": "test/images",
         "names": {0: "panel", 1: "cracked", 2: "dusty"}}))
    (yolo / "TRACEABILITY.csv").write_text("")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "remediation_report.json").write_text(json.dumps({
        "no_group_crosses_splits": True, "residual_cross_split_near_dup_pairs": 0,
        "final_split_counts": {"train": 2, "valid": 1, "test": 1},
    }))
    (reports / "remediation_report.md").write_text("# report\n")
    (splits / "leakage_clean_split").mkdir(parents=True, exist_ok=True)
    (splits / "official_split").mkdir(parents=True, exist_ok=True)
    with (splits / "leakage_clean_split" / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (splits / "official_split" / "manifest.csv").write_text("source_path,official_split\n")
    return yolo, reports, splits


def test_detect_taxonomy():
    assert _detect_taxonomy({0: "panel", 1: "cracked", 2: "dusty"}) == "experiment3"
    assert _detect_taxonomy({0: "panel", 1: "bird_drop", 2: "cracked", 3: "dusty"}) == "research4"
    with pytest.raises(SystemExit):
        _detect_taxonomy({0: "panel", 1: "hotspot"})


def test_build_stages_3class_with_provenance(tmp_path):
    yolo, reports, splits = _mini_derived(tmp_path)
    staging = tmp_path / "pkg"
    summary = build(yolo, staging, reports, splits, overwrite=True)

    assert summary["taxonomy"] == "experiment3"
    assert summary["class_names"] == {0: "panel", 1: "cracked", 2: "dusty"}
    assert summary["total_images"] == 4
    # Kaggle layout
    assert (staging / "images" / "train").is_dir()
    assert (staging / "labels" / "test").is_dir()
    dy = yaml.safe_load((staging / "data.yaml").read_text())
    assert dy["train"] == "images/train" and dy["names"] == {0: "panel", 1: "cracked", 2: "dusty"}
    # provenance carried into the package
    prov = staging / "provenance"
    assert (prov / "remediation_report.json").is_file()
    assert (prov / "leakage_clean_split__manifest.csv").is_file()
    assert "leakage_clean_split__manifest.csv" in summary["provenance_files_copied"]
    # content hash present + stable
    assert len(summary["dataset_content_sha256"]) == 64
    summary2 = build(yolo, tmp_path / "pkg2", reports, splits, overwrite=True)
    assert summary2["dataset_content_sha256"] == summary["dataset_content_sha256"]


def test_build_refuses_when_family_crosses_splits(tmp_path):
    yolo, reports, splits = _mini_derived(tmp_path)
    # corrupt the manifest so a family spans train+test
    mp = splits / "leakage_clean_split" / "manifest.csv"
    rows = list(csv.DictReader(mp.open()))
    rows[-1]["family_id"] = rows[0]["family_id"]  # test row now shares train's family
    with mp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with pytest.raises(SystemExit, match="overlap"):
        build(yolo, tmp_path / "pkg", reports, splits, overwrite=True)


def test_build_refuses_when_remediation_not_clean(tmp_path):
    yolo, reports, splits = _mini_derived(tmp_path)
    (reports / "remediation_report.json").write_text(json.dumps({
        "no_group_crosses_splits": False, "residual_cross_split_near_dup_pairs": 3}))
    with pytest.raises(SystemExit):
        build(yolo, tmp_path / "pkg", reports, splits, overwrite=True)


def test_build_rejects_bird_drop_class_in_labels(tmp_path):
    yolo, reports, splits = _mini_derived(tmp_path)
    (yolo / "train" / "labels" / "a.txt").write_text("3 0.5 0.5 0.4 0.4")  # class 3 = out of 0..2
    with pytest.raises(SystemExit, match="class id 3 out of"):
        build(yolo, tmp_path / "pkg", reports, splits, overwrite=True)
