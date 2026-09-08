"""tests/test_ground_pv_condition_kaggle_package.py — the staging builder that
packages datasets/ground_pv_research/condition/ for a PRIVATE Kaggle Dataset.

Guards: taxonomy, flat YOLO layout, provenance bundle + warnings carried into
the package, family-isolation re-check, deterministic content hash, and that
the source condition dataset is never modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from PIL import Image

from training.detection.ground_pv.build_condition_kaggle_package import (
    CONDITION_NAMES,
    build,
)


def _img(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (32, 32))
    for x in range(32):
        for y in range(32):
            im.putpixel((x, y), ((x * 7 + seed) % 256, (y * 5 + seed) % 256, (seed * 3) % 256))
    im.save(path)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _dir_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _make_condition_dir(tmp_path: Path) -> Path:
    """A miniature but structurally-faithful condition/ dataset."""
    d = tmp_path / "condition"
    layout = {
        "train": [("clean_a.jpg", "0 0.5 0.5 0.3 0.3\n"),
                  ("dusty_a.jpg", "1 0.5 0.5 0.4 0.4\n"),
                  ("elec_a.jpg", "4 0.4 0.4 0.2 0.2\n")],
        "valid": [("bird_a.jpg", "2 0.5 0.5 0.2 0.2\n")],
        "test": [("phys_a.jpg", "3 0.5 0.5 0.3 0.3\n"),
                 ("elec_b.jpg", "4 0.6 0.6 0.2 0.2\n")],
    }
    seed = 0
    fam = 0
    rows = []
    for split, items in layout.items():
        for name, lbl in items:
            seed += 1
            fam += 1
            _img(d / "images" / split / "images" / name, seed)
            _write(d / "images" / split / "labels" / f"{Path(name).stem}.txt", lbl)
            uni = CONDITION_NAMES[int(lbl.split()[0])]
            rows.append(f"{name},{split},fam{fam:05d},S1,ws/proj,2,{name},sha{seed},{split},"
                        f"{uni.lower()},{uni},32,32,1,0,0,C,personal-experimental-unverified-provenance,"
                        f"{'HIGH' if uni == 'ELECTRICAL_DAMAGE' else ''}")

    _write(d / "data.yaml", yaml.safe_dump({
        "path": "datasets/ground_pv_research/condition",
        "train": "images/train/images", "val": "images/valid/images", "test": "images/test/images",
        "names": CONDITION_NAMES,
    }, sort_keys=False))

    header = ("derived_filename,derived_split,family_id,source,source_dataset,source_version,"
              "source_filename,source_sha256,source_split,original_classes,unified_classes,"
              "image_w,image_h,n_boxes_kept,n_malformed_dropped,n_snow_boxes_dropped,"
              "provenance_grade,provenance_tag,electrical_damage_semantic_risk")
    _write(d / "splits" / "condition_split_manifest.csv", header + "\n" + "\n".join(rows) + "\n")

    report = {
        "final_split_counts": {"train": 3, "valid": 1, "test": 2},
        "class_image_counts": {"total": {"CLEAN": 1, "DUSTY": 1, "BIRD_DROP": 1,
                                         "PHYSICAL_DAMAGE": 1, "ELECTRICAL_DAMAGE": 2}},
        "unique_family_counts_per_class": {"CLEAN": 1, "DUSTY": 1, "BIRD_DROP": 1,
                                           "PHYSICAL_DAMAGE": 1, "ELECTRICAL_DAMAGE": 2},
        "families_per_split": {"train": 3, "valid": 1, "test": 2},
        "no_family_crosses_splits": True,
        "electrical_damage_images": 2,
        "electrical_damage_instances": 2,
        "quality_gates": {"all_pass": True},
    }
    _write(d / "manifests" / "condition_dataset_manifest.json",
           json.dumps({"provenance_grade": "C", "report": report, "manifest_rows": []}))
    _write(d / "manifests" / "electrical_damage_annotations.csv",
           "derived_split,derived_filename,source_class,unified_class,electrical_damage_semantic_risk,note\n"
           "train,elec_a.jpg,electrical-damage,ELECTRICAL_DAMAGE,HIGH,audit: cell cracks\n")
    _write(d / "metadata" / "condition_taxonomy.json", json.dumps({"unified_classes": CONDITION_NAMES}))
    _write(d / "reports" / "condition_build_report.md", "# build report\nAPPROVED WITH CAVEATS\n")
    _write(d / "reports" / "condition_overlap_report.md", "# overlap report\n")
    _write(d / "raw_refs" / "SOURCES.json", json.dumps({"S1": {}, "S3": {}}))
    return d


def _run(tmp_path: Path):
    cond = _make_condition_dir(tmp_path)
    out = tmp_path / "pkg"
    summary = build(cond, out, overwrite=True)
    return cond, out, summary


def test_flat_yolo_layout_and_taxonomy(tmp_path):
    _, out, _ = _run(tmp_path)
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert {int(k): v for k, v in cfg["names"].items()} == CONDITION_NAMES
    assert "panel" not in [v.lower() for v in cfg["names"].values()]
    for split, n in (("train", 3), ("valid", 1), ("test", 2)):
        imgs = list((out / "images" / split).glob("*.jpg"))
        lbls = list((out / "labels" / split).glob("*.txt"))
        assert len(imgs) == n and len(lbls) == n


def test_provenance_and_electrical_warning_carried_into_package(tmp_path):
    _, out, summary = _run(tmp_path)
    assert summary["provenance_grade"] == "C"
    assert summary["provenance_tag"] == "personal-experimental-unverified-provenance"
    assert summary["electrical_damage_semantic_risk"]["flag"] == "HIGH"
    meta = json.loads((out / "dataset-metadata.json").read_text())
    assert meta["licenses"][0]["name"] == "other"           # NOT CC-BY
    assert "HIGH SEMANTIC RISK" in meta["description"]
    assert "pythonafroz" in meta["description"]
    assert 20 <= len(meta["subtitle"]) <= 80
    prov = out / "provenance"
    assert (prov / "condition_dataset_manifest.json").is_file()
    assert (prov / "electrical_damage_annotations.csv").is_file()
    assert (prov / "condition_split_manifest.csv").is_file()
    assert "cell cracks" in (prov / "electrical_damage_annotations.csv").read_text()


def test_family_isolation_rechecked_and_content_hash_deterministic(tmp_path):
    cond, out, s1 = _run(tmp_path)
    assert s1["no_family_crosses_splits"] is True
    s2 = build(cond, out, overwrite=True)
    assert s1["dataset_content_sha256"] == s2["dataset_content_sha256"]
    assert len(s1["dataset_content_sha256"]) == 64


def test_family_crossing_splits_is_rejected(tmp_path):
    import pytest
    cond = _make_condition_dir(tmp_path)
    csv_path = cond / "splits" / "condition_split_manifest.csv"
    text = csv_path.read_text().splitlines()
    # force a test row into fam00001, which a train row already belongs to
    assert ",train,fam00001," in text[1]
    text[5] = text[5].replace(",test,fam00005,", ",test,fam00001,")
    assert ",test,fam00001," in text[5]
    csv_path.write_text("\n".join(text) + "\n")
    with pytest.raises(SystemExit):
        build(cond, tmp_path / "pkg2", overwrite=True)


def test_source_condition_dataset_not_modified(tmp_path):
    cond = _make_condition_dir(tmp_path)
    before = _dir_digest(cond)
    build(cond, tmp_path / "pkg", overwrite=True)
    assert _dir_digest(cond) == before
