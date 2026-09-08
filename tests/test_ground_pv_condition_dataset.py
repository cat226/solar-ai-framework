"""tests/test_ground_pv_condition_dataset.py — the unified condition/defect
research dataset build (S1 + S3 -> CLEAN/DUSTY/BIRD_DROP/PHYSICAL_DAMAGE/
ELECTRICAL_DAMAGE).

PERSONAL, NON-COMMERCIAL RESEARCH ONLY — these tests also guard that the build
never touches the immutable raw sources or the existing ground detector dataset.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from PIL import Image

from training.detection.ground_pv.build_condition_dataset import (
    CONDITION_CLASSES,
    PROVENANCE_GRADE,
    PROVENANCE_TAG,
    _ALIAS,
    build,
)


def _grad(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (48, 48))
    for x in range(48):
        for y in range(48):
            im.putpixel((x, y), ((x * 5 + seed) % 256, (y * 3 + seed) % 256, (x + y + seed) % 256))
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


def _make_sources(tmp_path: Path) -> tuple[Path, Path]:
    s1 = tmp_path / "S1"
    _write(s1 / "data.yaml", yaml.safe_dump({
        "names": ["bird-drop", "clean", "dusty", "electrical-damage", "physical-damage", "snow-covered"],
        "nc": 6,
        "roboflow": {"workspace": "defects-kgig6", "project": "detection-for-defects-in-solar-panels",
                     "version": 2, "license": "CC BY 4.0"},
    }))
    # train
    _grad(s1 / "train/images/clean_1.jpg", 1)
    _write(s1 / "train/labels/clean_1.txt", "1 0.5 0.5 0.3 0.3\n")
    _grad(s1 / "train/images/elec_1.jpg", 2)
    _write(s1 / "train/labels/elec_1.txt", "3 0.4 0.4 0.2 0.2\n1 0.7 0.7 0.1 0.1\n")
    _grad(s1 / "train/images/snowonly_1.jpg", 3)
    _write(s1 / "train/labels/snowonly_1.txt", "5 0.5 0.5 0.4 0.4\n")
    _grad(s1 / "train/images/malformed_1.jpg", 4)
    _write(s1 / "train/labels/malformed_1.txt", "2 0.1 0.2 0.3\n0 0.5 0.5 0.1 0.1\n")
    _grad(s1 / "train/images/dupe.jpg", 5)
    _write(s1 / "train/labels/dupe.txt", "2 0.5 0.5 0.2 0.2\n")
    _grad(s1 / "train/images/phys_1.jpg", 6)
    _write(s1 / "train/labels/phys_1.txt", "4 0.5 0.5 0.3 0.3\n")
    # valid
    _grad(s1 / "valid/images/dusty_1.jpg", 7)
    _write(s1 / "valid/labels/dusty_1.txt", "2 0.5 0.5 0.25 0.25\n")
    # test
    _grad(s1 / "test/images/bird_1.jpg", 8)
    _write(s1 / "test/labels/bird_1.txt", "0 0.5 0.5 0.2 0.2\n")

    s3 = tmp_path / "S3"
    _write(s3 / "data.yaml", yaml.safe_dump({
        "names": ["bird-droppings", "dusdusty"], "nc": 2,
        "roboflow": {"workspace": "solar-panel-a5cyn", "project": "dusty-and-bird-droppings",
                     "version": 1, "license": "CC BY 4.0"},
    }))
    _grad(s3 / "train/images/Imgdirty_1.jpg", 20)
    _write(s3 / "train/labels/Imgdirty_1.txt", "1 0.5 0.5 0.3 0.3\n")
    _grad(s3 / "train/images/Imgclean_1.jpg", 21)
    _write(s3 / "train/labels/Imgclean_1.txt", "0 0.4 0.4 0.2 0.2\n" + "9 " * 17 + "\n")  # 2nd line malformed
    # byte-identical copy of S1/train/dupe.jpg -> must collapse to one family + dedup
    (s3 / "valid/images").mkdir(parents=True, exist_ok=True)
    (s3 / "valid/labels").mkdir(parents=True, exist_ok=True)
    (s3 / "valid/images/dupe_copy.jpg").write_bytes((s1 / "train/images/dupe.jpg").read_bytes())
    _write(s3 / "valid/labels/dupe_copy.txt", "1 0.5 0.5 0.2 0.2\n")
    return s1, s3


def _fake_existing_yolo(tmp_path: Path) -> Path:
    d = tmp_path / "existing_yolo" / "images" / "train"
    _grad(d / "unrelated_a.jpg", 200)
    _grad(d / "unrelated_b.jpg", 201)
    return tmp_path / "existing_yolo"


def _run(tmp_path: Path):
    s1, s3 = _make_sources(tmp_path)
    existing = _fake_existing_yolo(tmp_path)
    rep = build(s1, s3, tmp_path / "condition", existing_yolo_dir=existing,
                seed=42, dhash16_merge_hamming=0, overwrite=True)
    out = tmp_path / "condition"
    manifest = json.loads((out / "manifests" / "condition_dataset_manifest.json").read_text())
    return s1, s3, existing, out, rep, manifest


# ---- 1. source traceability -------------------------------------------------

def test_every_derived_image_traces_to_a_source(tmp_path):
    _, _, _, _, _, manifest = _run(tmp_path)
    assert manifest["manifest_rows"]
    for r in manifest["manifest_rows"]:
        assert r["source"] in {"S1", "S3"}
        assert r["source_dataset"] and r["source_filename"]
        assert len(r["source_sha256"]) == 64
        assert r["family_id"].startswith("fam")
        assert r["source_split"] in {"train", "valid", "test"}


# ---- 2. taxonomy mapping --------------------------------------------------

def test_taxonomy_is_the_five_conditions_and_snow_is_dropped(tmp_path):
    _, _, _, out, _, manifest = _run(tmp_path)
    dy = yaml.safe_load((out / "data.yaml").read_text())
    assert list(dy["names"].values()) == list(CONDITION_CLASSES)
    assert _ALIAS["snow-covered"] is None and _ALIAS["snow"] is None
    assert _ALIAS["dusdusty"] == "DUSTY" and _ALIAS["bird-droppings"] == "BIRD_DROP"
    # no snow class anywhere in the output
    all_classes = set(manifest["report"]["class_image_counts"]["total"])
    assert not any("snow" in c.lower() for c in all_classes)
    assert "Defective" not in all_classes


def test_snow_only_image_is_dropped_and_counted(tmp_path):
    _, _, _, out, rep, manifest = _run(tmp_path)
    names = {r["derived_filename"] for r in manifest["manifest_rows"]}
    assert "snowonly_1.jpg" not in names
    assert rep.snow_only_images_dropped >= 1
    assert rep.snow_boxes_dropped >= 1


# ---- 3. provenance flags -------------------------------------------------

def test_provenance_grade_c_and_tag_on_every_row(tmp_path):
    _, _, _, _, _, manifest = _run(tmp_path)
    assert manifest["provenance_grade"] == PROVENANCE_GRADE == "C"
    assert manifest["provenance_tag"] == PROVENANCE_TAG == "personal-experimental-unverified-provenance"
    assert manifest["personal_non_commercial_research_only"] is True
    for r in manifest["manifest_rows"]:
        assert r["provenance_grade"] == "C"
        assert r["provenance_tag"] == PROVENANCE_TAG


# ---- 4. no split leakage -------------------------------------------------

def test_no_family_and_no_exact_hash_crosses_splits(tmp_path):
    _, _, _, _, rep, manifest = _run(tmp_path)
    assert rep.no_family_crosses_splits is True
    assert rep.crossing_families == {}
    by_fam: dict[str, set[str]] = {}
    by_sha: dict[str, set[str]] = {}
    for r in manifest["manifest_rows"]:
        by_fam.setdefault(r["family_id"], set()).add(r["derived_split"])
        by_sha.setdefault(r["source_sha256"], set()).add(r["derived_split"])
    assert all(len(v) == 1 for v in by_fam.values())
    assert all(len(v) == 1 for v in by_sha.values())
    assert manifest["report"]["quality_gates"]["2_no_family_crosses_splits"] is True


# ---- 5. duplicate removal ----------------------------------------------

def test_byte_identical_image_is_deduplicated(tmp_path):
    _, _, _, _, rep, manifest = _run(tmp_path)
    assert rep.exact_duplicate_removed_within >= 1
    shas = [r["source_sha256"] for r in manifest["manifest_rows"]]
    assert len(shas) == len(set(shas))  # no exact duplicate survives


# ---- 6. invalid annotation rejection ---------------------------------

def test_malformed_annotation_lines_are_rejected_not_fabricated(tmp_path):
    _, _, _, out, rep, manifest = _run(tmp_path)
    assert rep.malformed_lines_dropped >= 2  # the 4-field S1 line + the 17-field S3 line
    # malformed_1: the 4-field line is dropped, the 1 valid box (class 0 bird-drop) survives
    row = next(r for r in manifest["manifest_rows"] if r["derived_filename"] == "malformed_1.jpg")
    lbl = (out / "images" / row["derived_split"] / "labels" / "malformed_1.txt").read_text().split()
    assert len(lbl) == 5 and lbl[0] == str(CONDITION_CLASSES.index("BIRD_DROP"))
    # every output label line is well-formed
    assert manifest["report"]["quality_gates"]["4_no_malformed_annotations_in_output"] is True


# ---- 7. electrical-damage semantic-risk preservation ----------------

def test_electrical_damage_flagged_high_risk_and_source_class_preserved(tmp_path):
    _, _, _, out, rep, manifest = _run(tmp_path)
    assert rep.electrical_damage_images >= 1
    assert rep.electrical_damage_semantic_risk == "HIGH"
    row = next(r for r in manifest["manifest_rows"] if r["derived_filename"] == "elec_1.jpg")
    assert row["electrical_damage_semantic_risk"] == "HIGH"
    assert "electrical-damage" in row["original_classes"]  # original source class preserved

    csv_path = out / "manifests" / "electrical_damage_annotations.csv"
    body = csv_path.read_text()
    assert "HIGH" in body and "electrical-damage" in body
    assert "cell cracks" in body  # the audit caveat travels with the annotation
    assert manifest["report"]["quality_gates"]["9_electrical_damage_semantic_risk_preserved"] is True


# ---- 8. raw source immutability ------------------------------------

def test_raw_sources_are_byte_identical_after_build(tmp_path):
    s1, s3 = _make_sources(tmp_path)
    existing = _fake_existing_yolo(tmp_path)
    before = (_dir_digest(s1), _dir_digest(s3))
    build(s1, s3, tmp_path / "condition", existing_yolo_dir=existing,
          seed=42, dhash16_merge_hamming=0, overwrite=True)
    assert (_dir_digest(s1), _dir_digest(s3)) == before


# ---- 9. existing ground dataset immutability ---------------------

def test_existing_ground_yolo_dataset_untouched(tmp_path):
    s1, s3 = _make_sources(tmp_path)
    existing = _fake_existing_yolo(tmp_path)
    before = _dir_digest(existing)
    build(s1, s3, tmp_path / "condition", existing_yolo_dir=existing,
          seed=42, dhash16_merge_hamming=0, overwrite=True)
    assert _dir_digest(existing) == before
