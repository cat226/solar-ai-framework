"""tests/test_ground_pv_condition_4class_dataset.py — Phase 15A conversion
of the existing 5-class research condition dataset into the active 4-class
dataset (ELECTRICAL_DAMAGE removed, never relabelled).

Uses a small SYNTHETIC 5-class dataset fixture (not the real 2,509-image
dataset) so tests are fast and self-contained; the real conversion is
exercised for real by scripts/build_condition_4class_dataset.py (see the
Phase 15A final report for the actual counts).
"""
from __future__ import annotations

import csv

import pytest
from PIL import Image

from training.detection.ground_pv.build_condition_4class_dataset import build
from training.detection.ground_pv.condition_taxonomy import CONDITION_4CLASS_ORDER


def _make_5class_dataset(root, images_by_split: dict[str, list[tuple[str, list[str]]]]):
    """images_by_split: split -> [(filename, [label_lines]), ...]. Writes a
    minimal but structurally faithful 5-class dataset (images/<split>/{images,labels},
    splits/condition_split_manifest.csv, data.yaml)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "path: x\ntrain: images/train/images\nval: images/valid/images\ntest: images/test/images\n"
        "names:\n  0: CLEAN\n  1: DUSTY\n  2: BIRD_DROP\n  3: PHYSICAL_DAMAGE\n  4: ELECTRICAL_DAMAGE\n",
        encoding="utf-8",
    )
    manifest_rows = []
    for split, entries in images_by_split.items():
        images_dir = root / "images" / split / "images"
        labels_dir = root / "images" / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for i, (filename, lines) in enumerate(entries):
            Image.new("RGB", (100, 100), (i, i, i)).save(images_dir / filename)
            (labels_dir / f"{filename.rsplit('.', 1)[0]}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8",
            )
            manifest_rows.append({
                "derived_filename": filename, "derived_split": split,
                "family_id": f"fam{i:03d}_{split}", "source": "S1",
                "source_sha256": f"sha_{split}_{i}",
                "electrical_damage_semantic_risk": "HIGH" if any(l.startswith("4 ") for l in lines) else "",
            })
    (root / "splits").mkdir(exist_ok=True)
    fieldnames = ["derived_filename", "derived_split", "family_id", "source", "source_sha256",
                 "electrical_damage_semantic_risk"]
    with (root / "splits" / "condition_split_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest_rows)
    return root


class TestConditionDatasetConversion:
    def test_electrical_damage_boxes_removed_not_relabelled(self, tmp_path):
        src = _make_5class_dataset(tmp_path / "source", {
            "train": [("a.jpg", ["0 0.1 0.1 0.1 0.1", "4 0.5 0.5 0.1 0.1"])],  # CLEAN + ELECTRICAL_DAMAGE
            "valid": [], "test": [],
        })
        rep = build(src, tmp_path / "derived")
        label_text = (tmp_path / "derived" / "images" / "train" / "labels" / "a.txt").read_text(encoding="utf-8")
        assert "4 " not in label_text  # electrical_damage line gone
        assert "0 0.1 0.1 0.1 0.1" in label_text  # CLEAN line preserved verbatim
        # crucially: NOT relabelled as PHYSICAL_DAMAGE (class 3)
        assert "3 0.5 0.5 0.1 0.1" not in label_text
        assert rep.electrical_damage_instances_removed == 1

    def test_all_electrical_damage_image_dropped(self, tmp_path):
        src = _make_5class_dataset(tmp_path / "source", {
            "train": [("only_elec.jpg", ["4 0.5 0.5 0.2 0.2"])],
            "valid": [], "test": [],
        })
        rep = build(src, tmp_path / "derived")
        assert rep.electrical_damage_only_images_dropped == 1
        assert rep.derived_image_count == 0
        assert not (tmp_path / "derived" / "images" / "train" / "images" / "only_elec.jpg").exists()

    def test_mixed_image_kept_with_reduced_boxes(self, tmp_path):
        src = _make_5class_dataset(tmp_path / "source", {
            "train": [("mixed.jpg", ["1 0.2 0.2 0.1 0.1", "4 0.5 0.5 0.1 0.1", "4 0.6 0.6 0.1 0.1"])],
            "valid": [], "test": [],
        })
        rep = build(src, tmp_path / "derived")
        assert rep.derived_image_count == 1
        assert rep.images_with_electrical_damage_partially_removed == 1
        assert rep.electrical_damage_instances_removed == 2
        label_text = (tmp_path / "derived" / "images" / "train" / "labels" / "mixed.txt").read_text(encoding="utf-8")
        assert label_text.strip() == "1 0.2 0.2 0.1 0.1"

    def test_image_with_no_electrical_damage_passes_through_unchanged(self, tmp_path):
        src = _make_5class_dataset(tmp_path / "source", {
            "train": [("clean_only.jpg", ["0 0.5 0.5 0.2 0.2"])],
            "valid": [], "test": [],
        })
        original_bytes = (src / "images" / "train" / "images" / "clean_only.jpg").read_bytes()
        rep = build(src, tmp_path / "derived")
        derived_bytes = (tmp_path / "derived" / "images" / "train" / "images" / "clean_only.jpg").read_bytes()
        assert derived_bytes == original_bytes  # byte-identical, no re-encode
        label_text = (tmp_path / "derived" / "images" / "train" / "labels" / "clean_only.txt").read_text(encoding="utf-8")
        assert label_text.strip() == "0 0.5 0.5 0.2 0.2"

    def test_split_assignment_preserved_exactly(self, tmp_path):
        src = _make_5class_dataset(tmp_path / "source", {
            "train": [("t.jpg", ["0 0.5 0.5 0.2 0.2"])],
            "valid": [("v.jpg", ["1 0.5 0.5 0.2 0.2"])],
            "test": [("te.jpg", ["2 0.5 0.5 0.2 0.2"])],
        })
        rep = build(src, tmp_path / "derived")
        assert rep.final_split_counts == {"train": 1, "valid": 1, "test": 1}
        assert (tmp_path / "derived" / "images" / "train" / "images" / "t.jpg").exists()
        assert (tmp_path / "derived" / "images" / "valid" / "images" / "v.jpg").exists()
        assert (tmp_path / "derived" / "images" / "test" / "images" / "te.jpg").exists()

    def test_source_dataset_never_mutated(self, tmp_path):
        src = _make_5class_dataset(tmp_path / "source", {
            "train": [("a.jpg", ["4 0.5 0.5 0.1 0.1"])], "valid": [], "test": [],
        })
        original = (src / "images" / "train" / "labels" / "a.txt").read_text(encoding="utf-8")
        rep = build(src, tmp_path / "derived")
        after = (src / "images" / "train" / "labels" / "a.txt").read_text(encoding="utf-8")
        assert original == after
        assert rep.source_tree_unchanged is True

    def test_data_yaml_is_four_class(self, tmp_path):
        src = _make_5class_dataset(tmp_path / "source", {
            "train": [("a.jpg", ["0 0.5 0.5 0.1 0.1"])], "valid": [], "test": [],
        })
        build(src, tmp_path / "derived")
        import yaml
        data = yaml.safe_load((tmp_path / "derived" / "data.yaml").read_text(encoding="utf-8"))
        assert data["names"] == {0: "CLEAN", 1: "DUSTY", 2: "BIRD_DROP", 3: "PHYSICAL_DAMAGE"}
        assert "ELECTRICAL_DAMAGE" not in data["names"].values()

    def test_quality_gates_all_pass_on_clean_conversion(self, tmp_path):
        src = _make_5class_dataset(tmp_path / "source", {
            "train": [("a.jpg", ["0 0.5 0.5 0.1 0.1"])],
            "valid": [("b.jpg", ["1 0.5 0.5 0.1 0.1"])],
            "test": [],
        })
        rep = build(src, tmp_path / "derived")
        assert rep.quality_gates["no_electrical_damage_class_present"] is True
        assert rep.quality_gates["no_snow_class_present"] is True
        assert rep.quality_gates["no_defective_class_present"] is True
        assert rep.quality_gates["no_panel_class_present"] is True
        assert rep.quality_gates["no_family_crosses_splits"] is True
        assert rep.quality_gates["all_pass"] is True

    def test_refuses_to_overwrite_without_flag(self, tmp_path):
        src = _make_5class_dataset(tmp_path / "source", {
            "train": [("a.jpg", ["0 0.5 0.5 0.1 0.1"])], "valid": [], "test": [],
        })
        build(src, tmp_path / "derived")
        with pytest.raises(FileExistsError):
            build(src, tmp_path / "derived")
        build(src, tmp_path / "derived", overwrite=True)  # should not raise

    def test_dropped_images_manifest_written(self, tmp_path):
        src = _make_5class_dataset(tmp_path / "source", {
            "train": [("dropped.jpg", ["4 0.5 0.5 0.1 0.1"]), ("kept.jpg", ["0 0.5 0.5 0.1 0.1"])],
            "valid": [], "test": [],
        })
        build(src, tmp_path / "derived")
        dropped_csv = (tmp_path / "derived" / "manifests" / "dropped_images.csv").read_text(encoding="utf-8")
        assert "dropped.jpg" in dropped_csv
        assert "kept.jpg" not in dropped_csv

    def test_empty_source_produces_empty_derived_dataset(self, tmp_path):
        src = _make_5class_dataset(tmp_path / "source", {"train": [], "valid": [], "test": []})
        rep = build(src, tmp_path / "derived")
        assert rep.derived_image_count == 0
        assert rep.final_split_counts == {"train": 0, "valid": 0, "test": 0}
