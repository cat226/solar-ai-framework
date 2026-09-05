"""Tests for training/detection/create_smoke_dataset.py.

Core coverage uses synthetic fixtures (a tiny fake "prepared dataset") so it
runs reliably in CI. A bonus real-data pass against this machine's actual
prepared BDAPPV IGN dataset runs when present, skipping cleanly otherwise
(the real dataset is a local artifact, never committed).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from training.detection.create_smoke_dataset import (
    _sha256_file,
    _validate_label_file,
    create_smoke_dataset,
)

_REAL_PREPARED_ROOT = Path("E:/Solar AI Training Images/yolo_prepared")


def _make_fake_prepared_dataset(root: Path, *, per_split_count: int = 5, class_names=("solar panel",)) -> Path:
    """Build a tiny synthetic prepared-dataset tree + manifest.json matching
    prepare_dataset.py's real output shape closely enough to exercise the
    subsetting logic without needing real images."""
    records = []
    for split in ("train", "val", "test"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(per_split_count):
            image_id = f"{split}-{i:03d}"
            img_path = root / split / "images" / f"{image_id}.png"
            lbl_path = root / split / "labels" / f"{image_id}.txt"
            img_path.write_bytes(f"fake-image-{image_id}".encode())
            has_instance = i % 2 == 0
            if has_instance:
                lbl_path.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            else:
                lbl_path.write_text("", encoding="utf-8")
            records.append({
                "id": image_id,
                "split": split,
                "sha256": hashlib.sha256(img_path.read_bytes()).hexdigest(),
                "num_instances": 1 if has_instance else 0,
                "has_mask": has_instance,
                "image_path": str(img_path),
                "label_path": str(lbl_path),
                "source_shard": "fake-shard.parquet",
            })
    manifest = {
        "source": "fake test source",
        "class_names": list(class_names),
        "split_policy": "fake",
        "counts": {},
        "total_records": len(records),
        "total_anomalies": 0,
        "anomalies": [],
        "records": records,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


class TestLabelValidation:
    def test_empty_label_file_is_valid(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("", encoding="utf-8")
        assert _validate_label_file(f, num_classes=1) == []

    def test_valid_single_box_is_valid(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("0 0.5 0.5 0.1 0.2\n", encoding="utf-8")
        assert _validate_label_file(f, num_classes=1) == []

    def test_wrong_field_count_is_invalid(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("0 0.5 0.5 0.1\n", encoding="utf-8")
        errors = _validate_label_file(f, num_classes=1)
        assert errors and "5 fields" in errors[0]

    def test_class_id_out_of_range_is_invalid(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("5 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        errors = _validate_label_file(f, num_classes=1)
        assert errors and "outside valid range" in errors[0]

    def test_negative_class_id_is_invalid(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("-1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        errors = _validate_label_file(f, num_classes=1)
        assert errors

    @pytest.mark.parametrize("bad_value", ["1.5", "-0.1"])
    def test_coordinate_outside_unit_range_is_invalid(self, tmp_path, bad_value):
        f = tmp_path / "x.txt"
        f.write_text(f"0 {bad_value} 0.5 0.1 0.1\n", encoding="utf-8")
        errors = _validate_label_file(f, num_classes=1)
        assert errors and "outside normalized range" in errors[0]

    def test_zero_width_box_is_invalid(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("0 0.5 0.5 0.0 0.1\n", encoding="utf-8")
        errors = _validate_label_file(f, num_classes=1)
        assert errors and "non-positive box dimensions" in errors[0]

    def test_non_numeric_box_is_invalid(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("0 a b c d\n", encoding="utf-8")
        errors = _validate_label_file(f, num_classes=1)
        assert errors and "not valid floats" in errors[0]


class TestCreateSmokeDataset:
    def test_creates_expected_directory_structure(self, tmp_path):
        source = tmp_path / "source"
        output = tmp_path / "smoke"
        _make_fake_prepared_dataset(source, per_split_count=5)
        create_smoke_dataset(source, output, per_split=3)
        for split in ("train", "val", "test"):
            assert (output / split / "images").is_dir()
            assert (output / split / "labels").is_dir()
        assert (output / "data.yaml").is_file()
        assert (output / "manifest.json").is_file()

    def test_selects_requested_count_per_split(self, tmp_path):
        source = tmp_path / "source"
        output = tmp_path / "smoke"
        _make_fake_prepared_dataset(source, per_split_count=10)
        create_smoke_dataset(source, output, per_split={"train": 3, "val": 2, "test": 1})
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["counts"] == {"train": 3, "val": 2, "test": 1}
        assert len(list((output / "train" / "images").glob("*.png"))) == 3
        assert len(list((output / "val" / "images").glob("*.png"))) == 2
        assert len(list((output / "test" / "images").glob("*.png"))) == 1

    def test_selection_is_deterministic_across_runs(self, tmp_path):
        source = tmp_path / "source"
        _make_fake_prepared_dataset(source, per_split_count=10)
        out1 = tmp_path / "smoke1"
        out2 = tmp_path / "smoke2"
        create_smoke_dataset(source, out1, per_split=4)
        create_smoke_dataset(source, out2, per_split=4)
        m1 = json.loads((out1 / "manifest.json").read_text(encoding="utf-8"))
        m2 = json.loads((out2 / "manifest.json").read_text(encoding="utf-8"))
        ids1 = [r["id"] for r in m1["records"]]
        ids2 = [r["id"] for r in m2["records"]]
        assert ids1 == ids2

    def test_splits_remain_disjoint(self, tmp_path):
        source = tmp_path / "source"
        output = tmp_path / "smoke"
        _make_fake_prepared_dataset(source, per_split_count=10)
        create_smoke_dataset(source, output, per_split=5)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        by_split = {"train": set(), "val": set(), "test": set()}
        for rec in manifest["records"]:
            by_split[rec["split"]].add(rec["sha256"])
        assert not (by_split["train"] & by_split["val"])
        assert not (by_split["train"] & by_split["test"])
        assert not (by_split["val"] & by_split["test"])

    def test_source_manifest_hash_is_recorded_and_correct(self, tmp_path):
        source = tmp_path / "source"
        output = tmp_path / "smoke"
        manifest_path = _make_fake_prepared_dataset(source, per_split_count=5)
        expected_hash = _sha256_file(manifest_path)
        create_smoke_dataset(source, output, per_split=2)
        smoke_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert smoke_manifest["source_manifest_hash"] == expected_hash

    def test_original_dataset_is_unchanged(self, tmp_path):
        source = tmp_path / "source"
        output = tmp_path / "smoke"
        _make_fake_prepared_dataset(source, per_split_count=5)
        original_files = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
        create_smoke_dataset(source, output, per_split=2)
        for path, content in original_files.items():
            assert path.read_bytes() == content, f"source file was modified: {path}"

    def test_uses_class_names_from_source_manifest_not_hardcoded(self, tmp_path):
        source = tmp_path / "source"
        output = tmp_path / "smoke"
        _make_fake_prepared_dataset(source, per_split_count=3, class_names=("a-totally-different-class",))
        create_smoke_dataset(source, output, per_split=1)
        data_yaml = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
        assert data_yaml["names"] == {0: "a-totally-different-class"}

    def test_missing_source_manifest_raises(self, tmp_path):
        source = tmp_path / "does-not-exist"
        with pytest.raises(RuntimeError, match="manifest.json"):
            create_smoke_dataset(source, tmp_path / "smoke", per_split=1)

    def test_missing_source_image_raises(self, tmp_path):
        source = tmp_path / "source"
        output = tmp_path / "smoke"
        _make_fake_prepared_dataset(source, per_split_count=3)
        # Delete an image the manifest still references.
        img = next((source / "train" / "images").glob("*.png"))
        img.unlink()
        with pytest.raises(RuntimeError, match="missing"):
            create_smoke_dataset(source, output, per_split=3)

    def test_invalid_label_in_source_stops_and_reports_not_repairs(self, tmp_path):
        source = tmp_path / "source"
        output = tmp_path / "smoke"
        _make_fake_prepared_dataset(source, per_split_count=3)
        bad_label = next((source / "train" / "labels").glob("*.txt"))
        bad_label.write_text("0 2.0 0.5 0.1 0.1\n", encoding="utf-8")  # out-of-range coordinate
        with pytest.raises(RuntimeError, match="invalid label"):
            create_smoke_dataset(source, output, per_split=3)
        # The output must not have been left in a partially-written state claiming success.
        assert not (output / "manifest.json").is_file()

    def test_requesting_more_than_available_raises(self, tmp_path):
        source = tmp_path / "source"
        _make_fake_prepared_dataset(source, per_split_count=3)
        with pytest.raises(ValueError, match="only 3"):
            create_smoke_dataset(source, tmp_path / "smoke", per_split=100)

    def test_unknown_split_key_raises(self, tmp_path):
        source = tmp_path / "source"
        _make_fake_prepared_dataset(source, per_split_count=3)
        with pytest.raises(ValueError, match="unknown split"):
            create_smoke_dataset(source, tmp_path / "smoke", per_split={"bogus": 1})


@pytest.mark.skipif(not _REAL_PREPARED_ROOT.is_dir(), reason="real prepared BDAPPV dataset not present on this machine")
class TestRealPreparedDataset:
    """Bonus coverage against the actual audited BDAPPV IGN dataset, when present."""

    def test_real_dataset_produces_valid_smoke_subset(self, tmp_path):
        output = tmp_path / "smoke"
        create_smoke_dataset(_REAL_PREPARED_ROOT, output, per_split=30)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["counts"] == {"train": 30, "val": 30, "test": 30}
        assert manifest["class_names"] == ["solar panel"]
        # Every selected label must independently re-validate clean.
        for split in ("train", "val", "test"):
            for lbl in (output / split / "labels").glob("*.txt"):
                assert _validate_label_file(lbl, num_classes=1) == []
