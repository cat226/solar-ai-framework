"""Tests for training/classification/create_smoke_dataset.py."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.classification.create_smoke_dataset import create_smoke_dataset


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def source_root(tmp_path):
    root = tmp_path / "prepared"
    records = []
    for split in ("train", "val", "test"):
        for cls in ("Clean", "Dusty", "Hotspot"):
            for i in range(4):
                content = f"{split}-{cls}-{i}".encode()
                fname = f"{cls}_{split}_{i}.jpg"
                src_dir = tmp_path / "raw" / cls
                src_dir.mkdir(parents=True, exist_ok=True)
                src_path = src_dir / fname
                src_path.write_bytes(content)
                records.append({
                    "source_path": str(src_path),
                    "original_filename": fname,
                    "class_name": cls,
                    "sha256": _sha256_bytes(content),
                    "group": None,
                    "split": split,
                })
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({
        "classes": ["Clean", "Dusty", "Hotspot"],
        "is_production_class_set": False,
        "seed": 42,
        "records": records,
    }), encoding="utf-8")
    return root


class TestCreateSmokeDataset:
    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="no manifest.json"):
            create_smoke_dataset(tmp_path / "nope", tmp_path / "out", classes=["Clean"])

    def test_unknown_class_raises(self, source_root, tmp_path):
        with pytest.raises(RuntimeError, match="not present"):
            create_smoke_dataset(source_root, tmp_path / "out", classes=["Bird-Drop"])

    def test_builds_correct_structure(self, source_root, tmp_path):
        out = tmp_path / "smoke"
        manifest_path = create_smoke_dataset(
            source_root, out, classes=["Clean", "Dusty", "Hotspot"], per_class_per_split=2,
        )
        assert manifest_path.is_file()
        for split in ("train", "val", "test"):
            for cls in ("Clean", "Dusty", "Hotspot"):
                files = list((out / split / cls).iterdir())
                assert len(files) == 2, f"{split}/{cls}: {files}"

    def test_manifest_records_source_hash_and_counts(self, source_root, tmp_path):
        out = tmp_path / "smoke"
        manifest_path = create_smoke_dataset(
            source_root, out, classes=["Clean", "Dusty", "Hotspot"], per_class_per_split=2,
        )
        smoke_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert smoke_manifest["classes"] == ["Clean", "Dusty", "Hotspot"]
        assert smoke_manifest["counts"]["train"]["Clean"] == 2
        assert len(smoke_manifest["records"]) == 2 * 3 * 3  # per_class_per_split * classes * splits
        expected_hash = hashlib.sha256((source_root / "manifest.json").read_bytes()).hexdigest()
        assert smoke_manifest["source_manifest_hash"] == expected_hash

    def test_deterministic_across_runs(self, source_root, tmp_path):
        out1 = tmp_path / "smoke1"
        out2 = tmp_path / "smoke2"
        create_smoke_dataset(source_root, out1, classes=["Clean", "Dusty", "Hotspot"], per_class_per_split=2)
        create_smoke_dataset(source_root, out2, classes=["Clean", "Dusty", "Hotspot"], per_class_per_split=2)
        m1 = json.loads((out1 / "manifest.json").read_text(encoding="utf-8"))
        m2 = json.loads((out2 / "manifest.json").read_text(encoding="utf-8"))
        sel1 = sorted((r["split"], r["class_name"], r["sha256"]) for r in m1["records"])
        sel2 = sorted((r["split"], r["class_name"], r["sha256"]) for r in m2["records"])
        assert sel1 == sel2

    def test_insufficient_records_raises(self, source_root, tmp_path):
        with pytest.raises(ValueError, match="requested"):
            create_smoke_dataset(
                source_root, tmp_path / "out", classes=["Clean", "Dusty", "Hotspot"],
                per_class_per_split=100,
            )

    def test_subsets_only_requested_classes(self, source_root, tmp_path):
        """A source manifest that also had Bird-Drop etc. must never leak an
        unrequested class into the smoke subset - only what's explicitly
        asked for."""
        out = tmp_path / "smoke"
        create_smoke_dataset(source_root, out, classes=["Clean", "Hotspot"], per_class_per_split=2)
        assert not (out / "train" / "Dusty").exists()
        assert (out / "train" / "Clean").is_dir()
        assert (out / "train" / "Hotspot").is_dir()
