"""Tests for MobileNet training pipeline class mapping and dataset preparation."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from torchvision import datasets, transforms

from training.classification.evaluate_mobilenet import _map_dataset_to_production, CLASSES
from training.classification.prepare_dataset import (
    REQUIRED_CLASSES,
    SPLITS,
    prepare_dataset,
    _collect_images,
    _split_records,
)
from training.classification.train_mobilenet import _dataset as _train_dataset


def _unique_color(cls: str, idx: int) -> tuple[int, int, int]:
    """Deterministic unique color for test image generation.

    Uses hashlib, not Python's built-in ``hash()`` - the latter is salted
    per-process (PYTHONHASHSEED) precisely so it's *not* reproducible across
    runs, which occasionally mapped two different class names to the same
    ``% 256`` base value and produced two SHA-256-identical synthetic images,
    intermittently tripping prepare_dataset's real duplicate-image guard in
    CI and locally (observed for Electrical-Damage/Physical-Damage). This
    reproduces identically on every run, process, and machine, and is
    verified collision-free for every real class name this suite uses.
    """
    base = int(hashlib.sha256(cls.encode()).hexdigest(), 16) % 256
    r = (base + idx * 37) % 256
    g = (base + 128 + idx * 53) % 256
    b = (base + 64 + idx * 71) % 256
    return (r, g, b)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_dataset_root(tmp_path: Path) -> Path:
    """Create a fake ImageFolder dataset with alphabetical class ordering."""
    class_names = sorted(REQUIRED_CLASSES)
    for cls in class_names:
        (tmp_path / cls).mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (32, 32), _unique_color(cls, 0))
        img.save(tmp_path / cls / f"{cls}_1.jpg")

    return tmp_path


# ---------------------------------------------------------------------------
# Class mapping / ImageFolder ordering
# ---------------------------------------------------------------------------

class TestClassMapping:
    """ImageFolder alphabetical ordering must not silently alter production labels."""

    def test_imagefolder_alphabetical_order_detected(self, fake_dataset_root: Path):
        """ImageFolder sorts classes alphabetically, which differs from production order."""
        ds = datasets.ImageFolder(fake_dataset_root, transform=transforms.ToTensor())
        # ImageFolder alphabetical order
        expected_alphabetical = sorted(REQUIRED_CLASSES)
        assert ds.classes == expected_alphabetical
        # Production order is different
        assert ds.classes != CLASSES

    def test_map_dataset_enforces_production_order(self, fake_dataset_root: Path):
        """After mapping, dataset classes and targets use production order."""
        raw_ds = datasets.ImageFolder(fake_dataset_root, transform=transforms.ToTensor())
        original_targets = list(raw_ds.targets)  # save before mapping
        mapped = _map_dataset_to_production(raw_ds)

        assert mapped.classes == CLASSES
        assert mapped.class_to_idx == {name: idx for idx, name in enumerate(CLASSES)}

        # Targets should be remapped from alphabetical to production indices
        for original_target, expected_target in zip(original_targets, mapped.targets):
            original_name = raw_ds.classes[original_target]
            expected_idx = CLASSES.index(original_name)
            assert expected_target == expected_idx

    def test_missing_class_raises(self, tmp_path: Path):
        """Dataset missing a required class must fail loudly."""
        (tmp_path / "Clean").mkdir(parents=True)
        (tmp_path / "Dusty").mkdir(parents=True)
        for cls in ["Clean", "Dusty"]:
            img = Image.new("RGB", (32, 32), _unique_color(cls, 0))
            img.save(tmp_path / cls / f"{cls}_1.jpg")

        raw_ds = datasets.ImageFolder(tmp_path, transform=transforms.ToTensor())
        with pytest.raises(RuntimeError, match="dataset classes must exactly equal"):
            _map_dataset_to_production(raw_ds)

    def test_unknown_class_raises(self, tmp_path: Path):
        """Dataset with an unknown class must fail loudly."""
        for cls in REQUIRED_CLASSES + ["Snow-Covered"]:
            (tmp_path / cls).mkdir(parents=True, exist_ok=True)
            img = Image.new("RGB", (32, 32), _unique_color(cls, 0))
            img.save(tmp_path / cls / f"{cls}_1.jpg")

        raw_ds = datasets.ImageFolder(tmp_path, transform=transforms.ToTensor())
        with pytest.raises(RuntimeError, match="dataset classes must exactly equal"):
            _map_dataset_to_production(raw_ds)

    def test_snow_covered_not_mapped_to_hotspot(self, tmp_path: Path):
        """Snow-Covered must be rejected, not silently mapped to Hotspot."""
        for cls in REQUIRED_CLASSES + ["Snow-Covered"]:
            (tmp_path / cls).mkdir(parents=True, exist_ok=True)
            img = Image.new("RGB", (32, 32), _unique_color(cls, 0))
            img.save(tmp_path / cls / f"{cls}_1.jpg")

        raw_ds = datasets.ImageFolder(tmp_path, transform=transforms.ToTensor())
        with pytest.raises(RuntimeError, match="dataset classes must exactly equal"):
            _map_dataset_to_production(raw_ds)

    def test_map_dataset_supports_explicit_class_subset(self, tmp_path: Path):
        """An interim (non-production) class subset maps to its own order, not the full six."""
        subset = ["Clean", "Dusty", "Hotspot"]
        for cls in subset:
            (tmp_path / cls).mkdir(parents=True)
            img = Image.new("RGB", (32, 32), _unique_color(cls, 0))
            img.save(tmp_path / cls / f"{cls}_1.jpg")

        raw_ds = datasets.ImageFolder(tmp_path, transform=transforms.ToTensor())
        mapped = _map_dataset_to_production(raw_ds, subset)

        assert mapped.classes == subset
        assert mapped.class_to_idx == {name: idx for idx, name in enumerate(subset)}
        assert len(mapped) == len(subset)

    def test_map_dataset_subset_rejects_mismatched_folders(self, tmp_path: Path):
        """A subset dataset containing a class outside the requested subset must still fail closed."""
        for cls in ["Clean", "Dusty", "Hotspot"]:
            (tmp_path / cls).mkdir(parents=True)
            img = Image.new("RGB", (32, 32), _unique_color(cls, 0))
            img.save(tmp_path / cls / f"{cls}_1.jpg")

        raw_ds = datasets.ImageFolder(tmp_path, transform=transforms.ToTensor())
        with pytest.raises(RuntimeError, match="dataset classes must exactly equal"):
            _map_dataset_to_production(raw_ds, ["Clean", "Dusty"])

    def test_train_dataset_supports_explicit_class_subset(self, tmp_path: Path):
        """train_mobilenet's _dataset() helper supports the same subset override."""
        subset = ["Clean", "Dusty", "Hotspot"]
        for cls in subset:
            (tmp_path / cls).mkdir(parents=True)
            img = Image.new("RGB", (32, 32), _unique_color(cls, 0))
            img.save(tmp_path / cls / f"{cls}_1.jpg")

        ds = _train_dataset(tmp_path, transforms.ToTensor(), subset)
        assert set(ds.targets) == {0, 1, 2}
        assert max(ds.targets) == len(subset) - 1

    def test_map_dataset_getitem_returns_remapped_target_not_just_attribute(self, tmp_path: Path):
        """Regression: ImageFolder.__getitem__ reads self.samples, not self.targets -
        reassigning .targets alone does not change what a DataLoader actually yields.
        A non-alphabetical requested order must produce correct __getitem__ targets."""
        requested = ["Dusty", "Clean", "Hotspot"]  # deliberately not alphabetical
        for cls in requested:
            (tmp_path / cls).mkdir(parents=True)
            img = Image.new("RGB", (32, 32), _unique_color(cls, 0))
            img.save(tmp_path / cls / f"{cls}_1.jpg")

        raw_ds = datasets.ImageFolder(tmp_path, transform=transforms.ToTensor())
        mapped = _map_dataset_to_production(raw_ds, requested)

        for idx in range(len(mapped)):
            _, actual_target = mapped[idx]  # what a DataLoader really sees
            assert actual_target == mapped.targets[idx], (
                f"__getitem__ target {actual_target} does not match .targets "
                f"{mapped.targets[idx]} at index {idx} - .targets was reassigned "
                "but __getitem__ still reads the original alphabetical-order label"
            )
            # Cross-check against the source-of-truth: which folder the file came from.
            src_class = raw_ds.classes[raw_ds.targets[idx]]
            assert actual_target == requested.index(src_class)

    def test_train_dataset_getitem_returns_remapped_target_not_just_attribute(self, tmp_path: Path):
        """Same regression as above, for train_mobilenet.py's _dataset()/_RemappedImageFolder."""
        requested = ["Dusty", "Clean", "Hotspot"]  # deliberately not alphabetical
        for cls in requested:
            (tmp_path / cls).mkdir(parents=True)
            img = Image.new("RGB", (32, 32), _unique_color(cls, 0))
            img.save(tmp_path / cls / f"{cls}_1.jpg")

        ds = _train_dataset(tmp_path, transforms.ToTensor(), requested)
        raw_ds = datasets.ImageFolder(tmp_path, transform=transforms.ToTensor())

        for idx in range(len(ds)):
            _, actual_target = ds[idx]  # what a DataLoader really sees
            assert actual_target == ds.targets[idx]
            src_class = raw_ds.classes[raw_ds.targets[idx]]
            assert actual_target == requested.index(src_class)


# ---------------------------------------------------------------------------
# prepare_dataset.py
# ---------------------------------------------------------------------------

class TestPrepareDataset:
    """Dataset preparation utility."""

    def test_requires_all_six_classes(self, tmp_path: Path):
        """Missing required class must raise."""
        source = tmp_path / "source"
        for cls in ["Clean", "Dusty"]:
            (source / cls).mkdir(parents=True)
            img = Image.new("RGB", (32, 32), _unique_color(cls, 0))
            img.save(source / cls / f"{cls}_1.jpg")

        output = tmp_path / "output"
        with pytest.raises(RuntimeError, match="missing required classes"):
            prepare_dataset([source], output, seed=42)

    def test_rejects_unknown_classes(self, tmp_path: Path):
        """Unknown class in source must raise."""
        source = tmp_path / "source"
        for cls in REQUIRED_CLASSES + ["Snow-Covered"]:
            (source / cls).mkdir(parents=True, exist_ok=True)
            img = Image.new("RGB", (32, 32), _unique_color(cls, 0))
            img.save(source / cls / f"{cls}_1.jpg")

        output = tmp_path / "output"
        with pytest.raises(RuntimeError, match="unknown classes found in source"):
            prepare_dataset([source], output, seed=42)

    def test_deterministic_split(self, tmp_path: Path):
        """Same seed produces identical splits."""
        source = tmp_path / "source"
        for cls in REQUIRED_CLASSES:
            (source / cls).mkdir(parents=True)
            for i in range(5):
                img = Image.new("RGB", (32, 32), _unique_color(cls, i))
                img.save(source / cls / f"{cls}_{i}.jpg")

        output1 = tmp_path / "out1"
        output2 = tmp_path / "out2"
        prepare_dataset([source], output1, seed=42)
        prepare_dataset([source], output2, seed=42)

        manifest1 = json.loads((output1 / "manifest.json").read_text())
        manifest2 = json.loads((output2 / "manifest.json").read_text())

        for rec1, rec2 in zip(manifest1["records"], manifest2["records"]):
            assert rec1["split"] == rec2["split"]
            assert rec1["sha256"] == rec2["sha256"]

    def test_different_seed_produces_different_split(self, tmp_path: Path):
        """Different seeds can produce different splits."""
        source = tmp_path / "source"
        for cls in REQUIRED_CLASSES:
            (source / cls).mkdir(parents=True)
            for i in range(10):
                img = Image.new("RGB", (32, 32), _unique_color(cls, i))
                img.save(source / cls / f"{cls}_{i}.jpg")

        output1 = tmp_path / "out1"
        output2 = tmp_path / "out2"
        prepare_dataset([source], output1, seed=42)
        prepare_dataset([source], output2, seed=99)

        manifest1 = json.loads((output1 / "manifest.json").read_text())
        manifest2 = json.loads((output2 / "manifest.json").read_text())

        splits1 = [rec["split"] for rec in manifest1["records"]]
        splits2 = [rec["split"] for rec in manifest2["records"]]
        assert splits1 != splits2

    def test_duplicate_detection(self, tmp_path: Path):
        """Duplicate images across classes must raise."""
        source = tmp_path / "source"
        (source / "Clean").mkdir(parents=True)
        (source / "Dusty").mkdir(parents=True)

        img = Image.new("RGB", (32, 32), _unique_color("Dusty", 0))
        buf1 = tmp_path / "img1.jpg"
        buf2 = tmp_path / "img2.jpg"
        img.save(buf1)
        img.save(buf2)
        shutil.copy(buf1, source / "Clean" / "clean_1.jpg")
        shutil.copy(buf2, source / "Dusty" / "dirty_1.jpg")

        output = tmp_path / "output"
        with pytest.raises(RuntimeError, match="duplicate image detected"):
            prepare_dataset([source], output, seed=42)

    def test_no_duplicate_across_splits(self, tmp_path: Path):
        """Same image must not appear in multiple splits."""
        source = tmp_path / "source"
        for cls in REQUIRED_CLASSES:
            (source / cls).mkdir(parents=True)
            for i in range(10):
                img = Image.new("RGB", (32, 32), _unique_color(cls, i))
                img.save(source / cls / f"{cls}_{i}.jpg")

        output = tmp_path / "output"
        prepare_dataset([source], output, seed=42)

        hashes_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
        manifest = json.loads((output / "manifest.json").read_text())
        for rec in manifest["records"]:
            hashes_by_split[rec["split"]].add(rec["sha256"])

        for split_a, split_b in [("train", "val"), ("train", "test"), ("val", "test")]:
            assert not hashes_by_split[split_a].intersection(hashes_by_split[split_b])

    def test_produces_expected_directory_structure(self, tmp_path: Path):
        """Output must contain train/val/test with all six classes."""
        source = tmp_path / "source"
        for cls in REQUIRED_CLASSES:
            (source / cls).mkdir(parents=True)
            for i in range(5):
                img = Image.new("RGB", (32, 32), _unique_color(cls, i))
                img.save(source / cls / f"{cls}_{i}.jpg")

        output = tmp_path / "output"
        prepare_dataset([source], output, seed=42)

        for split in SPLITS:
            for cls in REQUIRED_CLASSES:
                assert (output / split / cls).is_dir()

    def test_manifest_contains_provenance(self, tmp_path: Path):
        """Manifest must contain source paths, SHA-256, splits, and classes."""
        source = tmp_path / "source"
        for cls in REQUIRED_CLASSES:
            (source / cls).mkdir(parents=True)
            img = Image.new("RGB", (32, 32), _unique_color(cls, 0))
            img.save(source / cls / f"{cls}_1.jpg")

        output = tmp_path / "output"
        prepare_dataset([source], output, seed=42)

        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["source_roots"] == [str(source.resolve())]
        assert manifest["seed"] == 42
        assert manifest["classes"] == REQUIRED_CLASSES
        assert len(manifest["records"]) == len(REQUIRED_CLASSES)

        for rec in manifest["records"]:
            assert "source_path" in rec
            assert "sha256" in rec
            assert "split" in rec
            assert "class_name" in rec
            assert rec["class_name"] in REQUIRED_CLASSES
            assert rec["split"] in SPLITS

    def test_no_class_dominates_single_split(self, tmp_path: Path):
        """No single class should be entirely confined to one split when there are enough samples."""
        source = tmp_path / "source"
        for cls in REQUIRED_CLASSES:
            (source / cls).mkdir(parents=True)
            for i in range(20):
                img = Image.new("RGB", (32, 32), _unique_color(cls, i))
                img.save(source / cls / f"{cls}_{i}.jpg")

        output = tmp_path / "output"
        prepare_dataset([source], output, seed=42)

        manifest = json.loads((output / "manifest.json").read_text())
        class_splits: dict[str, set[str]] = {cls: set() for cls in REQUIRED_CLASSES}
        for rec in manifest["records"]:
            class_splits[rec["class_name"]].add(rec["split"])

        for cls in REQUIRED_CLASSES:
            assert len(class_splits[cls]) == 3, (
                f"Class {cls} only appears in splits: {class_splits[cls]}"
            )

    def test_stratified_proportions_per_class(self, tmp_path: Path):
        """Each class should be split approximately 80/10/10 when no grouping metadata exists."""
        source = tmp_path / "source"
        for cls in REQUIRED_CLASSES:
            (source / cls).mkdir(parents=True)
            for i in range(100):
                img = Image.new("RGB", (32, 32), _unique_color(cls, i))
                img.save(source / cls / f"{cls}_{i}.jpg")

        output = tmp_path / "output"
        prepare_dataset([source], output, seed=42)

        manifest = json.loads((output / "manifest.json").read_text())
        class_splits: dict[str, dict[str, int]] = {cls: {"train": 0, "val": 0, "test": 0} for cls in REQUIRED_CLASSES}
        for rec in manifest["records"]:
            class_splits[rec["class_name"]][rec["split"]] += 1

        for cls in REQUIRED_CLASSES:
            counts = class_splits[cls]
            total = sum(counts.values())
            train_frac = counts["train"] / total
            val_frac = counts["val"] / total
            test_frac = counts["test"] / total
            assert 0.6 <= train_frac <= 0.95, f"Class {cls} train fraction {train_frac} out of expected range"
            assert 0.0 <= val_frac <= 0.3, f"Class {cls} val fraction {val_frac} out of expected range"
            assert 0.0 <= test_frac <= 0.3, f"Class {cls} test fraction {test_frac} out of expected range"


# ---------------------------------------------------------------------------
# prepare_dataset.py --classes (interim, non-production subset runs)
# ---------------------------------------------------------------------------

class TestPrepareDatasetClassSubset:
    """Opt-in --classes override for interim runs while some classes are blocked."""

    def _make_source(self, tmp_path: Path, classes: list[str], n: int = 5) -> Path:
        source = tmp_path / "source"
        for cls in classes:
            (source / cls).mkdir(parents=True)
            for i in range(n):
                img = Image.new("RGB", (32, 32), _unique_color(cls, i))
                img.save(source / cls / f"{cls}_{i}.jpg")
        return source

    def test_default_behavior_unchanged(self, tmp_path: Path):
        """Omitting --classes still requires all six and marks the set as production."""
        source = self._make_source(tmp_path, REQUIRED_CLASSES)
        output = tmp_path / "output"
        prepare_dataset([source], output, seed=42)

        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["classes"] == REQUIRED_CLASSES
        assert manifest["is_production_class_set"] is True
        assert manifest["production_classes"] == REQUIRED_CLASSES

    def test_subset_produces_interim_manifest(self, tmp_path: Path):
        """A class subset only requires/includes the requested classes and is marked non-production."""
        subset = ["Clean", "Dusty", "Hotspot"]
        source = self._make_source(tmp_path, subset)
        output = tmp_path / "output"
        prepare_dataset([source], output, seed=42, classes=subset)

        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["classes"] == subset
        assert manifest["is_production_class_set"] is False
        assert manifest["production_classes"] == REQUIRED_CLASSES
        assert {rec["class_name"] for rec in manifest["records"]} == set(subset)

        for split in SPLITS:
            for cls in subset:
                assert (output / split / cls).is_dir()
            # Classes outside the requested subset must not get output directories.
            assert not (output / split / "Bird-Drop").exists()

    def test_subset_still_requires_all_requested_classes(self, tmp_path: Path):
        """A requested class with zero images still fails closed, even in subset mode."""
        source = self._make_source(tmp_path, ["Clean", "Dusty"])
        output = tmp_path / "output"
        with pytest.raises(RuntimeError, match="missing required classes"):
            prepare_dataset([source], output, seed=42, classes=["Clean", "Dusty", "Hotspot"])

    def test_subset_rejects_class_outside_production_set(self, tmp_path: Path):
        """A class not in the production six must be rejected even when passed explicitly."""
        source = self._make_source(tmp_path, ["Clean", "Dusty"])
        output = tmp_path / "output"
        with pytest.raises(RuntimeError, match="outside the production set"):
            prepare_dataset([source], output, seed=42, classes=["Clean", "Dusty", "Snow-Covered"])

    def test_subset_rejects_empty_classes(self, tmp_path: Path):
        source = self._make_source(tmp_path, ["Clean"])
        output = tmp_path / "output"
        with pytest.raises(RuntimeError, match="must not be empty"):
            prepare_dataset([source], output, seed=42, classes=[])

    def test_subset_rejects_duplicate_classes(self, tmp_path: Path):
        """Duplicate class names would corrupt manifest/output-layer sizing downstream."""
        source = self._make_source(tmp_path, ["Clean", "Dusty"])
        output = tmp_path / "output"
        with pytest.raises(RuntimeError, match="duplicate entries"):
            prepare_dataset([source], output, seed=42, classes=["Clean", "Clean", "Dusty"])

    def test_subset_ignores_extra_class_folders_as_unknown(self, tmp_path: Path):
        """A source containing classes outside the requested subset must still fail closed."""
        source = self._make_source(tmp_path, ["Clean", "Dusty", "Hotspot"])
        output = tmp_path / "output"
        with pytest.raises(RuntimeError, match="unknown classes found in source"):
            prepare_dataset([source], output, seed=42, classes=["Clean", "Dusty"])

    def test_duplicate_detected_across_multiple_source_roots(self, tmp_path: Path):
        """Regression: duplicate detection must span ALL --source roots, not just
        each one individually. A fresh seen_hashes dict per source root would let a
        byte-identical image in two different roots slip into different splits."""
        source_a = tmp_path / "source_a"
        source_b = tmp_path / "source_b"
        (source_a / "Clean").mkdir(parents=True)
        (source_b / "Dusty").mkdir(parents=True)

        img = Image.new("RGB", (32, 32), _unique_color("shared", 0))
        img.save(source_a / "Clean" / "clean_1.jpg")
        img.save(source_b / "Dusty" / "dirty_1.jpg")  # byte-identical, different root+class

        output = tmp_path / "output"
        with pytest.raises(RuntimeError, match="duplicate image detected"):
            prepare_dataset([source_a, source_b], output, seed=42)
