"""Tests for training/detection/select_annotation_sample.py.

Synthetic fixtures only. This script picks images for a *human* to
annotate next - it must never produce a bounding box - so these tests
verify the selection mechanics (inventory, near-duplicate handling,
split-aware budget allocation, determinism) rather than anything about
detection correctness.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.detection.select_annotation_sample import (
    SOURCE_CLASSES,
    SPLITS,
    _allocate_budget,
    _allocate_split_budgets,
    _compute_visual_features,
    _dedupe_near_duplicates,
    _stratify,
    _tercile_bin,
    inventory,
    select_sample,
)


def _make_varied_image(seed: int, size: int = 64) -> Image.Image:
    """A solid-color fill has zero internal gradient, which makes dHash
    (a gradient-based hash) collapse EVERY solid-color image to the same
    hash regardless of color - degenerate for testing near-duplicate
    logic. Build a genuinely varied pattern instead (a per-pixel
    pseudo-random field seeded deterministically) so distinct `seed`
    values reliably produce distinct dHashes, and only a true
    byte-for-byte copy of the same seed collides."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_source_tree(root: Path, per_split_class_count: dict[tuple[str, str], int]) -> Path:
    """Build a tiny synthetic SolNET-shaped prepared/ tree.
    per_split_class_count: {(split, class): count}. Each image gets a
    distinct pseudo-random pattern (see _make_varied_image) so dHash
    naturally differs unless deliberately duplicated by the caller
    afterward."""
    prepared = root / "prepared"
    counter = 0
    for (split, cls), count in per_split_class_count.items():
        d = prepared / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            counter += 1
            _make_varied_image(counter).save(d / f"img_{split}_{cls}_{i}.jpg", format="JPEG", quality=95)
    return prepared


# ---------------------------------------------------------------------------
# A. inventory()
# ---------------------------------------------------------------------------

class TestInventory:
    def test_finds_only_clean_and_dusty_never_hotspot(self, tmp_path):
        prepared = _make_source_tree(tmp_path, {
            ("train", "Clean"): 2, ("train", "Dusty"): 2, ("train", "Hotspot"): 5,
        })
        records = inventory(prepared, with_features=False)
        assert len(records) == 4
        assert all(r["source_class"] in SOURCE_CLASSES for r in records)
        assert "Hotspot" not in {r["source_class"] for r in records}

    def test_records_real_sha256_and_dimensions(self, tmp_path):
        prepared = _make_source_tree(tmp_path, {("train", "Clean"): 1})
        records = inventory(prepared, with_features=False)
        assert len(records) == 1
        r = records[0]
        assert r["readable"] is True
        assert len(r["sha256"]) == 64
        assert r["width"] == 64 and r["height"] == 64
        assert r["format"] == "JPEG"
        assert r["license"] == "CC-BY-4.0"
        assert r["source_dataset"] == "SolNET"

    def test_unreadable_file_is_flagged_not_raised(self, tmp_path):
        prepared = tmp_path / "prepared" / "train" / "Clean"
        prepared.mkdir(parents=True)
        (prepared / "corrupt.jpg").write_bytes(b"not an image")
        records = inventory(prepared.parent.parent, with_features=False)
        assert len(records) == 1
        assert records[0]["readable"] is False
        assert "read_error" in records[0]

    def test_with_features_true_populates_visual_stats(self, tmp_path):
        prepared = _make_source_tree(tmp_path, {("train", "Clean"): 1})
        records = inventory(prepared, with_features=True)
        r = records[0]
        for key in ("dhash", "mean_brightness", "std_brightness", "mean_saturation", "edge_density", "aspect_ratio"):
            assert key in r

    def test_with_features_false_skips_visual_stats(self, tmp_path):
        prepared = _make_source_tree(tmp_path, {("train", "Clean"): 1})
        records = inventory(prepared, with_features=False)
        assert "dhash" not in records[0]

    def test_never_modifies_original_files(self, tmp_path):
        prepared = _make_source_tree(tmp_path, {("train", "Clean"): 1})
        img_path = next((prepared / "train" / "Clean").iterdir())
        before = img_path.read_bytes()
        inventory(prepared, with_features=True)
        after = img_path.read_bytes()
        assert before == after


# ---------------------------------------------------------------------------
# B. Near-duplicate dedup - per-split, not global
# ---------------------------------------------------------------------------

class TestDedupePerSplit:
    def test_duplicate_within_a_split_is_collapsed(self, tmp_path):
        prepared = _make_source_tree(tmp_path, {("train", "Clean"): 1})
        # Duplicate the one image under a new name in the same split/class.
        src = next((prepared / "train" / "Clean").iterdir())
        dup = src.parent / "duplicate.jpg"
        dup.write_bytes(src.read_bytes())

        records = inventory(prepared, with_features=True)
        deduped = _dedupe_near_duplicates(records)
        assert len(deduped) == 1

    def test_identical_image_in_different_splits_is_NOT_cross_split_collapsed(self, tmp_path):
        """The core fix under test: a near-duplicate must not be silently
        dropped from val/test just because an identical-looking image was
        already kept from train - each split dedups independently."""
        prepared = _make_source_tree(tmp_path, {})
        train_dir = prepared / "train" / "Clean"
        test_dir = prepared / "test" / "Clean"
        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)
        img = Image.new("RGB", (64, 64), (10, 20, 30))
        img.save(train_dir / "a.jpg", format="JPEG")
        img.save(test_dir / "a.jpg", format="JPEG")  # visually identical, different split

        records = inventory(prepared, with_features=True)
        deduped = _dedupe_near_duplicates(records)
        # Both survive - one per split - because dedup is scoped per-split.
        assert len(deduped) == 2
        assert {r["source_split"] for r in deduped} == {"train", "test"}

    def test_distinct_images_all_survive(self, tmp_path):
        prepared = _make_source_tree(tmp_path, {("train", "Clean"): 5})
        records = inventory(prepared, with_features=True)
        deduped = _dedupe_near_duplicates(records)
        assert len(deduped) == 5


# ---------------------------------------------------------------------------
# C. Stratification and budget allocation
# ---------------------------------------------------------------------------

class TestTercileBin:
    def test_low_med_high_boundaries(self):
        edges = (10.0, 20.0)
        assert _tercile_bin(5.0, edges) == "low"
        assert _tercile_bin(10.0, edges) == "low"
        assert _tercile_bin(15.0, edges) == "med"
        assert _tercile_bin(20.0, edges) == "med"
        assert _tercile_bin(25.0, edges) == "high"


class TestAllocateBudget:
    def test_exact_proportional_split_when_divisible(self):
        strata = {"a": list(range(50)), "b": list(range(50))}
        result = _allocate_budget(strata, 100)
        assert result == {"a": 50, "b": 50}

    def test_total_allocated_never_exceeds_pool_size(self):
        strata = {"a": list(range(3)), "b": list(range(3))}
        result = _allocate_budget(strata, 100)
        assert result["a"] <= 3
        assert result["b"] <= 3

    def test_allocation_sums_to_budget_when_pool_is_large_enough(self):
        strata = {"a": list(range(30)), "b": list(range(70))}
        result = _allocate_budget(strata, 20)
        assert sum(result.values()) == 20

    def test_every_nonempty_stratum_gets_at_least_one_when_budget_allows(self):
        strata = {f"s{i}": list(range(10)) for i in range(15)}
        result = _allocate_budget(strata, 50)
        assert all(v >= 1 for v in result.values())

    def test_more_strata_than_budget_gives_largest_strata_priority(self):
        strata = {"big": list(range(100)), "small": list(range(2))}
        result = _allocate_budget(strata, 1)
        assert result["big"] == 1
        assert result["small"] == 0

    def test_deterministic_across_repeated_calls(self):
        strata = {"a": list(range(17)), "b": list(range(23)), "c": list(range(11))}
        r1 = _allocate_budget(strata, 30)
        r2 = _allocate_budget(strata, 30)
        assert r1 == r2


class TestAllocateSplitBudgets:
    def test_proportional_to_split_size(self, tmp_path):
        prepared = _make_source_tree(tmp_path, {
            ("train", "Clean"): 80, ("val", "Clean"): 10, ("test", "Clean"): 10,
        })
        records = inventory(prepared, with_features=True)
        deduped = _dedupe_near_duplicates(records)
        budgets = _allocate_split_budgets(deduped, 20)
        assert sum(budgets.values()) == 20
        # 80/100=80% -> ~16, 10% each -> ~2 each
        assert budgets["train"] >= budgets["val"]
        assert budgets["train"] >= budgets["test"]
        assert budgets.get("val", 0) >= 1
        assert budgets.get("test", 0) >= 1


# ---------------------------------------------------------------------------
# D. select_sample() - end to end
# ---------------------------------------------------------------------------

class TestSelectSampleEndToEnd:
    def test_never_exceeds_requested_budget(self, tmp_path):
        prepared = _make_source_tree(tmp_path, {
            ("train", "Clean"): 40, ("train", "Dusty"): 30,
            ("val", "Clean"): 10, ("val", "Dusty"): 10,
            ("test", "Clean"): 10, ("test", "Dusty"): 10,
        })
        full_inv, selected, report = select_sample(prepared, total_budget=50)
        assert len(selected) <= 50
        assert report["selected_count"] == len(selected)

    def test_all_three_splits_represented_when_all_exist(self, tmp_path):
        prepared = _make_source_tree(tmp_path, {
            ("train", "Clean"): 40, ("train", "Dusty"): 40,
            ("val", "Clean"): 10, ("val", "Dusty"): 10,
            ("test", "Clean"): 10, ("test", "Dusty"): 10,
        })
        _, selected, report = select_sample(prepared, total_budget=60)
        splits_present = {r["source_split"] for r in selected}
        assert splits_present == {"train", "val", "test"}
        assert report["selected_by_source_split"]["val"] > 0
        assert report["selected_by_source_split"]["test"] > 0

    def test_both_classes_represented(self, tmp_path):
        prepared = _make_source_tree(tmp_path, {
            ("train", "Clean"): 30, ("train", "Dusty"): 30,
        })
        _, selected, report = select_sample(prepared, total_budget=20)
        assert report["selected_by_source_class"]["Clean"] > 0
        assert report["selected_by_source_class"]["Dusty"] > 0

    def test_every_selected_record_has_a_selection_reason(self, tmp_path):
        prepared = _make_source_tree(tmp_path, {("train", "Clean"): 20})
        _, selected, _ = select_sample(prepared, total_budget=5)
        assert all(r.get("selection_reason") for r in selected)
        assert all(r.get("stratum") for r in selected)

    def test_deterministic_across_repeated_runs(self, tmp_path):
        prepared = _make_source_tree(tmp_path, {
            ("train", "Clean"): 25, ("train", "Dusty"): 25,
        })
        _, sel1, _ = select_sample(prepared, total_budget=10)
        _, sel2, _ = select_sample(prepared, total_budget=10)
        assert sorted(r["sha256"] for r in sel1) == sorted(r["sha256"] for r in sel2)

    def test_selected_images_are_a_subset_of_the_full_inventory(self, tmp_path):
        prepared = _make_source_tree(tmp_path, {("train", "Clean"): 20})
        full_inv, selected, _ = select_sample(prepared, total_budget=5)
        full_hashes = {r["sha256"] for r in full_inv if r["readable"]}
        selected_hashes = {r["sha256"] for r in selected}
        assert selected_hashes <= full_hashes

    def test_budget_larger_than_pool_selects_the_whole_deduplicated_pool(self, tmp_path):
        prepared = _make_source_tree(tmp_path, {("train", "Clean"): 5})
        _, selected, report = select_sample(prepared, total_budget=1000)
        assert len(selected) == report["deduplicated_pool"]
