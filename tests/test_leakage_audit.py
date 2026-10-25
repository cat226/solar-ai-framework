"""tests/test_leakage_audit.py — Unit tests for the pure classification and
clean-subset-construction logic in training/evaluation/leakage_audit.py.

The classification rule (hamming_distance==0 + matching filename base =
highly_likely_near_duplicate; hamming_distance==0 without a matching base =
probable_false_positive; anything else = uncertain) is not arbitrary - it
was derived from two manually visually-verified pairs on the real dataset
(see docs/ML_EVALUATION_v1.0.0.md): a true duplicate with a matching
capture-timestamp filename, and a hash collision without one. These tests
exercise the rule directly, deterministically, without needing real images.
"""
from __future__ import annotations

from training.evaluation.leakage_audit import _base_id, _build_clean_test_subset, _classify_pairs


class TestBaseId:
    def test_strips_11zon_resize_tool_suffix_chain(self):
        assert _base_id("20210917_151404_2_11zon_4_11zon.jpg") == _base_id("20210917_151404.jpg")

    def test_strips_parenthesized_duplicate_marker(self):
        assert _base_id("photo(0).jpg") == _base_id("photo.jpg")

    def test_different_timestamps_have_different_base_ids(self):
        assert _base_id("20210917_151404.jpg") != _base_id("20210917_151412.jpg")


class TestClassifyPairs:
    def test_exact_hash_match_with_shared_base_is_highly_likely_duplicate(self):
        pairs = [{
            "a": "train/Clean/20210917_151404.jpg",
            "b": "test/Clean/20210917_151404_2_11zon_4_11zon.jpg",
            "hamming_distance": 0,
        }]
        result = _classify_pairs(pairs)
        assert result["counts"]["highly_likely_near_duplicate"] == 1
        assert result["counts"]["probable_false_positive"] == 0
        assert result["counts"]["uncertain"] == 0

    def test_exact_hash_match_without_shared_base_is_probable_false_positive(self):
        """The real manually-verified case: val/Dusty/20210916_130941.jpg vs
        test/Clean/20210917_151412.jpg - hamming=0, different timestamps,
        visually confirmed to be two different photographs."""
        pairs = [{
            "a": "val/Dusty/20210916_130941.jpg",
            "b": "test/Clean/20210917_151412.jpg",
            "hamming_distance": 0,
        }]
        result = _classify_pairs(pairs)
        assert result["counts"]["probable_false_positive"] == 1
        assert result["counts"]["highly_likely_near_duplicate"] == 0

    def test_nonzero_hamming_distance_is_always_uncertain_regardless_of_base(self):
        pairs = [{
            "a": "train/Clean/20210917_151404.jpg",
            "b": "test/Clean/20210917_151404_2_11zon_4_11zon.jpg",
            "hamming_distance": 3,
        }]
        result = _classify_pairs(pairs)
        assert result["counts"]["uncertain"] == 1
        assert result["counts"]["highly_likely_near_duplicate"] == 0

    def test_empty_input_yields_zero_counts(self):
        result = _classify_pairs([])
        assert result["counts"] == {"highly_likely_near_duplicate": 0, "probable_false_positive": 0, "uncertain": 0}


class TestBuildCleanTestSubset:
    def test_test_image_with_highly_likely_duplicate_is_excluded(self):
        images_by_split = {"test": [_FakePath("Clean", "dup.jpg"), _FakePath("Clean", "clean.jpg")]}
        classified = {"highly_likely_near_duplicate": [
            {"a": "train/Clean/other.jpg", "b": "test/Clean/dup.jpg"},
        ]}
        result = _build_clean_test_subset(images_by_split, classified)
        assert result["original_test_count"] == 2
        assert result["clean_test_count"] == 1
        assert "test/Clean/dup.jpg" not in result["clean_test_images"]
        assert "test/Clean/clean.jpg" in result["clean_test_images"]

    def test_no_flagged_duplicates_keeps_full_test_set(self):
        images_by_split = {"test": [_FakePath("Clean", "a.jpg"), _FakePath("Dusty", "b.jpg")]}
        classified = {"highly_likely_near_duplicate": []}
        result = _build_clean_test_subset(images_by_split, classified)
        assert result["clean_test_count"] == result["original_test_count"] == 2
        assert result["contaminated_test_count"] == 0

    def test_original_test_split_files_are_never_referenced_for_deletion(self):
        """This function only returns filename lists - it must never touch
        the filesystem (no os.remove, no Path.unlink usage anywhere)."""
        import inspect
        from training.evaluation import leakage_audit
        source = inspect.getsource(leakage_audit._build_clean_test_subset)
        assert "unlink" not in source
        assert "remove" not in source
        assert "rmdir" not in source


class _FakePath:
    """Minimal stand-in for pathlib.Path with just the .parent.name / .name
    attributes _build_clean_test_subset actually uses."""
    def __init__(self, class_name: str, filename: str):
        self.name = filename
        self.parent = _FakeParent(class_name)


class _FakeParent:
    def __init__(self, name: str):
        self.name = name
