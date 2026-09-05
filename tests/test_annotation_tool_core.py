"""Tests for training/detection/annotation_tool/core.py.

Covers the save/load/validation/progress logic behind the local
annotation UI, using synthetic tmp_path fixtures - no real campaign
data needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from training.detection.annotation_tool.core import (
    CLASS_ID,
    compute_progress,
    label_path_for,
    list_images,
    load_boxes,
    save_boxes,
    validate_box,
)


def _make_images(images_dir: Path, names: list[str]) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        Image.new("RGB", (64, 64), (100, 100, 100)).save(images_dir / name)


# ---------------------------------------------------------------------------
# A. list_images / label_path_for
# ---------------------------------------------------------------------------

class TestListImagesAndPaths:
    def test_lists_supported_extensions_only(self, tmp_path):
        _make_images(tmp_path, ["a.jpg", "b.png"])
        (tmp_path / "notes.txt").write_text("x")
        assert list_images(tmp_path) == ["a.jpg", "b.png"]

    def test_sorted_deterministic_order(self, tmp_path):
        _make_images(tmp_path, ["c.jpg", "a.jpg", "b.jpg"])
        assert list_images(tmp_path) == ["a.jpg", "b.jpg", "c.jpg"]

    def test_label_path_strips_directory_components(self, tmp_path):
        labels_dir = tmp_path / "labels"
        result = label_path_for(labels_dir, "../../etc/passwd.jpg")
        assert result == labels_dir / "passwd.txt"
        assert ".." not in str(result)

    def test_label_path_matches_image_stem(self, tmp_path):
        labels_dir = tmp_path / "labels"
        assert label_path_for(labels_dir, "IMG_0001.jpg") == labels_dir / "IMG_0001.txt"


# ---------------------------------------------------------------------------
# B. validate_box
# ---------------------------------------------------------------------------

class TestValidateBox:
    def test_valid_box_passes(self):
        validate_box({"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.3})

    def test_missing_field_rejected(self):
        with pytest.raises(ValueError, match="missing required field"):
            validate_box({"cx": 0.5, "cy": 0.5, "w": 0.2})

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            validate_box({"cx": float("nan"), "cy": 0.5, "w": 0.2, "h": 0.2})

    def test_zero_width_rejected(self):
        with pytest.raises(ValueError, match="zero/negative-area"):
            validate_box({"cx": 0.5, "cy": 0.5, "w": 0.0, "h": 0.2})

    def test_negative_height_rejected(self):
        with pytest.raises(ValueError, match="zero/negative-area"):
            validate_box({"cx": 0.5, "cy": 0.5, "w": 0.2, "h": -0.1})

    def test_center_out_of_bounds_rejected(self):
        with pytest.raises(ValueError, match="out of \\[0,1\\]"):
            validate_box({"cx": 1.5, "cy": 0.5, "w": 0.2, "h": 0.2})

    def test_box_edge_extending_outside_image_rejected(self):
        # Center and dimensions each individually within [0,1], but the
        # left edge (cx - w/2 = 0.05 - 0.5 = -0.45) falls outside the
        # image - this is the gap the original QC validator (Phase 8
        # domain-adaptation task) didn't check for at the box-edit layer.
        with pytest.raises(ValueError, match="outside the image bounds"):
            validate_box({"cx": 0.05, "cy": 0.5, "w": 0.5, "h": 0.2})

    def test_box_touching_edge_exactly_is_allowed(self):
        # cx=0.1, w=0.2 -> left edge exactly 0.0 - a real, legitimate
        # panel clipped exactly at the frame edge must not be rejected.
        validate_box({"cx": 0.1, "cy": 0.5, "w": 0.2, "h": 0.2})


# ---------------------------------------------------------------------------
# C. save_boxes / load_boxes round trip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundTrip:
    def test_never_visited_image_returns_none(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        assert load_boxes(labels_dir, "img.jpg") is None

    def test_saved_empty_list_is_distinguishable_from_never_visited(self, tmp_path):
        labels_dir = tmp_path / "labels"
        save_boxes(labels_dir, "img.jpg", [])
        result = load_boxes(labels_dir, "img.jpg")
        assert result == []  # not None - "visited, confirmed zero panels"

    def test_round_trip_preserves_box_values(self, tmp_path):
        labels_dir = tmp_path / "labels"
        boxes = [
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.3},
            {"class_id": 0, "cx": 0.1, "cy": 0.9, "w": 0.15, "h": 0.1},
        ]
        save_boxes(labels_dir, "img.jpg", boxes)
        loaded = load_boxes(labels_dir, "img.jpg")
        assert len(loaded) == 2
        assert loaded[0]["cx"] == pytest.approx(0.5)
        assert loaded[1]["h"] == pytest.approx(0.1)

    def test_all_saved_boxes_use_class_id_zero(self, tmp_path):
        labels_dir = tmp_path / "labels"
        save_boxes(labels_dir, "img.jpg", [{"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}])
        loaded = load_boxes(labels_dir, "img.jpg")
        assert loaded[0]["class_id"] == CLASS_ID == 0

    def test_invalid_box_raises_and_writes_nothing(self, tmp_path):
        labels_dir = tmp_path / "labels"
        with pytest.raises(ValueError):
            save_boxes(labels_dir, "img.jpg", [{"cx": 0.5, "cy": 0.5, "w": -1, "h": 0.2}])
        assert not (labels_dir / "img.txt").exists()

    def test_resave_overwrites_cleanly_no_leftover_tmp_file(self, tmp_path):
        labels_dir = tmp_path / "labels"
        save_boxes(labels_dir, "img.jpg", [{"cx": 0.2, "cy": 0.2, "w": 0.1, "h": 0.1}])
        save_boxes(labels_dir, "img.jpg", [{"cx": 0.8, "cy": 0.8, "w": 0.1, "h": 0.1}])
        loaded = load_boxes(labels_dir, "img.jpg")
        assert len(loaded) == 1
        assert loaded[0]["cx"] == pytest.approx(0.8)
        assert not (labels_dir / "img.jpg.tmp").exists()

    def test_partial_write_never_corrupts_existing_label(self, tmp_path, monkeypatch):
        """Simulates a crash between writing the temp file and the atomic
        rename - the previous good label file must remain intact."""
        labels_dir = tmp_path / "labels"
        save_boxes(labels_dir, "img.jpg", [{"cx": 0.3, "cy": 0.3, "w": 0.1, "h": 0.1}])
        original_content = (labels_dir / "img.txt").read_text()

        import training.detection.annotation_tool.core as core_mod

        def boom(*args, **kwargs):
            raise OSError("simulated crash before rename")

        monkeypatch.setattr(core_mod.os, "replace", boom)
        with pytest.raises(OSError):
            save_boxes(labels_dir, "img.jpg", [{"cx": 0.9, "cy": 0.9, "w": 0.1, "h": 0.1}])

        # The original file is untouched - only the .tmp file was affected.
        assert (labels_dir / "img.txt").read_text() == original_content

    def test_writes_file_directly_in_labels_dir_never_elsewhere(self, tmp_path):
        labels_dir = tmp_path / "labels"
        path = save_boxes(labels_dir, "sub/../img.jpg", [{"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}])
        assert path.parent == labels_dir


# ---------------------------------------------------------------------------
# D. compute_progress
# ---------------------------------------------------------------------------

class TestComputeProgress:
    def test_zero_annotated_initially(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        _make_images(images_dir, ["a.jpg", "b.jpg", "c.jpg"])
        progress = compute_progress(images_dir, labels_dir)
        assert progress["total"] == 3
        assert progress["annotated"] == 0
        assert progress["remaining"] == 3

    def test_counts_update_after_save(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        _make_images(images_dir, ["a.jpg", "b.jpg"])
        save_boxes(labels_dir, "a.jpg", [{"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}])
        progress = compute_progress(images_dir, labels_dir)
        assert progress["annotated"] == 1
        assert progress["remaining"] == 1
        assert progress["total_boxes_so_far"] == 1

    def test_zero_panel_confirmation_counts_as_annotated_not_pending(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        _make_images(images_dir, ["a.jpg"])
        save_boxes(labels_dir, "a.jpg", [])
        progress = compute_progress(images_dir, labels_dir)
        assert progress["annotated"] == 1
        assert progress["images"][0]["box_count"] == 0

    def test_per_image_entries_match_filenames(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        _make_images(images_dir, ["x.jpg", "y.jpg"])
        progress = compute_progress(images_dir, labels_dir)
        filenames = {im["filename"] for im in progress["images"]}
        assert filenames == {"x.jpg", "y.jpg"}
