"""Tests for training/detection/validate_yolo_annotations.py.

Synthetic fixtures only - this script's job is pure syntax/geometry
validation of already-written label files, so no real annotation
campaign data is needed to exercise every rule.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from training.detection.validate_yolo_annotations import (
    validate_directory,
    validate_label_file,
)


def _make_image(path: Path, size=(64, 64)) -> None:
    Image.new("RGB", size, (100, 100, 100)).save(path)


# ---------------------------------------------------------------------------
# A. validate_label_file - single-file mechanics
# ---------------------------------------------------------------------------

class TestValidateLabelFile:
    def test_empty_file_is_valid_and_marked_empty(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("")
        result = validate_label_file(p)
        assert result["is_empty"] is True
        assert result["errors"] == []
        assert result["boxes"] == []

    def test_valid_single_box(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 0.5 0.5 0.2 0.3\n")
        result = validate_label_file(p)
        assert result["errors"] == []
        assert len(result["boxes"]) == 1

    def test_valid_multiple_boxes(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 0.2 0.2 0.1 0.1\n0 0.8 0.8 0.1 0.1\n")
        result = validate_label_file(p)
        assert result["errors"] == []
        assert len(result["boxes"]) == 2

    def test_wrong_field_count_is_error(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 0.5 0.5 0.2\n")
        result = validate_label_file(p)
        assert any("expected 5 fields" in e for e in result["errors"])
        assert result["boxes"] == []

    def test_non_numeric_field_is_error(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 x 0.5 0.2 0.2\n")
        result = validate_label_file(p)
        assert any("could not parse" in e for e in result["errors"])

    def test_wrong_class_id_is_error(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("1 0.5 0.5 0.2 0.2\n")
        result = validate_label_file(p)
        assert any("class id 1" in e for e in result["errors"])

    def test_custom_allowed_class_id(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("3 0.5 0.5 0.2 0.2\n")
        result = validate_label_file(p, allowed_class_id=3)
        assert result["errors"] == []
        assert len(result["boxes"]) == 1

    def test_coordinate_above_one_is_error(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 1.5 0.5 0.2 0.2\n")
        result = validate_label_file(p)
        assert any("outside [0,1]" in e for e in result["errors"])

    def test_negative_coordinate_is_error(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 -0.1 0.5 0.2 0.2\n")
        result = validate_label_file(p)
        assert any("outside [0,1]" in e for e in result["errors"])

    def test_zero_width_is_error(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 0.5 0.5 0.0 0.2\n")
        result = validate_label_file(p)
        assert any("zero/negative-area" in e for e in result["errors"])

    def test_zero_height_is_error(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 0.5 0.5 0.2 0.0\n")
        result = validate_label_file(p)
        assert any("zero/negative-area" in e for e in result["errors"])

    def test_exact_duplicate_boxes_flagged(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n")
        result = validate_label_file(p)
        assert any("duplicate boxes" in e for e in result["errors"])
        # Both individually-valid boxes are still recorded even though flagged as duplicates.
        assert len(result["boxes"]) == 2

    def test_near_but_not_exact_boxes_not_flagged_as_duplicate(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 0.500 0.5 0.2 0.2\n0 0.510 0.5 0.2 0.2\n")
        result = validate_label_file(p)
        assert not any("duplicate" in e for e in result["errors"])

    def test_blank_lines_are_ignored(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 0.5 0.5 0.2 0.2\n\n\n")
        result = validate_label_file(p)
        assert result["errors"] == []
        assert len(result["boxes"]) == 1


# ---------------------------------------------------------------------------
# B. validate_directory - whole-campaign mechanics
# ---------------------------------------------------------------------------

class TestValidateDirectory:
    def test_fully_annotated_clean_directory_is_ready(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        for i in range(3):
            _make_image(images_dir / f"img{i}.jpg")
            (labels_dir / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        summary = validate_directory(images_dir, labels_dir)
        assert summary["total_images"] == 3
        assert summary["annotated_images"] == 3
        assert summary["missing_label_count"] == 0
        assert summary["files_with_errors_count"] == 0
        assert summary["total_boxes"] == 3
        assert summary["ready_for_next_stage"] is True

    def test_missing_label_file_detected(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        _make_image(images_dir / "img0.jpg")
        _make_image(images_dir / "img1.jpg")
        (labels_dir / "img0.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        # img1.txt intentionally missing

        summary = validate_directory(images_dir, labels_dir)
        assert summary["missing_label_count"] == 1
        assert "img1.jpg" in summary["missing_label_files"]
        assert summary["ready_for_next_stage"] is False

    def test_empty_label_file_is_not_an_error_but_is_flagged_for_review(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        _make_image(images_dir / "img0.jpg")
        (labels_dir / "img0.txt").write_text("")

        summary = validate_directory(images_dir, labels_dir)
        assert summary["files_with_errors_count"] == 0
        assert summary["images_with_zero_annotations_count"] == 1
        assert "img0.jpg" in summary["images_with_zero_annotations"]
        # Zero annotations is not itself a mechanical error, so this can still be "ready".
        assert summary["ready_for_next_stage"] is True

    def test_malformed_label_file_blocks_readiness(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        _make_image(images_dir / "img0.jpg")
        (labels_dir / "img0.txt").write_text("0 5.0 0.5 0.2 0.2\n")

        summary = validate_directory(images_dir, labels_dir)
        assert summary["files_with_errors_count"] == 1
        assert summary["ready_for_next_stage"] is False

    def test_orphaned_label_file_detected(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        _make_image(images_dir / "img0.jpg")
        (labels_dir / "img0.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (labels_dir / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        summary = validate_directory(images_dir, labels_dir)
        assert "orphan" in summary["orphaned_label_files"]
        assert summary["ready_for_next_stage"] is False

    def test_notes_txt_is_not_treated_as_an_orphaned_label(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        _make_image(images_dir / "img0.jpg")
        (labels_dir / "img0.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (labels_dir / "NOTES.txt").write_text("img0: ambiguous overlap, annotated as one box.\n")

        summary = validate_directory(images_dir, labels_dir)
        assert summary["orphaned_label_files"] == []
        assert summary["ready_for_next_stage"] is True

    def test_empty_directory_is_ready_with_zero_images(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        summary = validate_directory(images_dir, labels_dir)
        assert summary["total_images"] == 0
        assert summary["ready_for_next_stage"] is True

    def test_total_box_count_aggregated_correctly(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        _make_image(images_dir / "img0.jpg")
        _make_image(images_dir / "img1.jpg")
        (labels_dir / "img0.txt").write_text("0 0.2 0.2 0.1 0.1\n0 0.8 0.8 0.1 0.1\n")
        (labels_dir / "img1.txt").write_text("0 0.5 0.5 0.3 0.3\n")

        summary = validate_directory(images_dir, labels_dir)
        assert summary["total_boxes"] == 3
