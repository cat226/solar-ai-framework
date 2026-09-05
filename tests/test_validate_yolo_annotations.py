"""Tests for training/detection/validate_yolo_annotations.py.

Synthetic fixtures only - this script's job is pure syntax/geometry
validation of already-written label files, so no real annotation
campaign data is needed to exercise every rule.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from training.detection.validate_yolo_annotations import (
    compute_statistics,
    render_contact_sheets,
    validate_directory,
    validate_label_file,
    verify_split_lock,
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

    def test_box_edge_outside_image_is_error_even_with_in_range_center_and_size(self, tmp_path):
        # cx=0.05, w=0.5 -> left edge = -0.2, outside the image, even
        # though cx and w are each individually within [0,1].
        p = tmp_path / "a.txt"
        p.write_text("0 0.05 0.5 0.5 0.2\n")
        result = validate_label_file(p)
        assert any("outside the image" in e for e in result["errors"])
        assert result["boxes"] == []

    def test_box_touching_edge_exactly_is_not_an_error(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 0.1 0.5 0.2 0.2\n")  # left edge exactly 0.0
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


# ---------------------------------------------------------------------------
# E. verify_split_lock
# ---------------------------------------------------------------------------

def _make_locked_campaign(tmp_path, split_counts: dict[str, int], class_by_index=None):
    """Builds a tiny synthetic images_dir + selected_N.csv matching the
    real campaign's CSV schema (filename, sha256, source_split,
    source_class - the columns verify_split_lock/compute_statistics
    actually read)."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    csv_path = tmp_path / "selected.csv"
    rows = []
    i = 0
    for split, count in split_counts.items():
        for _ in range(count):
            filename = f"img_{i}.jpg"
            Image.new("RGB", (32, 32), (i % 255, 0, 0)).save(images_dir / filename)
            sha = hashlib.sha256((images_dir / filename).read_bytes()).hexdigest()
            rows.append({
                "filename": filename, "sha256": sha, "source_split": split,
                "source_class": (class_by_index(i) if class_by_index else "Clean"),
            })
            i += 1
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "sha256", "source_split", "source_class"])
        writer.writeheader()
        writer.writerows(rows)
    return images_dir, csv_path


class TestVerifySplitLock:
    def test_matching_split_counts_locks_correctly(self, tmp_path):
        images_dir, csv_path = _make_locked_campaign(tmp_path, {"train": 152, "val": 24, "test": 24})
        result = verify_split_lock(images_dir, csv_path)
        assert result["locked_correctly"] is True
        assert result["split_counts_match"] is True
        assert result["missing_from_images_dir"] == []
        assert result["extra_in_images_dir"] == []
        assert result["sha256_mismatches"] == []

    def test_drifted_split_counts_fail_lock(self, tmp_path):
        images_dir, csv_path = _make_locked_campaign(tmp_path, {"train": 150, "val": 25, "test": 25})
        result = verify_split_lock(images_dir, csv_path)
        assert result["locked_correctly"] is False
        assert result["split_counts_match"] is False

    def test_missing_image_detected(self, tmp_path):
        images_dir, csv_path = _make_locked_campaign(tmp_path, {"train": 3, "val": 1, "test": 1})
        (images_dir / "img_0.jpg").unlink()
        result = verify_split_lock(images_dir, csv_path)
        assert result["locked_correctly"] is False
        assert "img_0.jpg" in result["missing_from_images_dir"]

    def test_extra_unlocked_image_detected(self, tmp_path):
        images_dir, csv_path = _make_locked_campaign(tmp_path, {"train": 3, "val": 1, "test": 1})
        Image.new("RGB", (32, 32), (9, 9, 9)).save(images_dir / "sneaky_extra.jpg")
        result = verify_split_lock(images_dir, csv_path)
        assert result["locked_correctly"] is False
        assert "sneaky_extra.jpg" in result["extra_in_images_dir"]

    def test_content_tampering_detected_via_sha256(self, tmp_path):
        images_dir, csv_path = _make_locked_campaign(tmp_path, {"train": 3, "val": 1, "test": 1})
        # Overwrite one image's content without updating the CSV's
        # recorded hash - simulates an image silently replaced/edited.
        Image.new("RGB", (32, 32), (255, 255, 255)).save(images_dir / "img_0.jpg")
        result = verify_split_lock(images_dir, csv_path)
        assert result["locked_correctly"] is False
        assert any(m["filename"] == "img_0.jpg" for m in result["sha256_mismatches"])

    def test_untampered_images_pass_even_when_one_is_swapped(self, tmp_path):
        images_dir, csv_path = _make_locked_campaign(tmp_path, {"train": 3, "val": 1, "test": 1})
        Image.new("RGB", (32, 32), (255, 255, 255)).save(images_dir / "img_1.jpg")
        result = verify_split_lock(images_dir, csv_path)
        mismatched = {m["filename"] for m in result["sha256_mismatches"]}
        assert "img_1.jpg" in mismatched
        assert "img_2.jpg" not in mismatched


# ---------------------------------------------------------------------------
# F. compute_statistics
# ---------------------------------------------------------------------------

class TestComputeStatistics:
    def test_no_labels_yet_reports_zero(self, tmp_path):
        images_dir, csv_path = _make_locked_campaign(tmp_path, {"train": 3, "val": 1, "test": 1})
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        stats = compute_statistics(images_dir, labels_dir, csv_path)
        assert stats["images_with_a_label_file"] == 0
        assert stats["total_boxes"] == 0
        assert stats["total_images_in_campaign"] == 5

    def test_partial_annotation_counted_correctly(self, tmp_path):
        images_dir, csv_path = _make_locked_campaign(tmp_path, {"train": 3, "val": 1, "test": 1})
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "img_0.txt").write_text("0 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.1 0.1\n")
        (labels_dir / "img_1.txt").write_text("0 0.5 0.5 0.3 0.3\n")

        stats = compute_statistics(images_dir, labels_dir, csv_path)
        assert stats["images_with_a_label_file"] == 2
        assert stats["total_boxes"] == 3
        assert stats["min_boxes_per_image"] == 1
        assert stats["max_boxes_per_image"] == 2

    def test_per_split_box_counts_attributed_correctly(self, tmp_path):
        images_dir, csv_path = _make_locked_campaign(tmp_path, {"train": 2, "val": 2, "test": 2})
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        # img_0, img_1 = train; img_2, img_3 = val; img_4, img_5 = test
        (labels_dir / "img_0.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (labels_dir / "img_2.txt").write_text("0 0.5 0.5 0.2 0.2\n0 0.3 0.3 0.1 0.1\n")

        stats = compute_statistics(images_dir, labels_dir, csv_path)
        assert stats["per_split_box_counts"]["train"] == 1
        assert stats["per_split_box_counts"]["val"] == 2
        assert stats["per_split_box_counts"]["test"] == 0

    def test_per_source_class_box_counts(self, tmp_path):
        images_dir, csv_path = _make_locked_campaign(
            tmp_path, {"train": 4}, class_by_index=lambda i: "Clean" if i % 2 == 0 else "Dusty"
        )
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "img_0.txt").write_text("0 0.5 0.5 0.2 0.2\n")  # Clean
        (labels_dir / "img_1.txt").write_text("0 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.1 0.1\n")  # Dusty

        stats = compute_statistics(images_dir, labels_dir, csv_path)
        assert stats["per_source_class_box_counts"]["Clean"] == 1
        assert stats["per_source_class_box_counts"]["Dusty"] == 2

    def test_zero_box_images_counted_as_confirmed_negatives(self, tmp_path):
        images_dir, csv_path = _make_locked_campaign(tmp_path, {"train": 2})
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "img_0.txt").write_text("")  # confirmed zero-panel
        (labels_dir / "img_1.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        stats = compute_statistics(images_dir, labels_dir, csv_path)
        assert stats["zero_box_images"] == 1


# ---------------------------------------------------------------------------
# G. render_contact_sheets
# ---------------------------------------------------------------------------

class TestRenderContactSheets:
    def test_writes_expected_number_of_sheets(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        output_dir = tmp_path / "sheets"
        images_dir.mkdir()
        labels_dir.mkdir()
        for i in range(7):
            Image.new("RGB", (64, 64), (10, 10, 10)).save(images_dir / f"img_{i}.jpg")
        (labels_dir / "img_0.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        sheets = render_contact_sheets(images_dir, labels_dir, output_dir, images_per_sheet=5)
        assert len(sheets) == 2  # 7 images, 5 per sheet -> 2 sheets
        assert all(p.is_file() for p in sheets)

    def test_sheets_are_valid_images(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        output_dir = tmp_path / "sheets"
        images_dir.mkdir()
        labels_dir.mkdir()
        for i in range(3):
            Image.new("RGB", (64, 64), (10, 10, 10)).save(images_dir / f"img_{i}.jpg")

        sheets = render_contact_sheets(images_dir, labels_dir, output_dir)
        with Image.open(sheets[0]) as im:
            im.verify()

    def test_empty_directory_produces_no_sheets(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        output_dir = tmp_path / "sheets"
        images_dir.mkdir()
        labels_dir.mkdir()
        sheets = render_contact_sheets(images_dir, labels_dir, output_dir)
        assert sheets == []
