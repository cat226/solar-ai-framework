"""tests/test_evaluation_common.py — Unit tests for the pure, reusable
evaluation-metric helpers in training/evaluation/common.py.

These are ordinary deterministic functions (no model loading, no I/O beyond
what's explicitly passed in), so they're tested directly rather than through
any evaluation script.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from training.evaluation.common import (
    accuracy,
    confusion_matrix,
    default_output_root,
    dhash,
    hamming_distance,
    iou,
    load_yolo_ground_truth,
    macro_and_weighted_prf1,
    match_detections_to_ground_truth,
    per_class_prf1,
    sha256_file,
    yolo_label_to_xyxy,
)


class TestIoU:
    def test_identical_boxes_is_one(self):
        assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0

    def test_disjoint_boxes_is_zero(self):
        assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_partial_overlap(self):
        # overlap area = 5x5=25, union = 100+100-25=175
        assert iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)

    def test_zero_area_box_is_zero(self):
        assert iou([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0

    def test_touching_edges_is_zero_area_overlap(self):
        assert iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0


class TestMatchDetectionsToGroundTruth:
    def test_perfect_single_match(self):
        tp, fp, fn, ious = match_detections_to_ground_truth(
            [[0, 0, 10, 10]], [0.9], [[0, 0, 10, 10]]
        )
        assert (tp, fp, fn) == (1, 0, 0)
        assert ious == [1.0]

    def test_false_positive_no_ground_truth(self):
        tp, fp, fn, ious = match_detections_to_ground_truth(
            [[0, 0, 10, 10]], [0.9], []
        )
        assert (tp, fp, fn) == (0, 1, 0)

    def test_false_negative_missed_ground_truth(self):
        tp, fp, fn, ious = match_detections_to_ground_truth(
            [], [], [[0, 0, 10, 10]]
        )
        assert (tp, fp, fn) == (0, 0, 1)

    def test_below_threshold_is_fp_and_fn_not_a_match(self):
        # boxes overlap only slightly - IoU < 0.5
        tp, fp, fn, ious = match_detections_to_ground_truth(
            [[0, 0, 10, 10]], [0.9], [[8, 8, 18, 18]], iou_threshold=0.5
        )
        assert (tp, fp, fn) == (0, 1, 1)

    def test_higher_confidence_prediction_claims_ground_truth_first(self):
        """Two predictions both overlap the same single ground-truth box -
        only the highest-confidence one may claim it; the other is a false
        positive, never double-counted as two true positives."""
        tp, fp, fn, ious = match_detections_to_ground_truth(
            [[0, 0, 10, 10], [1, 1, 11, 11]], [0.5, 0.95], [[0, 0, 10, 10]]
        )
        assert (tp, fp, fn) == (1, 1, 0)

    def test_no_predictions_no_ground_truth_is_all_zero(self):
        assert match_detections_to_ground_truth([], [], []) == (0, 0, 0, [])


class TestYoloLabelParsing:
    def test_center_box_converts_correctly(self):
        cls, box = yolo_label_to_xyxy("0 0.5 0.5 0.2 0.2", 100, 100)
        assert cls == 0
        assert box == pytest.approx((40.0, 40.0, 60.0, 60.0))

    def test_malformed_line_raises(self):
        with pytest.raises(ValueError):
            yolo_label_to_xyxy("0 0.5 0.5", 100, 100)

    def test_load_ground_truth_missing_file_is_empty(self, tmp_path):
        assert load_yolo_ground_truth(tmp_path / "nope.txt", 100, 100) == []

    def test_load_ground_truth_empty_file_is_empty(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        assert load_yolo_ground_truth(p, 100, 100) == []

    def test_load_ground_truth_parses_multiple_lines(self, tmp_path):
        p = tmp_path / "two.txt"
        p.write_text("0 0.5 0.5 0.2 0.2\n0 0.25 0.25 0.1 0.1\n")
        boxes = load_yolo_ground_truth(p, 100, 100)
        assert len(boxes) == 2


class TestHashing:
    def test_sha256_matches_known_content(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello world")
        import hashlib
        assert sha256_file(p) == hashlib.sha256(b"hello world").hexdigest()

    def test_dhash_identical_images_zero_distance(self):
        img = Image.new("RGB", (64, 64))
        for x in range(64):
            for y in range(64):
                img.putpixel((x, y), (x * 3 % 256, y * 5 % 256, (x + y) % 256))
        assert hamming_distance(dhash(img), dhash(img)) == 0

    def test_dhash_survives_jpeg_style_resize(self, tmp_path):
        """A perceptual hash should tolerate a resize, unlike SHA-256."""
        img = Image.new("RGB", (200, 200))
        for x in range(200):
            for y in range(200):
                img.putpixel((x, y), (x % 256, y % 256, (x * y) % 256))
        resized = img.resize((180, 180), Image.LANCZOS)
        distance = hamming_distance(dhash(img), dhash(resized))
        # Not identical (resizing does change pixel values) but should be
        # much closer than two unrelated random hashes (max distance 64).
        assert distance < 20

    def test_hamming_distance_symmetry(self):
        assert hamming_distance(0b1010, 0b0101) == hamming_distance(0b0101, 0b1010)

    def test_hamming_distance_self_is_zero(self):
        assert hamming_distance(0xABCDEF, 0xABCDEF) == 0


class TestClassificationMetrics:
    def test_confusion_matrix_shape_and_counts(self):
        cm = confusion_matrix(["Clean", "Dusty", "Clean"], ["Clean", "Clean", "Clean"], ["Clean", "Dusty"])
        assert cm == [[2, 0], [1, 0]]

    def test_perfect_predictions_all_metrics_are_one(self):
        y_true = ["Clean", "Dusty", "Hotspot", "Clean"]
        y_pred = ["Clean", "Dusty", "Hotspot", "Clean"]
        labels = ["Clean", "Dusty", "Hotspot"]
        per_class = per_class_prf1(y_true, y_pred, labels)
        for label in labels:
            assert per_class[label]["precision"] == 1.0
            assert per_class[label]["recall"] == 1.0
            assert per_class[label]["f1"] == 1.0
        agg = macro_and_weighted_prf1(per_class)
        assert agg["macro_f1"] == 1.0
        assert agg["weighted_f1"] == 1.0
        assert accuracy(y_true, y_pred) == 1.0

    def test_support_reflects_true_label_counts(self):
        per_class = per_class_prf1(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
        assert per_class["a"]["support"] == 2
        assert per_class["b"]["support"] == 1

    def test_accuracy_empty_input_is_zero_not_error(self):
        assert accuracy([], []) == 0.0

    def test_class_with_no_support_has_zero_metrics_not_division_error(self):
        per_class = per_class_prf1(["a", "a"], ["a", "a"], ["a", "b"])
        assert per_class["b"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}


class TestDefaultOutputRoot:
    def test_env_override_takes_priority(self):
        root = default_output_root(env={"SOLAR_AI_DATA_ROOT": "/custom/root"}, platform="win32")
        assert root == Path("/custom/root") / "evaluation_runs"

    def test_windows_default_is_e_drive(self):
        root = default_output_root(env={}, platform="win32")
        assert root == Path("E:/Solar AI Training Images/evaluation_runs")

    def test_non_windows_falls_back_to_repo_relative(self):
        root = default_output_root(env={}, platform="linux")
        assert root == Path("training/evaluation/runs")
