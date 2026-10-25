"""tests/test_condition_error_analysis.py — Phase 15C: unit tests for the
deterministic error-analysis primitives in
training/detection/ground_pv/condition_error_analysis.py. Pure logic only
— no torch, no ultralytics, no dataset/checkpoint required."""
from __future__ import annotations

import pytest

from training.detection.ground_pv.condition_error_analysis import (
    FP_CATEGORIES,
    box_area_fraction,
    categorize_false_positive,
    find_overlapping_gt_pairs,
    iou_xyxy,
    match_class_instances,
    nearest_same_class_confidence,
    parse_yolo_label_text,
    quartile_cuts,
    size_bin_label,
    summarize_confidences,
    touches_border,
)


class TestIoU:
    def test_identical_boxes_iou_one(self):
        assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_zero(self):
        assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_half_overlap(self):
        # box A: 0..10 x 0..10 (area 100); box B: 5..15 x 0..10 (area 100)
        # intersection: 5..10 x 0..10 = area 50; union = 100+100-50=150
        assert iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)

    def test_degenerate_zero_area_box(self):
        assert iou_xyxy((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0

    def test_touching_edges_no_overlap(self):
        assert iou_xyxy((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


class TestBoxGeometry:
    def test_area_fraction(self):
        assert box_area_fraction((0, 0, 10, 10), img_w=100, img_h=100) == pytest.approx(0.01)

    def test_area_fraction_zero_image(self):
        assert box_area_fraction((0, 0, 10, 10), img_w=0, img_h=0) == 0.0

    def test_touches_border_true_at_edge(self):
        assert touches_border((0, 5, 10, 15), img_w=100, img_h=100, margin_px=2.0) is True

    def test_touches_border_false_in_middle(self):
        assert touches_border((40, 40, 60, 60), img_w=100, img_h=100, margin_px=2.0) is False

    def test_touches_border_right_edge(self):
        assert touches_border((80, 40, 99, 60), img_w=100, img_h=100, margin_px=2.0) is True


class TestParseYoloLabel:
    def test_single_box_center(self):
        text = "0 0.5 0.5 0.2 0.2"
        out = parse_yolo_label_text(text, img_w=100, img_h=100)
        assert out == [(0, (40.0, 40.0, 60.0, 60.0))]

    def test_multiple_lines_and_classes(self):
        text = "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n"
        out = parse_yolo_label_text(text, img_w=100, img_h=100)
        assert len(out) == 2
        assert out[0][0] == 0
        assert out[1][0] == 1

    def test_blank_lines_skipped(self):
        text = "0 0.5 0.5 0.2 0.2\n\n\n"
        out = parse_yolo_label_text(text, img_w=100, img_h=100)
        assert len(out) == 1

    def test_malformed_line_skipped_not_raised(self):
        text = "0 0.5 0.5 0.2\nnot a valid line at all\n0 0.1 0.1 0.1 0.1"
        out = parse_yolo_label_text(text, img_w=100, img_h=100)
        assert len(out) == 1

    def test_empty_text(self):
        assert parse_yolo_label_text("", img_w=100, img_h=100) == []


class TestMatchClassInstances:
    def test_perfect_single_match(self):
        gt = [(0, 0, 10, 10)]
        preds = [((0, 0, 10, 10), 0.9)]
        tp, fp, fn = match_class_instances(gt, preds, iou_threshold=0.5)
        assert tp == [0] and fp == [] and fn == []

    def test_no_gt_all_preds_are_fp(self):
        gt: list = []
        preds = [((0, 0, 10, 10), 0.9), ((20, 20, 30, 30), 0.5)]
        tp, fp, fn = match_class_instances(gt, preds, iou_threshold=0.5)
        assert tp == [] and sorted(fp) == [0, 1] and fn == []

    def test_no_preds_all_gt_are_fn(self):
        gt = [(0, 0, 10, 10), (20, 20, 30, 30)]
        preds: list = []
        tp, fp, fn = match_class_instances(gt, preds, iou_threshold=0.5)
        assert tp == [] and fp == [] and sorted(fn) == [0, 1]

    def test_low_iou_prediction_is_fp_and_gt_is_fn(self):
        gt = [(0, 0, 10, 10)]
        # overlaps but IoU well below 0.5 threshold
        preds = [((8, 8, 20, 20), 0.9)]
        tp, fp, fn = match_class_instances(gt, preds, iou_threshold=0.5)
        assert tp == [] and fp == [0] and fn == [0]

    def test_higher_confidence_wins_the_gt_match(self):
        gt = [(0, 0, 10, 10)]
        # two predictions both overlap the same GT well enough; higher-conf wins
        preds = [((0, 0, 10, 10), 0.4), ((1, 0, 10, 10), 0.9)]
        tp, fp, fn = match_class_instances(gt, preds, iou_threshold=0.5)
        assert tp == [1]  # index 1 (conf 0.9) claims the match
        assert fp == [0]  # index 0 becomes a duplicate FP
        assert fn == []

    def test_two_gt_two_preds_disjoint_regions(self):
        gt = [(0, 0, 10, 10), (100, 100, 110, 110)]
        preds = [((0, 0, 10, 10), 0.9), ((100, 100, 110, 110), 0.8)]
        tp, fp, fn = match_class_instances(gt, preds, iou_threshold=0.5)
        assert sorted(tp) == [0, 1] and fp == [] and fn == []

    def test_exact_iou_threshold_boundary_counts_as_match(self):
        # box A area 100, box B area 100 shifted by 5 in x -> IoU = 50/150 = 0.3333
        gt = [(0, 0, 10, 10)]
        preds = [((5, 0, 15, 10), 0.9)]
        tp, fp, fn = match_class_instances(gt, preds, iou_threshold=1 / 3)
        assert tp == [0]


class TestQuartileBinning:
    def test_quartile_cuts_basic(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8]
        q1, q2, q3 = quartile_cuts(values)
        assert q1 < q2 < q3

    def test_quartile_cuts_needs_four_values(self):
        with pytest.raises(ValueError):
            quartile_cuts([1, 2, 3])

    def test_size_bin_label_covers_all_four_bins(self):
        cuts = (10.0, 20.0, 30.0)
        assert size_bin_label(5.0, cuts) == "tiny"
        assert size_bin_label(15.0, cuts) == "small"
        assert size_bin_label(25.0, cuts) == "medium"
        assert size_bin_label(35.0, cuts) == "large"

    def test_size_bin_label_boundary_values_inclusive_lower(self):
        cuts = (10.0, 20.0, 30.0)
        assert size_bin_label(10.0, cuts) == "tiny"
        assert size_bin_label(30.0, cuts) == "medium"


class TestCategorizeFalsePositive:
    IMG_W = IMG_H = 100
    TINY = 0.01  # 1% of image area

    def test_duplicate_of_accepted_same_class_prediction(self):
        fp_box = (0, 0, 10, 10)
        cat = categorize_false_positive(
            fp_box, same_image_other_accepted_boxes=[(1, 1, 11, 11)],
            same_image_gt_same_class=[], same_image_gt_other_class=[],
            img_w=self.IMG_W, img_h=self.IMG_H, tiny_area_fraction=self.TINY,
        )
        assert cat == "duplicate/overlapping prediction"

    def test_overlaps_different_class_gt(self):
        fp_box = (40, 40, 60, 60)
        cat = categorize_false_positive(
            fp_box, same_image_other_accepted_boxes=[],
            same_image_gt_same_class=[], same_image_gt_other_class=[(40, 40, 60, 60)],
            img_w=self.IMG_W, img_h=self.IMG_H, tiny_area_fraction=self.TINY,
        )
        assert cat == "other defect mistaken for this class"

    def test_low_iou_overlap_with_same_class_gt_is_localization(self):
        fp_box = (0, 0, 10, 10)
        # same-class GT overlaps a little (low IoU) but not enough to have matched
        cat = categorize_false_positive(
            fp_box, same_image_other_accepted_boxes=[],
            same_image_gt_same_class=[(8, 8, 20, 20)], same_image_gt_other_class=[],
            img_w=self.IMG_W, img_h=self.IMG_H, tiny_area_fraction=self.TINY,
        )
        assert cat == "localization (low-IoU overlap with same-class GT)"

    def test_edge_artifact(self):
        fp_box = (0, 40, 10, 60)  # touches left border
        cat = categorize_false_positive(
            fp_box, same_image_other_accepted_boxes=[],
            same_image_gt_same_class=[], same_image_gt_other_class=[],
            img_w=self.IMG_W, img_h=self.IMG_H, tiny_area_fraction=self.TINY,
        )
        assert cat == "edge/crop artifact"

    def test_tiny_region(self):
        fp_box = (50, 50, 51, 51)  # area 1 / 10000 = 0.0001 << tiny_area_fraction
        cat = categorize_false_positive(
            fp_box, same_image_other_accepted_boxes=[],
            same_image_gt_same_class=[], same_image_gt_other_class=[],
            img_w=self.IMG_W, img_h=self.IMG_H, tiny_area_fraction=self.TINY,
        )
        assert cat == "tiny/low-information region"

    def test_background_no_overlap_anything(self):
        fp_box = (40, 40, 60, 60)  # area 400/10000=0.04, not tiny, not at border
        cat = categorize_false_positive(
            fp_box, same_image_other_accepted_boxes=[],
            same_image_gt_same_class=[], same_image_gt_other_class=[],
            img_w=self.IMG_W, img_h=self.IMG_H, tiny_area_fraction=self.TINY,
        )
        assert cat == "background / non-panel surface"

    def test_precedence_duplicate_beats_everything_else(self):
        # This FP both touches the border AND duplicates an accepted prediction —
        # duplicate must win per the documented precedence order.
        fp_box = (0, 0, 10, 10)
        cat = categorize_false_positive(
            fp_box, same_image_other_accepted_boxes=[(1, 1, 11, 11)],
            same_image_gt_same_class=[], same_image_gt_other_class=[],
            img_w=self.IMG_W, img_h=self.IMG_H, tiny_area_fraction=self.TINY,
        )
        assert cat == "duplicate/overlapping prediction"

    def test_all_categories_are_from_the_documented_set(self):
        # sanity: every category this function can return is listed in FP_CATEGORIES
        cases = [
            ((0, 0, 10, 10), [(1, 1, 11, 11)], [], []),
            ((40, 40, 60, 60), [], [], [(40, 40, 60, 60)]),
            ((0, 0, 10, 10), [], [(8, 8, 20, 20)], []),
            ((0, 40, 10, 60), [], [], []),
            ((50, 50, 51, 51), [], [], []),
            ((40, 40, 60, 60), [], [], []),
        ]
        for fp_box, others, same_cls, other_cls in cases:
            cat = categorize_false_positive(
                fp_box, same_image_other_accepted_boxes=others,
                same_image_gt_same_class=same_cls, same_image_gt_other_class=other_cls,
                img_w=self.IMG_W, img_h=self.IMG_H, tiny_area_fraction=self.TINY,
            )
            assert cat in FP_CATEGORIES


class TestNearestSameClassConfidence:
    def test_returns_none_when_no_overlap(self):
        fn_box = (0, 0, 10, 10)
        preds = [((50, 50, 60, 60), 0.3)]
        assert nearest_same_class_confidence(fn_box, preds) is None

    def test_returns_max_confidence_among_overlapping(self):
        fn_box = (0, 0, 10, 10)
        preds = [((1, 1, 9, 9), 0.05), ((2, 2, 8, 8), 0.15), ((90, 90, 99, 99), 0.99)]
        assert nearest_same_class_confidence(fn_box, preds) == pytest.approx(0.15)

    def test_empty_preds_returns_none(self):
        assert nearest_same_class_confidence((0, 0, 10, 10), []) is None


class TestFindOverlappingGtPairs:
    def test_no_pairs_when_disjoint(self):
        gt = [(0, (0, 0, 10, 10)), (1, (50, 50, 60, 60))]
        assert find_overlapping_gt_pairs(gt, iou_threshold=0.8) == []

    def test_same_class_near_duplicate_detected(self):
        gt = [(0, (0, 0, 10, 10)), (0, (0.5, 0.5, 10.5, 10.5))]
        pairs = find_overlapping_gt_pairs(gt, iou_threshold=0.8)
        assert len(pairs) == 1
        i, j, iou = pairs[0]
        assert (i, j) == (0, 1)
        assert iou > 0.8

    def test_different_class_conflicting_label_detected(self):
        gt = [(0, (0, 0, 10, 10)), (2, (0, 0, 10, 10))]
        pairs = find_overlapping_gt_pairs(gt, iou_threshold=0.8)
        assert len(pairs) == 1
        assert pairs[0][2] == pytest.approx(1.0)

    def test_threshold_excludes_moderate_overlap(self):
        gt = [(0, (0, 0, 10, 10)), (0, (5, 0, 15, 10))]  # IoU = 1/3
        assert find_overlapping_gt_pairs(gt, iou_threshold=0.8) == []

    def test_single_box_no_pairs(self):
        assert find_overlapping_gt_pairs([(0, (0, 0, 10, 10))], iou_threshold=0.8) == []


class TestSummarizeConfidences:
    def test_empty_returns_zero_n_and_none_stats(self):
        s = summarize_confidences([])
        assert s["n"] == 0
        assert s["mean"] is None and s["median"] is None

    def test_basic_stats(self):
        s = summarize_confidences([0.1, 0.2, 0.3, 0.4, 0.5])
        assert s["n"] == 5
        assert s["min"] == pytest.approx(0.1)
        assert s["max"] == pytest.approx(0.5)
        assert s["median"] == pytest.approx(0.3)

    def test_fewer_than_four_values_no_quartiles(self):
        s = summarize_confidences([0.5, 0.6])
        assert s["n"] == 2
        assert s["p25"] is None and s["p75"] is None
        assert s["mean"] == pytest.approx(0.55)
