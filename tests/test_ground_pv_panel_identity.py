"""tests/test_ground_pv_panel_identity.py — unit tests for the pure
geometry-matching primitives in
training/detection/ground_pv/panel_identity.py.
"""
from __future__ import annotations

import pytest

from training.detection.ground_pv.panel_identity import (
    CaptureSlotPanels,
    PanelRef,
    match_panels,
    match_run_to_previous,
    normalized_center,
)


class TestNormalizedCenter:
    def test_center_of_full_image_box_is_half_half(self):
        assert normalized_center([0, 0, 100, 200], 100, 200) == pytest.approx((0.5, 0.5))

    def test_center_of_quarter_box(self):
        cx, cy = normalized_center([0, 0, 50, 50], 100, 100)
        assert (cx, cy) == pytest.approx((0.25, 0.25))

    def test_zero_width_raises(self):
        with pytest.raises(ValueError, match="positive"):
            normalized_center([0, 0, 10, 10], 0, 100)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError, match="positive"):
            normalized_center([0, 0, 10, 10], 100, 0)


class TestMatchPanels:
    def test_empty_current_returns_empty(self):
        assert match_panels([], [PanelRef("prev1", 0.5, 0.5)]) == []

    def test_empty_previous_everything_unmatched(self):
        result = match_panels([PanelRef("cur1", 0.5, 0.5)], [])
        assert len(result) == 1
        assert result[0].current_source_id == "cur1"
        assert result[0].matched_previous_source_id is None
        assert result[0].distance is None

    def test_exact_position_match_has_zero_distance(self):
        result = match_panels(
            [PanelRef("cur1", 0.3, 0.4)],
            [PanelRef("prev1", 0.3, 0.4)],
        )
        assert result[0].matched_previous_source_id == "prev1"
        assert result[0].distance == pytest.approx(0.0)

    def test_beyond_cutoff_is_unmatched(self):
        result = match_panels(
            [PanelRef("cur1", 0.1, 0.1)],
            [PanelRef("prev1", 0.9, 0.9)],
            max_distance=0.25,
        )
        assert result[0].matched_previous_source_id is None
        assert result[0].distance is None

    def test_closest_pair_wins_not_first_in_list(self):
        """Two current panels, two previous panels — each current panel
        must claim its OWN closest previous panel, not just whichever
        comes first in the input list."""
        current = [PanelRef("cur_left", 0.1, 0.5), PanelRef("cur_right", 0.9, 0.5)]
        previous = [PanelRef("prev_right", 0.85, 0.5), PanelRef("prev_left", 0.15, 0.5)]
        result = {r.current_source_id: r.matched_previous_source_id for r in match_panels(current, previous)}
        assert result["cur_left"] == "prev_left"
        assert result["cur_right"] == "prev_right"

    def test_one_previous_panel_claimed_by_only_one_current_panel(self):
        """Two current panels both near the SAME single previous panel —
        only the genuinely closest one may claim it; the other is
        unmatched (a real new panel), never double-assigned."""
        current = [PanelRef("cur_near", 0.51, 0.5), PanelRef("cur_far", 0.6, 0.5)]
        previous = [PanelRef("prev1", 0.5, 0.5)]
        result = match_panels(current, previous, max_distance=0.5)
        by_id = {r.current_source_id: r for r in result}
        assert by_id["cur_near"].matched_previous_source_id == "prev1"
        assert by_id["cur_far"].matched_previous_source_id is None

    def test_deterministic_tie_break_by_source_id(self):
        """Two current panels exactly equidistant from one previous panel
        — the tie must break deterministically (by current_source_id),
        not depend on input/iteration order."""
        current = [PanelRef("cur_b", 0.6, 0.5), PanelRef("cur_a", 0.4, 0.5)]
        previous = [PanelRef("prev1", 0.5, 0.5)]
        result = match_panels(current, previous, max_distance=0.5)
        by_id = {r.current_source_id: r for r in result}
        # "cur_a" sorts before "cur_b", so cur_a claims the tied-distance match.
        assert by_id["cur_a"].matched_previous_source_id == "prev1"
        assert by_id["cur_b"].matched_previous_source_id is None

    def test_result_order_matches_current_input_order(self):
        current = [PanelRef("z", 0.1, 0.1), PanelRef("a", 0.9, 0.9)]
        result = match_panels(current, [])
        assert [r.current_source_id for r in result] == ["z", "a"]


class TestMatchRunToPrevious:
    def test_matches_only_overlapping_slots(self):
        current = [CaptureSlotPanels(0, [PanelRef("c0p1", 0.5, 0.5)])]
        previous = [
            CaptureSlotPanels(0, [PanelRef("p0p1", 0.5, 0.5)]),
            CaptureSlotPanels(1, [PanelRef("p1p1", 0.5, 0.5)]),
        ]
        result = match_run_to_previous(current, previous)
        assert set(result.keys()) == {0}
        assert result[0][0].matched_previous_source_id == "p0p1"

    def test_extra_current_slot_is_all_new(self):
        current = [
            CaptureSlotPanels(0, [PanelRef("c0p1", 0.5, 0.5)]),
            CaptureSlotPanels(1, [PanelRef("c1p1", 0.5, 0.5)]),
        ]
        previous = [CaptureSlotPanels(0, [PanelRef("p0p1", 0.5, 0.5)])]
        result = match_run_to_previous(current, previous)
        assert result[1][0].matched_previous_source_id is None

    def test_missing_current_slot_not_in_result(self):
        current = [CaptureSlotPanels(0, [PanelRef("c0p1", 0.5, 0.5)])]
        previous = [
            CaptureSlotPanels(0, [PanelRef("p0p1", 0.5, 0.5)]),
            CaptureSlotPanels(1, [PanelRef("p1p1", 0.5, 0.5)]),
        ]
        result = match_run_to_previous(current, previous)
        assert 1 not in result

    def test_no_cross_slot_matching(self):
        """A panel in slot 1 must never match a previous panel that was
        only ever seen in slot 0, even at identical position — matching is
        strictly within the same capture slot."""
        current = [CaptureSlotPanels(1, [PanelRef("c1p1", 0.5, 0.5)])]
        previous = [CaptureSlotPanels(0, [PanelRef("p0p1", 0.5, 0.5)])]
        result = match_run_to_previous(current, previous)
        assert result[1][0].matched_previous_source_id is None

    def test_empty_current_returns_empty_dict(self):
        assert match_run_to_previous([], [CaptureSlotPanels(0, [PanelRef("p", 0.5, 0.5)])]) == {}
