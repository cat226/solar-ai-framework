"""tests/test_ui_helpers.py - Tests for utils/ui_helpers.py's pure,
Streamlit-free logic.

Streamlit's own rendering calls (st.subheader, st.metric, ...) require a
running app/session context and are not unit-tested here - only the pure
decision/text logic extracted specifically so it CAN be tested without
one. The property under test matters for this project's no-fabrication
discipline: the whole-image classification result must never be
presented as if it were a panel-level (or, for thermal input, a
validated electrical-hotspot) diagnosis.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from services.pipeline import PipelineResult
from utils.ui_helpers import whole_image_classification_caption


def _make_result(detection_successful: bool) -> PipelineResult:
    result = PipelineResult()
    result.detection_result = MagicMock(detection_successful=detection_successful)
    return result


class TestWholeImageClassificationCaption:
    def test_always_labels_as_whole_image_not_panel_level(self):
        for detected in (True, False):
            caption = whole_image_classification_caption(_make_result(detected))
            assert "Whole-image classification" in caption
            assert "not panel-level diagnosis" in caption

    def test_zero_detection_caption_explains_it_is_the_only_signal(self):
        caption = whole_image_classification_caption(_make_result(False))
        assert "No panel was detected" in caption
        assert "not a substitute for a panel-level result" in caption

    def test_detection_present_caption_points_to_panel_detail(self):
        caption = whole_image_classification_caption(_make_result(True))
        assert "Panel-by-Panel Detail" in caption
        assert "No panel was detected" not in caption
