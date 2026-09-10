"""tests/test_ground_pv_condition_taxonomy.py — Phase 15A canonical
4-class RGB panel-condition taxonomy."""
from __future__ import annotations

import pytest

from training.detection.ground_pv.condition_taxonomy import (
    CONDITION_4CLASS_IDS,
    CONDITION_4CLASS_NAMES,
    CONDITION_4CLASS_ORDER,
    REJECTED_CLASSES,
    assert_active_taxonomy,
    is_active_class,
)


def test_canonical_order_is_exactly_four_classes():
    assert CONDITION_4CLASS_ORDER == ("CLEAN", "DUSTY", "BIRD_DROP", "PHYSICAL_DAMAGE")
    assert len(CONDITION_4CLASS_ORDER) == 4


def test_canonical_ids_are_contiguous_from_zero():
    assert CONDITION_4CLASS_NAMES == {0: "CLEAN", 1: "DUSTY", 2: "BIRD_DROP", 3: "PHYSICAL_DAMAGE"}
    assert CONDITION_4CLASS_IDS == {"CLEAN": 0, "DUSTY": 1, "BIRD_DROP": 2, "PHYSICAL_DAMAGE": 3}


class TestIsActiveClass:
    def test_all_four_active_classes_recognized(self):
        for name in CONDITION_4CLASS_ORDER:
            assert is_active_class(name) is True

    def test_electrical_damage_rejected(self):
        assert is_active_class("ELECTRICAL_DAMAGE") is False

    def test_snow_rejected(self):
        assert is_active_class("SNOW") is False
        assert is_active_class("SNOW-COVERED") is False

    def test_generic_defective_rejected(self):
        assert is_active_class("DEFECTIVE") is False

    def test_panel_rejected(self):
        assert is_active_class("PANEL") is False

    def test_hotspot_rejected(self):
        assert is_active_class("HOTSPOT") is False


class TestRejectedClassesSet:
    def test_rejected_classes_never_overlap_active(self):
        assert REJECTED_CLASSES.isdisjoint(set(CONDITION_4CLASS_ORDER))

    def test_rejected_set_contains_expected_names(self):
        assert REJECTED_CLASSES == frozenset({
            "ELECTRICAL_DAMAGE", "SNOW", "SNOW-COVERED", "DEFECTIVE", "PANEL", "HOTSPOT",
        })


class TestAssertActiveTaxonomy:
    def test_accepts_correct_dict_taxonomy(self):
        assert_active_taxonomy({0: "CLEAN", 1: "DUSTY", 2: "BIRD_DROP", 3: "PHYSICAL_DAMAGE"})  # no raise

    def test_accepts_correct_list_taxonomy(self):
        assert_active_taxonomy(["CLEAN", "DUSTY", "BIRD_DROP", "PHYSICAL_DAMAGE"])  # no raise

    def test_rejects_five_class_taxonomy_with_electrical_damage(self):
        with pytest.raises(ValueError, match="ELECTRICAL_DAMAGE"):
            assert_active_taxonomy({0: "CLEAN", 1: "DUSTY", 2: "BIRD_DROP", 3: "PHYSICAL_DAMAGE", 4: "ELECTRICAL_DAMAGE"})

    def test_rejects_snow_taxonomy(self):
        with pytest.raises(ValueError):
            assert_active_taxonomy({0: "CLEAN", 1: "SNOW"})

    def test_rejects_wrong_order(self):
        with pytest.raises(ValueError):
            assert_active_taxonomy({0: "DUSTY", 1: "CLEAN", 2: "BIRD_DROP", 3: "PHYSICAL_DAMAGE"})

    def test_rejects_panel_only_taxonomy(self):
        # e.g. accidentally passing the PANEL DETECTOR's own taxonomy
        with pytest.raises(ValueError, match="PANEL"):
            assert_active_taxonomy({0: "panel", 1: "PANEL"})

    def test_rejects_empty_taxonomy(self):
        with pytest.raises(ValueError):
            assert_active_taxonomy({})

    def test_rejects_subset_taxonomy(self):
        with pytest.raises(ValueError):
            assert_active_taxonomy({0: "CLEAN", 1: "DUSTY"})

    def test_error_message_never_silently_passes(self):
        """A 5-class model's taxonomy must produce a clear, actionable error
        - never a silent pass-through as if it were 4-class."""
        with pytest.raises(ValueError) as exc_info:
            assert_active_taxonomy({0: "CLEAN", 1: "DUSTY", 2: "BIRD_DROP", 3: "PHYSICAL_DAMAGE", 4: "ELECTRICAL_DAMAGE"})
        assert "4-class" in str(exc_info.value) or "active" in str(exc_info.value)
