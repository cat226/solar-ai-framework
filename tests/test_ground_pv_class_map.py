"""tests/test_ground_pv_class_map.py — class-name normalization / remap for
the ground-level PV research detector."""
from __future__ import annotations

import pytest

from training.detection.ground_pv.class_map import (
    DEFECT_TO_CONDITION,
    RESEARCH_CLASS_NAMES,
    RESEARCH_CLASS_ORDER,
    build_class_id_remap,
    canonical_name,
    remap_is_identity,
)


def test_canonical_order_is_fixed():
    assert RESEARCH_CLASS_ORDER == ("panel", "bird_drop", "cracked", "dusty")
    assert RESEARCH_CLASS_NAMES == {0: "panel", 1: "bird_drop", 2: "cracked", 3: "dusty"}


def test_hotspot_is_not_a_class():
    assert "hotspot" not in RESEARCH_CLASS_ORDER
    with pytest.raises(KeyError):
        canonical_name("hotspot")


@pytest.mark.parametrize("raw,expected", [
    ("panel", "panel"), ("Solar Panel", "panel"),
    ("bird_drop", "bird_drop"), ("Bird-Drop", "bird_drop"), ("bird_dropping", "bird_drop"),
    ("cracked", "cracked"), ("crack", "cracked"), ("Damage", "cracked"),
    ("dusty", "dusty"), ("dust", "dusty"), ("Soiling", "dusty"),
])
def test_alias_resolution(raw, expected):
    assert canonical_name(raw) == expected


def test_unknown_class_fails_closed():
    with pytest.raises(KeyError):
        canonical_name("seagull")


def test_source_order_matching_research_order_is_identity_remap():
    src = {0: "panel", 1: "bird_drop", 2: "cracked", 3: "dusty"}
    remap = build_class_id_remap(src)
    assert remap == {0: 0, 1: 1, 2: 2, 3: 3}
    assert remap_is_identity(remap)


def test_reordered_source_produces_nonidentity_remap():
    # e.g. a re-export in alphabetical order: bird_drop, cracked, dusty, panel
    src = {0: "bird_drop", 1: "cracked", 2: "dusty", 3: "panel"}
    remap = build_class_id_remap(src)
    assert remap == {0: 1, 1: 2, 2: 3, 3: 0}
    assert not remap_is_identity(remap)


def test_ambiguous_source_taxonomy_rejected():
    with pytest.raises(ValueError):
        build_class_id_remap({0: "panel", 1: "solar_panel"})


def test_defect_to_condition_covers_all_defect_classes():
    for c in RESEARCH_CLASS_ORDER:
        if c == "panel":
            continue
        assert c in DEFECT_TO_CONDITION
    assert set(DEFECT_TO_CONDITION.values()) == {"BirdDrop", "Damage", "Dust"}
