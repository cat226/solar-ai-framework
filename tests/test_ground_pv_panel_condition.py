"""tests/test_ground_pv_panel_condition.py — defect/panel association and the
research visual condition assessment."""
from __future__ import annotations

from training.detection.ground_pv.panel_condition import assess_image


def _panel(x1, y1, x2, y2, conf=0.9):
    return {"class_name": "panel", "confidence": conf, "bbox": [x1, y1, x2, y2]}


def _defect(name, x1, y1, x2, y2, conf=0.8):
    return {"class_name": name, "confidence": conf, "bbox": [x1, y1, x2, y2]}


def test_panel_with_no_defects_is_clean():
    out = assess_image([_panel(0, 0, 100, 100)])
    assert out["num_panels"] == 1
    assert out["panels"][0]["condition"] == ["Clean"]
    assert out["panels"][0]["defects"] == []


def test_bird_drop_inside_panel_yields_birddrop_condition():
    out = assess_image([_panel(0, 0, 100, 100), _defect("bird_drop", 40, 40, 55, 55)])
    p = out["panels"][0]
    assert p["condition"] == ["BirdDrop"]
    assert len(p["defects"]) == 1
    assert p["defects"][0]["assoc"] == "centre_inside"


def test_panel_can_carry_multiple_conditions():
    out = assess_image([
        _panel(0, 0, 100, 100),
        _defect("bird_drop", 10, 10, 20, 20),
        _defect("dusty", 60, 60, 90, 90),
    ])
    assert sorted(out["panels"][0]["condition"]) == ["BirdDrop", "Dust"]


def test_defect_assigned_to_panel_with_greater_overlap():
    out = assess_image([
        _panel(0, 0, 100, 100, conf=0.9),
        _panel(90, 0, 190, 100, conf=0.9),
        _defect("cracked", 20, 20, 40, 40),
    ])
    owner = next(p for p in out["panels"] if p["defects"])
    assert owner["panel_bbox"] == [0, 0, 100, 100]
    assert owner["condition"] == ["Damage"]


def test_defect_matching_no_panel_is_unassociated_not_dropped():
    out = assess_image([_panel(0, 0, 50, 50), _defect("dusty", 500, 500, 550, 550)])
    assert out["panels"][0]["condition"] == ["Clean"]
    assert len(out["unassociated_defects"]) == 1
    assert out["unassociated_defects"][0]["type"] == "dusty"


def test_large_soiling_region_overlapping_panel_associates_via_iou():
    # centre of the soiling box is outside the panel, but IoU is high
    out = assess_image([
        _panel(0, 0, 100, 100),
        _defect("dusty", 50, 50, 200, 200),
    ], min_iou=0.05)
    assert out["panels"][0]["condition"] == ["Dust"]
    assert out["panels"][0]["defects"][0]["assoc"] == "iou"


def test_panel_ids_are_deterministic_by_confidence_then_position():
    out = assess_image([
        _panel(200, 0, 300, 100, conf=0.5),
        _panel(0, 0, 100, 100, conf=0.99),
    ])
    assert out["panels"][0]["panel_id"] == 1
    assert out["panels"][0]["panel_confidence"] == 0.99


def test_output_schema_contract():
    out = assess_image([_panel(0, 0, 10, 10), _defect("bird_drop", 1, 1, 3, 3)])
    assert set(out) >= {"num_panels", "num_defect_detections", "panels",
                        "unassociated_defects", "assessment_type", "association_rule"}
    panel = out["panels"][0]
    assert set(panel) == {"panel_id", "panel_bbox", "panel_confidence", "defects", "condition"}
    defect = panel["defects"][0]
    assert set(defect) == {"type", "confidence", "bbox", "assoc"}
    assert "not a validated diagnosis" in out["assessment_type"]
