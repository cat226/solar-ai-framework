"""Ground-level / close-up RGB PV defect research detector.

A SEPARATE research model from the production aerial detector
(weights/yolo_solar.pt). Nothing in this package loads, imports, mutates,
or re-trains the production artifact. See docs/GROUND_PV_RESEARCH.md.

Target classes (normalized, fixed order):
    0 = panel
    1 = bird_drop
    2 = cracked
    3 = dusty

Hotspot is deliberately NOT a class here - RGB photographs cannot validate
thermal-hotspot semantics (see the project's existing Hotspot data audit).
"""
from __future__ import annotations

from training.detection.ground_pv.class_map import (
    RESEARCH_CLASS_NAMES,
    RESEARCH_CLASS_ORDER,
    build_class_id_remap,
)
from training.detection.ground_pv.dynamic_panel_detection import (
    DynamicPanelConfig,
    detect_dynamic_panels,
)

__all__ = [
    "RESEARCH_CLASS_NAMES",
    "RESEARCH_CLASS_ORDER",
    "build_class_id_remap",
    "DynamicPanelConfig",
    "detect_dynamic_panels",
]
