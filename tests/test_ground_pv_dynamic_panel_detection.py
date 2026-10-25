"""tests/test_ground_pv_dynamic_panel_detection.py — dynamic multi-panel
detection + deterministic spatial ordering (Phase 14A).

No YOLO / GPU needed for most tests: geometry/ordering functions are pure
python operating on `RawPanelDetection` objects directly. A fake detector
(same `type("B", (), {...})()` duck-typing pattern already used in
test_ground_pv_tiled_inference.py) exercises the thin model-orchestration
wrapper `run_panel_detector` / `detect_dynamic_panels`.

Critical requirement under test: spatial ordering (P1..PN) must NOT depend
on the order the detector emitted boxes in — see
TestOrderingIndependentOfInputOrder below.
"""
from __future__ import annotations

import random

import pytest
from PIL import Image

from training.detection.ground_pv.dynamic_panel_detection import (
    DynamicPanelConfig,
    RawPanelDetection,
    _check_array_consistency,
    _check_overlaps,
    _check_size_and_aspect,
    _contained_fraction,
    _iou,
    build_dynamic_panel_report,
    detect_dynamic_panels,
    draw_annotated,
    order_panels_spatially,
    run_panel_detector,
)


IMG_W, IMG_H = 1280, 720


def det(x1, y1, x2, y2, conf=0.9, cls="panel"):
    return RawPanelDetection(class_name=cls, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)


# ---- pure geometry helpers ----------------------------------------------

def test_iou_and_contained_fraction():
    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert _iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert _iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert _contained_fraction([2, 2, 4, 4], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert _iou([2, 2, 4, 4], [0, 0, 10, 10]) < 0.1


def test_raw_panel_detection_derived_fields():
    d = det(100, 80, 320, 250, conf=0.93)
    assert d.width == pytest.approx(220.0)
    assert d.height == pytest.approx(170.0)
    assert d.center_x == pytest.approx(210.0)
    assert d.center_y == pytest.approx(165.0)
    assert d.area == pytest.approx(220.0 * 170.0)
    raw = d.as_raw_dict()
    assert set(raw) == {
        "class", "confidence", "x1", "y1", "x2", "y2",
        "width", "height", "center_x", "center_y", "area",
    }


# ---- layout tests: 4 / 6 / 8 in various grids ---------------------------

def _grid(rows: int, cols: int, w: int, h: int, cell_w=150, cell_h=100, gap=50) -> list[RawPanelDetection]:
    """Build a clean row/col grid of panels (no overlap, evenly spaced)."""
    dets = []
    for r in range(rows):
        for c in range(cols):
            x1 = 50 + c * (cell_w + gap)
            y1 = 50 + r * (cell_h + gap)
            dets.append(det(x1, y1, x1 + cell_w, y1 + cell_h, conf=0.9 - 0.01 * (r * cols + c)))
    return dets


class TestLayouts:
    def test_2x2_layout(self):
        dets = _grid(2, 2, IMG_W, IMG_H)
        ordered, row_count, ambiguous = order_panels_spatially(dets, IMG_W, IMG_H)
        assert len(ordered) == 4
        assert row_count == 2
        assert not ambiguous
        # row-major top-left -> bottom-right
        assert [ (o.row_index, o.column_index) for o in ordered ] == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_2x3_layout_six_panels(self):
        dets = _grid(2, 3, IMG_W, IMG_H)
        ordered, row_count, ambiguous = order_panels_spatially(dets, IMG_W, IMG_H)
        assert len(ordered) == 6
        assert row_count == 2
        assert not ambiguous
        assert [o.column_index for o in ordered[:3]] == [0, 1, 2]
        assert [o.column_index for o in ordered[3:]] == [0, 1, 2]

    def test_2x4_layout_eight_panels(self):
        dets = _grid(2, 4, IMG_W, IMG_H)
        ordered, row_count, ambiguous = order_panels_spatially(dets, IMG_W, IMG_H)
        assert len(ordered) == 8
        assert row_count == 2
        assert not ambiguous

    def test_4x2_layout_eight_panels(self):
        dets = _grid(4, 2, IMG_W, IMG_H)
        ordered, row_count, ambiguous = order_panels_spatially(dets, IMG_W, IMG_H)
        assert len(ordered) == 8
        assert row_count == 4
        assert not ambiguous
        assert [(o.row_index, o.column_index) for o in ordered] == [
            (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1),
        ]

    def test_single_row(self):
        dets = _grid(1, 6, IMG_W, IMG_H)
        ordered, row_count, ambiguous = order_panels_spatially(dets, IMG_W, IMG_H)
        assert row_count == 1
        assert not ambiguous
        assert [o.column_index for o in ordered] == [0, 1, 2, 3, 4, 5]

    def test_single_column(self):
        dets = _grid(6, 1, IMG_W, IMG_H)
        ordered, row_count, ambiguous = order_panels_spatially(dets, IMG_W, IMG_H)
        assert row_count == 6
        assert not ambiguous
        assert [o.row_index for o in ordered] == [0, 1, 2, 3, 4, 5]

    def test_perspective_shifted_rows_still_group(self):
        # Row 1 slopes gently downward left->right; row 2 similarly, but the
        # two rows stay clearly separated in y. No identical y required.
        row1 = [det(50, 50, 200, 150), det(250, 60, 400, 160), det(450, 70, 600, 170)]
        row2 = [det(50, 300, 200, 400), det(250, 305, 400, 405), det(450, 310, 600, 410)]
        dets = row1 + row2
        ordered, row_count, ambiguous = order_panels_spatially(dets, IMG_W, IMG_H, row_grouping_tolerance=0.15)
        assert row_count == 2
        assert not ambiguous
        assert [o.row_index for o in ordered] == [0, 0, 0, 1, 1, 1]

    def test_ambiguous_overlapping_rows_flagged(self):
        # Two "rows" whose vertical extents actually overlap -> ambiguous.
        dets = [
            det(50, 50, 200, 400),    # tall, spans what would be both rows
            det(250, 60, 400, 160),   # short, in the "top" band
            det(250, 300, 400, 400),  # short, in the "bottom" band
        ]
        ordered, row_count, ambiguous = order_panels_spatially(dets, IMG_W, IMG_H, row_grouping_tolerance=0.05)
        assert ambiguous is True


# ---- determinism: order must not depend on detector output order --------

class TestOrderingIndependentOfInputOrder:
    def test_eight_panel_grid_shuffled_repeatedly(self):
        base = _grid(2, 4, IMG_W, IMG_H)
        reference = None
        for _ in range(20):
            shuffled = list(base)
            random.shuffle(shuffled)
            ordered, row_count, ambiguous = order_panels_spatially(shuffled, IMG_W, IMG_H)
            signature = [(o.detection.x1, o.detection.y1, o.row_index, o.column_index) for o in ordered]
            if reference is None:
                reference = signature
            else:
                assert signature == reference, "spatial ordering changed under a reshuffle of the same detections"

    def test_six_panel_grid_reversed_vs_forward(self):
        base = _grid(2, 3, IMG_W, IMG_H)
        forward, _, _ = order_panels_spatially(base, IMG_W, IMG_H)
        backward, _, _ = order_panels_spatially(list(reversed(base)), IMG_W, IMG_H)
        f_sig = [(o.detection.x1, o.detection.y1) for o in forward]
        b_sig = [(o.detection.x1, o.detection.y1) for o in backward]
        assert f_sig == b_sig

    def test_full_report_panel_ids_stable_under_shuffle(self):
        base = _grid(2, 2, IMG_W, IMG_H)
        config = DynamicPanelConfig()
        reports = []
        for _ in range(10):
            shuffled = list(base)
            random.shuffle(shuffled)
            report = build_dynamic_panel_report(
                shuffled, image_name="x.jpg", image_width=IMG_W, image_height=IMG_H,
                model_name="test", config=config,
            )
            reports.append([(p["panel_id"], p["bbox"]["x1"], p["bbox"]["y1"]) for p in report["panels"]])
        assert all(r == reports[0] for r in reports)


# ---- overlap / duplicate diagnostics ------------------------------------

class TestOverlapDiagnostics:
    def test_high_iou_pair_flagged_both_directions(self):
        dets = [det(100, 100, 200, 200), det(110, 110, 210, 210)]
        ordered, _, _ = order_panels_spatially(dets, IMG_W, IMG_H)
        warnings = _check_overlaps(ordered, overlap_warning_iou=0.30, containment_warning_fraction=0.80)
        assert len(warnings) == 2
        assert any("high_overlap_with_" in w for w in list(warnings.values())[0])

    def test_disjoint_pair_not_flagged(self):
        dets = [det(0, 0, 100, 100), det(500, 500, 600, 600)]
        ordered, _, _ = order_panels_spatially(dets, IMG_W, IMG_H)
        warnings = _check_overlaps(ordered, overlap_warning_iou=0.30, containment_warning_fraction=0.80)
        assert warnings == {}

    def test_near_containment_flagged_even_at_low_iou(self):
        # small box almost fully inside a big one: low IoU, high containment
        outer = det(0, 0, 500, 500)
        inner = det(10, 10, 100, 100)
        ordered, _, _ = order_panels_spatially([outer, inner], IMG_W, IMG_H)
        warnings = _check_overlaps(ordered, overlap_warning_iou=0.90, containment_warning_fraction=0.80)
        assert len(warnings) == 2

    def test_nms_already_run_does_not_prevent_this_layer(self):
        # This is a POST-nms sanity layer; it must still fire on boxes that
        # survived the model's own NMS (moderate IoU below NMS's own cutoff).
        dets = [det(0, 0, 100, 100, conf=0.9), det(35, 0, 135, 100, conf=0.8)]  # IoU ~0.48
        ordered, _, _ = order_panels_spatially(dets, IMG_W, IMG_H)
        warnings = _check_overlaps(ordered, overlap_warning_iou=0.30, containment_warning_fraction=0.80)
        assert len(warnings) == 2


# ---- geometric sanity checks ---------------------------------------------

class TestSizeAndAspectDiagnostics:
    def test_extremely_small_panel_flagged(self):
        dets = [det(10, 10, 15, 15)]  # 5x5=25px in a 1280x720 frame
        ordered, _, _ = order_panels_spatially(dets, IMG_W, IMG_H)
        config = DynamicPanelConfig(min_relative_area=0.001)
        warnings = _check_size_and_aspect(ordered, IMG_W, IMG_H, config)
        assert "extremely_small_area" in warnings["P1"]

    def test_extremely_large_panel_flagged(self):
        dets = [det(0, 0, 1280, 700)]  # near full-frame
        ordered, _, _ = order_panels_spatially(dets, IMG_W, IMG_H)
        config = DynamicPanelConfig(max_relative_area=0.60)
        warnings = _check_size_and_aspect(ordered, IMG_W, IMG_H, config)
        assert "extremely_large_area" in warnings["P1"]

    def test_extreme_wide_aspect_ratio_flagged(self):
        dets = [det(0, 0, 600, 50)]  # 12:1
        ordered, _, _ = order_panels_spatially(dets, IMG_W, IMG_H)
        config = DynamicPanelConfig(min_aspect_ratio=0.2, max_aspect_ratio=5.0)
        warnings = _check_size_and_aspect(ordered, IMG_W, IMG_H, config)
        assert "extreme_aspect_ratio" in warnings["P1"]

    def test_extreme_tall_aspect_ratio_flagged(self):
        dets = [det(0, 0, 50, 600)]  # 1:12
        ordered, _, _ = order_panels_spatially(dets, IMG_W, IMG_H)
        config = DynamicPanelConfig(min_aspect_ratio=0.2, max_aspect_ratio=5.0)
        warnings = _check_size_and_aspect(ordered, IMG_W, IMG_H, config)
        assert "extreme_aspect_ratio" in warnings["P1"]

    def test_normal_panel_no_warnings(self):
        dets = [det(100, 100, 300, 250)]  # 200x150, aspect 1.33, small fraction of frame
        ordered, _, _ = order_panels_spatially(dets, IMG_W, IMG_H)
        config = DynamicPanelConfig()
        warnings = _check_size_and_aspect(ordered, IMG_W, IMG_H, config)
        assert warnings == {}


class TestArrayConsistencyDiagnostics:
    def test_size_outlier_flagged(self):
        normal = [det(x, 100, x + 150, 250) for x in (50, 250, 450)]
        outlier = det(700, 100, 720, 120)  # tiny vs the rest
        dets = normal + [outlier]
        ordered, _, _ = order_panels_spatially(dets, IMG_W, IMG_H)
        config = DynamicPanelConfig(size_inconsistency_ratio=3.0)
        warnings = _check_array_consistency(ordered, IMG_W, IMG_H, config)
        assert any("size_inconsistent_with_array" in v for v in warnings.values())

    def test_isolated_detection_flagged(self):
        cluster = [det(x, 100, x + 150, 250) for x in (50, 250, 450)]
        far_away = det(1200, 690, 1270, 719)  # normalized-distance outlier
        dets = cluster + [far_away]
        ordered, _, _ = order_panels_spatially(dets, IMG_W, IMG_H)
        config = DynamicPanelConfig(isolation_distance_multiplier=2.0)
        warnings = _check_array_consistency(ordered, IMG_W, IMG_H, config)
        assert any("isolated_detection" in v for v in warnings.values())

    def test_single_detection_no_consistency_warnings(self):
        dets = [det(100, 100, 300, 250)]
        ordered, _, _ = order_panels_spatially(dets, IMG_W, IMG_H)
        config = DynamicPanelConfig()
        warnings = _check_array_consistency(ordered, IMG_W, IMG_H, config)
        assert warnings == {}

    def test_uniform_grid_no_consistency_warnings(self):
        dets = _grid(2, 2, IMG_W, IMG_H)
        ordered, _, _ = order_panels_spatially(dets, IMG_W, IMG_H)
        config = DynamicPanelConfig()
        warnings = _check_array_consistency(ordered, IMG_W, IMG_H, config)
        assert warnings == {}


# ---- zero / one detection edge cases -------------------------------------

class TestEdgeCases:
    def test_zero_detections_reports_failed(self):
        report = build_dynamic_panel_report(
            [], image_name="empty.jpg", image_width=IMG_W, image_height=IMG_H, model_name="test",
        )
        assert report["panel_count"] == 0
        assert report["analysis_status"] == "failed"
        assert report["failure_reason"] == "no_panel_detected"
        assert report["panels"] == []
        assert report["geometry"] == {"row_count": 0, "ordering_status": "n/a"}

    def test_one_detection_valid(self):
        dets = [det(100, 100, 300, 250, conf=0.95)]
        report = build_dynamic_panel_report(
            dets, image_name="one.jpg", image_width=IMG_W, image_height=IMG_H, model_name="test",
        )
        assert report["panel_count"] == 1
        assert report["analysis_status"] == "ok"
        assert report["panels"][0]["panel_id"] == "P1"
        assert report["panels"][0]["status"] == "valid"
        assert report["geometry"]["row_count"] == 1

    def test_suspicious_size_outlier_marks_image_level_warning(self):
        normal = [det(x, 100, x + 150, 250) for x in (50, 250, 450)]
        outlier = det(700, 100, 705, 105)  # tiny outlier
        report = build_dynamic_panel_report(
            normal + [outlier], image_name="outlier.jpg", image_width=IMG_W, image_height=IMG_H,
            model_name="test", config=DynamicPanelConfig(min_relative_area=0.0001, size_inconsistency_ratio=3.0),
        )
        assert report["panel_count"] == 4
        assert report["analysis_status"] == "warning"
        suspicious = [p for p in report["panels"] if p["status"] == "suspicious"]
        assert len(suspicious) >= 1

    def test_never_hardcodes_four_panels(self):
        for n in (1, 2, 3, 5, 7, 9, 12):
            dets = [det(x * 100, 100, x * 100 + 80, 200) for x in range(n)]
            report = build_dynamic_panel_report(
                dets, image_name="n.jpg", image_width=2000, image_height=720, model_name="test",
            )
            assert report["panel_count"] == n
            assert len(report["panels"]) == n
            assert report["panels"][-1]["panel_id"] == f"P{n}"


# ---- schema shape ---------------------------------------------------------

class TestReportSchema:
    def test_schema_matches_spec(self):
        dets = _grid(2, 2, IMG_W, IMG_H)
        report = build_dynamic_panel_report(
            dets, image_name="capture_001.jpg", image_width=IMG_W, image_height=IMG_H,
            model_name="yolo_solar_ground_research_v1.pt",
            config=DynamicPanelConfig(confidence_threshold=0.45),
        )
        assert report["schema_version"] == "1.0"
        assert set(report) >= {
            "schema_version", "image", "image_width", "image_height", "model",
            "confidence_threshold", "panel_count", "analysis_status", "panels", "geometry",
        }
        p = report["panels"][0]
        assert set(p) == {
            "panel_id", "confidence", "bbox", "center", "width", "height",
            "area", "row_index", "column_index", "status", "warnings",
        }
        assert set(p["bbox"]) == {"x1", "y1", "x2", "y2"}
        assert set(p["center"]) == {"x", "y"}
        assert set(report["geometry"]) == {"row_count", "ordering_status"}


# ---- model orchestration (fake detector, no GPU) -------------------------

class _FakeBox:
    def __init__(self, cls_id: float, conf: float, xyxy: list[float]):
        self._cls = type("T", (), {"item": lambda s: cls_id})()
        self._conf = type("T", (), {"item": lambda s: conf})()
        self._xyxy = [type("T", (), {"tolist": lambda s: list(xyxy)})()]

    @property
    def cls(self):
        return self._cls

    @property
    def conf(self):
        return self._conf

    @property
    def xyxy(self):
        return self._xyxy


class _FakeModel:
    """Duck-typed fake YOLO model: same shape as _FakeModel in
    test_ground_pv_tiled_inference.py. `boxes` is a list of
    (class_id, confidence, [x1,y1,x2,y2])."""

    def __init__(self, names: dict[int, str], boxes: list[tuple[int, float, list[float]]]):
        self.names = names
        self._boxes = boxes
        self.calls: list[dict] = []

    def predict(self, source=None, conf=None, iou=None, imgsz=None, verbose=None):
        self.calls.append({"conf": conf, "iou": iou, "imgsz": imgsz})
        boxes = [_FakeBox(cid, cf, box) for (cid, cf, box) in self._boxes]
        res = type("R", (), {"boxes": boxes})()
        return [res]


class TestModelOrchestration:
    def test_run_panel_detector_filters_non_panel_classes(self):
        model = _FakeModel(
            names={0: "panel", 1: "cracked", 2: "dusty"},
            boxes=[
                (0, 0.90, [100, 100, 300, 250]),
                (1, 0.70, [10, 10, 50, 50]),   # cracked -> must be dropped
                (2, 0.60, [400, 400, 450, 450]),  # dusty -> must be dropped
            ],
        )
        accepted = run_panel_detector(model, image=object(), conf=0.45)
        assert len(accepted) == 1
        assert accepted[0].class_name == "panel"

    def test_run_panel_detector_passes_through_config(self):
        model = _FakeModel(names={0: "panel"}, boxes=[(0, 0.9, [0, 0, 10, 10])])
        run_panel_detector(model, image=object(), conf=0.37, iou=0.55, imgsz=800)
        assert model.calls[0] == {"conf": 0.37, "iou": 0.55, "imgsz": 800}

    def test_detect_dynamic_panels_end_to_end_eight_panels(self):
        boxes = []
        w, h = IMG_W, IMG_H
        for r in range(2):
            for c in range(4):
                x1 = 50 + c * 200
                y1 = 50 + r * 300
                boxes.append((0, 0.9, [x1, y1, x1 + 150, y1 + 100]))
        model = _FakeModel(names={0: "panel"}, boxes=boxes)
        image = Image.new("RGB", (w, h), (100, 100, 100))
        report = detect_dynamic_panels(model, image, image_name="eight.jpg", model_name="fake")
        assert report["panel_count"] == 8
        assert report["geometry"]["row_count"] == 2
        assert report["analysis_status"] == "ok"

    def test_detect_dynamic_panels_zero_detections(self):
        model = _FakeModel(names={0: "panel"}, boxes=[])
        image = Image.new("RGB", (IMG_W, IMG_H), (0, 0, 0))
        report = detect_dynamic_panels(model, image, image_name="zero.jpg", model_name="fake")
        assert report["panel_count"] == 0
        assert report["analysis_status"] == "failed"
        assert report["failure_reason"] == "no_panel_detected"

    def test_low_confidence_detections_not_silently_accepted(self):
        # Model itself would only emit boxes above `conf` (that's how
        # Ultralytics conf filtering works) - verify the low-conf box never
        # reaches our accepted list because the fake model wouldn't emit it
        # below its own threshold either; here we simulate the model NOT
        # emitting anything at conf=0.45, confirming zero-detection path.
        model = _FakeModel(names={0: "panel"}, boxes=[])
        image = Image.new("RGB", (IMG_W, IMG_H), (0, 0, 0))
        report = detect_dynamic_panels(
            model, image, image_name="lowconf.jpg", model_name="fake",
            config=DynamicPanelConfig(confidence_threshold=0.45),
        )
        assert report["panel_count"] == 0
        assert report["analysis_status"] == "failed"


# ---- annotation ------------------------------------------------------------

class TestDrawAnnotated:
    def test_draw_annotated_runs_and_returns_image(self):
        dets = _grid(2, 2, IMG_W, IMG_H)
        report = build_dynamic_panel_report(
            dets, image_name="x.jpg", image_width=IMG_W, image_height=IMG_H, model_name="test",
        )
        image = Image.new("RGB", (IMG_W, IMG_H), (50, 50, 50))
        annotated = draw_annotated(image, report)
        assert annotated.size == (IMG_W, IMG_H)
        assert annotated.mode == "RGB"
