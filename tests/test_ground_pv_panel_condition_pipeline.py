"""tests/test_ground_pv_panel_condition_pipeline.py — Phase 15A end-to-end
software pipeline: full image -> dynamic panel detection -> panel crops ->
4-class condition inference -> stable per-panel schema.

Critical requirement under test: the number of condition inferences must
ALWAYS equal the number of accepted panel detections — for 4, 6, 8, or any
other N — never hard-coded to 4. Uses the same duck-typed fake-model
pattern as test_ground_pv_dynamic_panel_detection.py (no GPU / Ultralytics
needed).
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from training.detection.ground_pv.condition_taxonomy import CONDITION_4CLASS_ORDER
from training.detection.ground_pv.dynamic_panel_detection import DynamicPanelConfig
from training.detection.ground_pv.panel_condition_pipeline import (
    ConditionInferenceConfig,
    check_crop_quality,
    compute_padded_crop_box,
    run_condition_on_crop,
    run_panel_condition_pipeline,
)

IMG_W, IMG_H = 1280, 720
_COND_NAMES = {0: "CLEAN", 1: "DUSTY", 2: "BIRD_DROP", 3: "PHYSICAL_DAMAGE"}
_PANEL_NAMES = {0: "panel"}


# ---------------------------------------------------------------------------
# Fake models (duck-typed, matches test_ground_pv_dynamic_panel_detection.py)
# ---------------------------------------------------------------------------


class _FakeBox:
    def __init__(self, cls_id, conf, xyxy):
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


class _FakePanelModel:
    def __init__(self, names, boxes):
        self.names = names
        self._boxes = boxes

    def predict(self, source=None, conf=None, iou=None, imgsz=None, verbose=None):
        boxes = [_FakeBox(cid, cf, box) for (cid, cf, box) in self._boxes]
        return [type("R", (), {"boxes": boxes})()]


class _FakeConditionModel:
    """Returns a DIFFERENT scripted response per call (one per panel crop,
    consumed in order) so tests can assign distinct conditions to distinct
    panels. `responses` is a list of box-lists; each box is
    (class_id, confidence, [x1,y1,x2,y2]) in CROP-local coordinates."""

    def __init__(self, names, responses):
        self.names = names
        self._responses = list(responses)
        self._call_index = 0
        self.calls: list[dict] = []

    def predict(self, source=None, conf=None, iou=None, imgsz=None, verbose=None):
        self.calls.append({"conf": conf, "iou": iou, "imgsz": imgsz})
        idx = min(self._call_index, len(self._responses) - 1) if self._responses else 0
        boxes_spec = self._responses[idx] if self._responses else []
        self._call_index += 1
        boxes = [_FakeBox(cid, cf, box) for (cid, cf, box) in boxes_spec]
        return [type("R", (), {"boxes": boxes})()]


def _panel_grid(rows, cols, cell_w=150, cell_h=100, gap=50):
    boxes = []
    for r in range(rows):
        for c in range(cols):
            x1 = 50 + c * (cell_w + gap)
            y1 = 50 + r * (cell_h + gap)
            boxes.append((0, 0.9, [x1, y1, x1 + cell_w, y1 + cell_h]))
    return boxes


# ---------------------------------------------------------------------------
# Crop geometry
# ---------------------------------------------------------------------------


class TestCropGeometry:
    def test_padded_crop_box_expands_and_clips(self):
        box = compute_padded_crop_box([100, 100, 200, 200], IMG_W, IMG_H, pad=0.1)
        x1, y1, x2, y2 = box
        assert x1 < 100 and y1 < 100 and x2 > 200 and y2 > 200

    def test_padded_crop_box_clips_at_image_edges(self):
        box = compute_padded_crop_box([0, 0, 50, 50], IMG_W, IMG_H, pad=0.5)
        x1, y1, x2, y2 = box
        assert x1 == 0 and y1 == 0  # never negative


class TestCropQualityDiagnostics:
    def test_normal_crop_no_warnings(self):
        # textured (not flat-color, or the blur check would legitimately fire)
        rng = np.random.default_rng(7)
        arr = rng.integers(0, 255, (150, 200, 3), dtype=np.uint8)
        crop = Image.fromarray(arr, "RGB")
        warnings = check_crop_quality(crop, (100, 100, 300, 250), IMG_W, IMG_H, ConditionInferenceConfig())
        assert warnings == []

    def test_tiny_crop_flagged(self):
        crop = Image.new("RGB", (5, 5), (128, 128, 128))
        warnings = check_crop_quality(crop, (100, 100, 105, 105), IMG_W, IMG_H, ConditionInferenceConfig())
        assert "panel_crop_too_small" in warnings

    def test_extreme_aspect_ratio_flagged(self):
        crop = Image.new("RGB", (300, 10), (128, 128, 128))
        warnings = check_crop_quality(crop, (100, 100, 400, 110), IMG_W, IMG_H, ConditionInferenceConfig())
        assert "panel_crop_extreme_aspect_ratio" in warnings

    def test_truncated_crop_flagged(self):
        # intended a much bigger box than what the crop actually is
        crop = Image.new("RGB", (50, 50), (128, 128, 128))
        warnings = check_crop_quality(crop, (0, 0, 500, 500), IMG_W, IMG_H, ConditionInferenceConfig())
        assert "panel_crop_truncated" in warnings

    def test_blurry_flat_color_crop_flagged(self):
        # a perfectly flat crop has zero Laplacian variance -> "blurry"
        crop = Image.new("RGB", (100, 100), (128, 128, 128))
        warnings = check_crop_quality(crop, (0, 0, 100, 100), IMG_W, IMG_H, ConditionInferenceConfig())
        assert "panel_crop_possibly_blurry" in warnings

    def test_textured_crop_not_flagged_as_blurry(self):
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        crop = Image.fromarray(arr, "RGB")
        warnings = check_crop_quality(crop, (0, 0, 100, 100), IMG_W, IMG_H, ConditionInferenceConfig())
        assert "panel_crop_possibly_blurry" not in warnings

    def test_diagnostics_never_raise_never_reject(self):
        """Diagnostics only - never an exception, never a rejection signal
        beyond the warnings list itself."""
        crop = Image.new("RGB", (1, 1), (0, 0, 0))
        warnings = check_crop_quality(crop, (0, 0, 1, 1), IMG_W, IMG_H, ConditionInferenceConfig())
        assert isinstance(warnings, list)  # did not raise


# ---------------------------------------------------------------------------
# Condition inference on one crop
# ---------------------------------------------------------------------------


class TestRunConditionOnCrop:
    def test_refuses_non_active_taxonomy(self):
        model = _FakeConditionModel(names={0: "CLEAN", 1: "DUSTY", 2: "BIRD_DROP", 3: "PHYSICAL_DAMAGE", 4: "ELECTRICAL_DAMAGE"},
                                    responses=[[]])
        crop = Image.new("RGB", (100, 100), (0, 0, 0))
        with pytest.raises(ValueError):
            run_condition_on_crop(model, crop)

    def test_predicted_status_above_threshold(self):
        model = _FakeConditionModel(names=_COND_NAMES, responses=[[(1, 0.85, [10, 10, 90, 90])]])
        crop = Image.new("RGB", (100, 100), (0, 0, 0))
        result = run_condition_on_crop(model, crop, ConditionInferenceConfig(predicted_threshold=0.45))
        assert result["status"] == "predicted"
        assert result["label"] == "DUSTY"
        assert result["confidence"] == pytest.approx(0.85)

    def test_uncertain_status_below_threshold(self):
        model = _FakeConditionModel(names=_COND_NAMES, responses=[[(1, 0.30, [10, 10, 90, 90])]])
        crop = Image.new("RGB", (100, 100), (0, 0, 0))
        result = run_condition_on_crop(model, crop, ConditionInferenceConfig(detection_floor=0.25, predicted_threshold=0.45))
        assert result["status"] == "uncertain"
        assert result["label"] == "DUSTY"

    def test_no_prediction_status_zero_detections(self):
        model = _FakeConditionModel(names=_COND_NAMES, responses=[[]])
        crop = Image.new("RGB", (100, 100), (0, 0, 0))
        result = run_condition_on_crop(model, crop)
        assert result["status"] == "no_prediction"
        assert result["label"] is None
        assert result["confidence"] is None

    def test_no_prediction_never_becomes_clean(self):
        model = _FakeConditionModel(names=_COND_NAMES, responses=[[]])
        crop = Image.new("RGB", (100, 100), (0, 0, 0))
        result = run_condition_on_crop(model, crop)
        assert result["label"] != "CLEAN"
        assert result["status"] != "predicted"

    def test_multiple_detections_preserved_as_list(self):
        model = _FakeConditionModel(names=_COND_NAMES, responses=[[
            (1, 0.80, [10, 10, 50, 50]), (2, 0.60, [40, 40, 90, 90]),
        ]])
        crop = Image.new("RGB", (100, 100), (0, 0, 0))
        result = run_condition_on_crop(model, crop)
        assert len(result["detections"]) == 2
        labels = {d.label for d in result["detections"]}
        assert labels == {"DUSTY", "BIRD_DROP"}

    def test_top_detection_is_highest_confidence(self):
        model = _FakeConditionModel(names=_COND_NAMES, responses=[[
            (1, 0.40, [10, 10, 50, 50]), (3, 0.75, [40, 40, 90, 90]),
        ]])
        crop = Image.new("RGB", (100, 100), (0, 0, 0))
        result = run_condition_on_crop(model, crop, ConditionInferenceConfig(predicted_threshold=0.45))
        assert result["label"] == "PHYSICAL_DAMAGE"
        assert result["status"] == "predicted"


# ---------------------------------------------------------------------------
# Full pipeline: panel_count == number of condition inferences
# ---------------------------------------------------------------------------


class TestFullPipelinePanelCountMatchesConditionInputs:
    def _run(self, n_panels_grid, condition_responses):
        rows, cols = n_panels_grid
        panel_model = _FakePanelModel(names=_PANEL_NAMES, boxes=_panel_grid(rows, cols))
        condition_model = _FakeConditionModel(names=_COND_NAMES, responses=condition_responses)
        image = Image.new("RGB", (IMG_W, IMG_H), (100, 100, 100))
        report = run_panel_condition_pipeline(
            panel_model, condition_model, image, image_name="test.jpg",
            panel_model_name="panel.pt", condition_model_name="cond.pt",
        )
        return report, condition_model

    def test_four_panels_2x2(self):
        report, cmodel = self._run((2, 2), [[(0, 0.9, [10, 10, 90, 90])]])
        assert report["panel_count"] == 4
        assert len(report["panels"]) == 4
        assert len(cmodel.calls) == 4  # exactly one condition call per panel

    def test_six_panels_2x3(self):
        report, cmodel = self._run((2, 3), [[(1, 0.9, [10, 10, 90, 90])]])
        assert report["panel_count"] == 6
        assert len(report["panels"]) == 6
        assert len(cmodel.calls) == 6

    def test_eight_panels_2x4(self):
        report, cmodel = self._run((2, 4), [[(2, 0.9, [10, 10, 90, 90])]])
        assert report["panel_count"] == 8
        assert len(report["panels"]) == 8
        assert len(cmodel.calls) == 8

    def test_arbitrary_n_panels(self):
        for n in (1, 3, 5, 7, 9, 12):
            panel_model = _FakePanelModel(names=_PANEL_NAMES, boxes=_panel_grid(1, n, cell_w=80, gap=20))
            condition_model = _FakeConditionModel(names=_COND_NAMES, responses=[[(0, 0.9, [10, 10, 90, 90])]])
            image = Image.new("RGB", (2000, 400), (0, 0, 0))
            report = run_panel_condition_pipeline(
                panel_model, condition_model, image, image_name="n.jpg",
                panel_model_name="p.pt", condition_model_name="c.pt",
            )
            assert report["panel_count"] == n, f"failed for n={n}"
            assert len(report["panels"]) == n
            assert len(condition_model.calls) == n

    def test_never_hardcodes_four(self):
        """The pipeline must not special-case N=4 in any way distinguishable
        from other N."""
        report4, _ = self._run((2, 2), [[(0, 0.9, [10, 10, 90, 90])]])
        report6, _ = self._run((2, 3), [[(0, 0.9, [10, 10, 90, 90])]])
        # same code path, same schema shape, differing only in count/content
        assert set(report4.keys()) == set(report6.keys())
        assert report4["panel_count"] != report6["panel_count"]


class TestZeroPanelDetections:
    def test_zero_panels_zero_condition_calls(self):
        panel_model = _FakePanelModel(names=_PANEL_NAMES, boxes=[])
        condition_model = _FakeConditionModel(names=_COND_NAMES, responses=[])
        image = Image.new("RGB", (IMG_W, IMG_H), (0, 0, 0))
        report = run_panel_condition_pipeline(
            panel_model, condition_model, image, image_name="zero.jpg",
            panel_model_name="p.pt", condition_model_name="c.pt",
        )
        assert report["panel_count"] == 0
        assert report["analysis_status"] == "failed"
        assert report["panels"] == []
        assert len(condition_model.calls) == 0  # condition model never invoked

    def test_zero_panels_never_reinterpreted_as_all_clean(self):
        panel_model = _FakePanelModel(names=_PANEL_NAMES, boxes=[])
        condition_model = _FakeConditionModel(names=_COND_NAMES, responses=[])
        image = Image.new("RGB", (IMG_W, IMG_H), (0, 0, 0))
        report = run_panel_condition_pipeline(
            panel_model, condition_model, image, image_name="zero.jpg",
            panel_model_name="p.pt", condition_model_name="c.pt",
        )
        assert "failure_reason" in report
        assert report["failure_reason"] == "no_panel_detected"


class TestOutputSchema:
    def test_schema_matches_spec_shape(self):
        panel_model = _FakePanelModel(names=_PANEL_NAMES, boxes=_panel_grid(2, 3))
        condition_model = _FakeConditionModel(names=_COND_NAMES, responses=[[(1, 0.78, [10, 10, 90, 90])]])
        image = Image.new("RGB", (IMG_W, IMG_H), (0, 0, 0))
        report = run_panel_condition_pipeline(
            panel_model, condition_model, image, image_name="capture_001.jpg",
            panel_model_name="panel.pt", condition_model_name="cond.pt",
        )
        assert report["schema_version"] == "1.0"
        assert set(report) >= {
            "schema_version", "image", "image_width", "image_height",
            "panel_model", "condition_model", "condition_taxonomy",
            "panel_count", "analysis_status", "panels",
        }
        assert report["condition_taxonomy"] == list(CONDITION_4CLASS_ORDER)

        p = report["panels"][0]
        assert set(p) >= {"panel_id", "detection", "condition", "condition_detections", "crop"}
        assert set(p["detection"]) >= {"confidence", "bbox"}
        assert set(p["condition"]) == {"status", "label", "confidence"}
        assert p["panel_id"] == "P1"

    def test_dynamic_panel_ids_preserved_through_pipeline(self):
        panel_model = _FakePanelModel(names=_PANEL_NAMES, boxes=_panel_grid(2, 4))
        condition_model = _FakeConditionModel(names=_COND_NAMES, responses=[[(0, 0.9, [10, 10, 90, 90])]])
        image = Image.new("RGB", (IMG_W, IMG_H), (0, 0, 0))
        report = run_panel_condition_pipeline(
            panel_model, condition_model, image, image_name="eight.jpg",
            panel_model_name="p.pt", condition_model_name="c.pt",
        )
        panel_ids = [p["panel_id"] for p in report["panels"]]
        assert panel_ids == [f"P{i+1}" for i in range(8)]

    def test_detection_confidence_separate_from_condition_confidence(self):
        panel_model = _FakePanelModel(names=_PANEL_NAMES, boxes=[(0, 0.91, [50, 50, 250, 250])])
        condition_model = _FakeConditionModel(names=_COND_NAMES, responses=[[(1, 0.77, [10, 10, 90, 90])]])
        image = Image.new("RGB", (IMG_W, IMG_H), (0, 0, 0))
        report = run_panel_condition_pipeline(
            panel_model, condition_model, image, image_name="one.jpg",
            panel_model_name="p.pt", condition_model_name="c.pt",
            condition_config=ConditionInferenceConfig(predicted_threshold=0.45),
        )
        panel = report["panels"][0]
        assert panel["detection"]["confidence"] == pytest.approx(0.91)
        assert panel["condition"]["confidence"] == pytest.approx(0.77)
        assert panel["detection"]["confidence"] != panel["condition"]["confidence"]

    def test_condition_status_field_present_and_valid(self):
        panel_model = _FakePanelModel(names=_PANEL_NAMES, boxes=[(0, 0.9, [50, 50, 250, 250])])
        condition_model = _FakeConditionModel(names=_COND_NAMES, responses=[[]])  # no_prediction case
        image = Image.new("RGB", (IMG_W, IMG_H), (0, 0, 0))
        report = run_panel_condition_pipeline(
            panel_model, condition_model, image, image_name="one.jpg",
            panel_model_name="p.pt", condition_model_name="c.pt",
        )
        assert report["panels"][0]["condition"]["status"] in ("predicted", "uncertain", "no_prediction")
        assert report["panels"][0]["condition"]["status"] == "no_prediction"

    def test_crop_warnings_surfaced_in_output(self):
        # a tiny panel box -> tiny crop -> warnings
        panel_model = _FakePanelModel(names=_PANEL_NAMES, boxes=[(0, 0.9, [50, 50, 55, 55])])
        condition_model = _FakeConditionModel(names=_COND_NAMES, responses=[[]])
        image = Image.new("RGB", (IMG_W, IMG_H), (0, 0, 0))
        report = run_panel_condition_pipeline(
            panel_model, condition_model, image, image_name="tiny.jpg",
            panel_model_name="p.pt", condition_model_name="c.pt",
        )
        assert "warnings" in report["panels"][0]["crop"]
        assert isinstance(report["panels"][0]["crop"]["warnings"], list)
        assert len(report["panels"][0]["crop"]["warnings"]) > 0


class TestMixedConditionStatusesAcrossPanels:
    def test_different_panels_get_independent_condition_results(self):
        panel_model = _FakePanelModel(names=_PANEL_NAMES, boxes=_panel_grid(1, 3, cell_w=200, gap=50))
        condition_model = _FakeConditionModel(names=_COND_NAMES, responses=[
            [(0, 0.90, [10, 10, 90, 90])],   # P1: CLEAN predicted
            [(1, 0.30, [10, 10, 90, 90])],   # P2: DUSTY uncertain
            [],                                # P3: no_prediction
        ])
        image = Image.new("RGB", (2000, 400), (0, 0, 0))
        report = run_panel_condition_pipeline(
            panel_model, condition_model, image, image_name="mixed.jpg",
            panel_model_name="p.pt", condition_model_name="c.pt",
            condition_config=ConditionInferenceConfig(predicted_threshold=0.45),
        )
        statuses = [p["condition"]["status"] for p in report["panels"]]
        assert statuses == ["predicted", "uncertain", "no_prediction"]
        assert report["panels"][0]["condition"]["label"] == "CLEAN"
        assert report["panels"][2]["condition"]["label"] is None
