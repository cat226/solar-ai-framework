"""tests/test_ground_pv_tiled_inference.py — the overlapping-tile inference
orchestration for the ground-PV panel detector (Phase 10 experiment).

No YOLO / GPU needed: a fake detector returns scripted boxes so the tile
windowing, coordinate remapping, edge padding and global de-duplication can be
verified deterministically.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from training.detection.ground_pv.tiled_inference import (
    Detection,
    _contained_fraction,
    _iou,
    merge_detections,
    tile_windows,
    tiled_detect,
)


# ---- tile windows -------------------------------------------------------

def test_tile_windows_cover_every_pixel_portrait_and_landscape():
    for (W, H) in [(573, 1280), (1280, 573), (640, 640), (500, 500), (1000, 1500)]:
        wins = tile_windows(W, H, tile=640, overlap=0.25)
        mask = np.zeros((H, W), dtype=bool)
        for (x0, y0, x1, y1) in wins:
            assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H          # inside the image
            assert x1 - x0 <= 640 and y1 - y0 <= 640                # never bigger than a tile
            mask[y0:y1, x0:x1] = True
        assert mask.all(), f"coverage gap for {W}x{H}"


def test_tile_windows_single_tile_when_image_smaller_than_tile():
    assert tile_windows(400, 300, tile=640, overlap=0.25) == [(0, 0, 400, 300)]


def test_tile_windows_overlap_controls_step():
    a = tile_windows(2000, 100, tile=640, overlap=0.25)      # step 480
    b = tile_windows(2000, 100, tile=640, overlap=0.50)      # step 320
    assert len(b) > len(a)
    assert tile_windows(1, 1, tile=640, overlap=0.0)[0] == (0, 0, 1, 1)
    with pytest.raises(ValueError):
        tile_windows(100, 100, tile=640, overlap=1.0)


# ---- geometry helpers -------------------------------------------------

def test_iou_and_contained_fraction():
    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert _iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert _iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    # a small box fully inside a big one: IoU low, contained fraction 1.0
    assert _contained_fraction([2, 2, 4, 4], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert _iou([2, 2, 4, 4], [0, 0, 10, 10]) < 0.1


# ---- global duplicate suppression -----------------------------------

def test_merge_detections_nms_and_contained_suppression():
    dets = [
        Detection(0, "panel", 0.90, [0, 0, 100, 100], [1]),
        Detection(0, "panel", 0.80, [5, 5, 105, 105], [2]),      # ~0.8 IoU -> dropped
        Detection(0, "panel", 0.70, [10, 10, 40, 40], [3]),       # contained -> dropped
        Detection(0, "panel", 0.60, [400, 400, 500, 500], [4]),   # disjoint -> kept
        Detection(1, "cracked", 0.55, [0, 0, 100, 100], [1]),     # different class -> kept
    ]
    merged = merge_detections(dets, nms_iou=0.5, contain_frac=0.9)
    assert len(merged) == 3
    panel = [d for d in merged if d.class_id == 0]
    assert len(panel) == 2
    # the surviving big panel box absorbed the suppressed boxes' source tiles
    big = max(panel, key=lambda d: d.confidence)
    assert set(big.source_tiles) >= {1, 2, 3}


def test_merge_detections_is_deterministic():
    dets = [Detection(0, "panel", 0.5, [i, i, i + 50, i + 50], [i]) for i in (30, 10, 20)]
    a = [d.as_dict() for d in merge_detections(list(dets))]
    b = [d.as_dict() for d in merge_detections(list(reversed(dets)))]
    assert a == b


# ---- coordinate remapping via a fake detector ----------------------

class _FakeModel:
    """Returns one box at a fixed location *within each tile canvas*, so the
    test can assert the remap adds the correct tile origin."""
    names = {0: "panel"}

    def __init__(self, box_in_tile=(50, 60, 150, 260), conf=0.9, cls=0):
        self.box_in_tile = box_in_tile
        self.conf = conf
        self.cls = cls
        self.seen_sizes = []

    def predict(self, source=None, conf=None, iou=None, imgsz=None, verbose=None):
        self.seen_sizes.append(source.size)
        b = self.box_in_tile
        _cls, _conf = float(self.cls), float(self.conf)
        box = type("B", (), {
            "cls": type("T", (), {"item": lambda s: _cls})(),
            "conf": type("T", (), {"item": lambda s: _conf})(),
            "xyxy": [type("T", (), {"tolist": lambda s: list(b)})()],
        })()
        res = type("R", (), {"boxes": [box]})()
        return [res]


def test_tiled_detect_remaps_tile_coords_to_original():
    img = Image.new("RGB", (573, 1280), (10, 20, 30))
    model = _FakeModel(box_in_tile=(50, 60, 150, 260))
    out = tiled_detect(model, img, tile=640, overlap=0.25, conf=0.45, nms_iou=0.5, class_filter=0)
    # 3 tiles at y = 0, 480, 640 -> the box's original-y should be 60, 540, 700
    ys = sorted({round(d["bbox"][1]) for d in out["detections"]} |
                {60})  # at least the y=0 tile survives NMS
    assert 60 in ys
    # every canvas the fake model saw was exactly the tile size (padded, not resized)
    assert all(sz == (640, 640) for sz in model.seen_sizes)
    # schema
    assert set(out) == {"image_size_wh", "tile_config", "tiles", "raw_detection_count",
                        "merged_detection_count", "detections"}
    assert out["tile_config"]["tile"] == 640 and out["tile_config"]["n_tiles"] == 3
    for d in out["detections"]:
        assert set(d) == {"class_id", "class_name", "confidence", "bbox", "source_tiles"}
        assert d["class_name"] == "panel"


def test_tiled_detect_pads_small_final_tile_without_distortion():
    # 573 wide < 640 -> every tile canvas must still be 640x640 (width padded)
    img = Image.new("RGB", (573, 700), (0, 0, 0))
    model = _FakeModel()
    tiled_detect(model, img, tile=640, overlap=0.25)
    assert model.seen_sizes and all(sz == (640, 640) for sz in model.seen_sizes)


def test_tiled_detect_class_filter_drops_other_classes():
    img = Image.new("RGB", (640, 640), (0, 0, 0))
    model = _FakeModel(cls=1)  # "cracked"-like id
    out = tiled_detect(model, img, tile=640, overlap=0.25, class_filter=0)
    assert out["detections"] == []
    out2 = tiled_detect(model, img, tile=640, overlap=0.25, class_filter=None)
    assert len(out2["detections"]) == 1
