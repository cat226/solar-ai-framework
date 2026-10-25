"""Tests for recovering owner burned-in boxes by pristine-vs-marked diff
(synthetic images only)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from training.detection.ground_pv.owner_mark_boxes import extract_boxes


def _scene(size=(400, 300)) -> Image.Image:
    rng = np.random.default_rng(0)
    arr = rng.integers(90, 200, (size[1], size[0], 3), dtype=np.uint8)
    arr[..., 2] = np.clip(arr[..., 2] + 40, 0, 255)  # bluish, never pen-black / pen-red
    return Image.fromarray(arr)


def test_black_frame_is_recovered_and_labelled():
    p = _scene(); m = p.copy()
    ImageDraw.Draw(m).rectangle((100, 80, 220, 230), outline=(0, 0, 0), width=4)
    (b,) = extract_boxes(p, m)
    assert (b["x1"], b["y1"], b["x2"], b["y2"]) == (100, 80, 220, 230) and b["ink"] == "black"


def test_red_frame_and_dark_red_frame_are_red():
    for colour in ((220, 20, 20), (130, 10, 35)):
        p = _scene(); m = p.copy()
        ImageDraw.Draw(m).rectangle((50, 50, 150, 200), outline=colour, width=4)
        (b,) = extract_boxes(p, m)
        assert b["ink"] == "red"


def test_broken_frame_fragments_merge_into_one_box():
    p = _scene(); m = p.copy(); d = ImageDraw.Draw(m)
    # frame with a 9px gap in the top edge and a 9px gap in the left edge
    d.line((100, 80, 150, 80), fill=(0, 0, 0), width=4); d.line((160, 80, 220, 80), fill=(0, 0, 0), width=4)
    d.line((220, 80, 220, 230), fill=(0, 0, 0), width=4); d.line((100, 230, 220, 230), fill=(0, 0, 0), width=4)
    d.line((100, 80, 100, 140), fill=(0, 0, 0), width=4); d.line((100, 150, 100, 230), fill=(0, 0, 0), width=4)
    assert len(extract_boxes(p, m)) == 1


def test_label_text_blob_is_not_a_box_and_does_not_stretch_one():
    p = _scene(); m = p.copy(); d = ImageDraw.Draw(m)
    d.rectangle((100, 80, 220, 230), outline=(0, 0, 0), width=4)
    d.rectangle((110, 60, 200, 76), fill=(0, 0, 0))  # dense "text" blob touching above the frame
    (b,) = extract_boxes(p, m)
    assert b["y1"] == 80


def test_unmarked_photo_has_no_boxes():
    p = _scene()
    assert extract_boxes(p, p.copy()) == []
