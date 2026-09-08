"""tests/test_ground_pv_dataset_scale_analysis.py — the Phase 11 training-data
failure analysis for the ground panel detector (scale / large-panel / wide-array
coverage). Label-only, synthetic fixture."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.analyze_ground_pv_dataset_scale import _area_bucket, analyze


def _img(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (0, 0, 0)).save(p)


def _lbl(p: Path, lines: list[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def test_area_bucket_thresholds():
    assert _area_bucket(0.10) == "<25%"
    assert _area_bucket(0.25) == "25-50%"
    assert _area_bucket(0.60) == "50-75%"
    assert _area_bucket(0.90) == ">75%"


def _make(tmp_path: Path) -> Path:
    d = tmp_path / "yolo"
    # train: one image-4-like (one huge panel), one image-6-like (5 small panels),
    #        one ordinary (small panel), one boundary-touching panel
    _img(d / "train/images/big.jpg")
    _lbl(d / "train/labels/big.txt", ["0 0.5 0.5 0.95 0.95"])                 # >75%, 1 panel
    _img(d / "train/images/wide.jpg")
    _lbl(d / "train/labels/wide.txt", [f"0 {0.1 + i * 0.18:.3f} 0.5 0.12 0.12" for i in range(5)])  # 5 panels, all tiny
    _img(d / "train/images/small.jpg")
    _lbl(d / "train/labels/small.txt", ["0 0.5 0.5 0.2 0.2"])                  # <25%
    _img(d / "train/images/edge.jpg")
    _lbl(d / "train/labels/edge.txt", ["0 0.05 0.5 0.12 0.6"])                 # touches left border
    # valid: one panel
    _img(d / "valid/images/v.jpg")
    _lbl(d / "valid/labels/v.txt", ["0 0.5 0.5 0.3 0.3"])
    # test: one crack-only image (0 panels)
    _img(d / "test/images/t.jpg")
    _lbl(d / "test/labels/t.txt", ["1 0.5 0.5 0.1 0.1"])
    return d


def test_analyze_counts_large_panels_wide_arrays_and_edges(tmp_path):
    res = analyze(_make(tmp_path))
    t = res["totals"]
    assert t["images"] == 6
    assert t["panel_boxes"] == 1 + 5 + 1 + 1 + 1        # big + wide(5) + small + edge + valid
    assert t["panel_area_buckets"][">75%"] == 1
    assert t["panel_area_buckets"]["<25%"] >= 6          # wide's 5 + small + edge etc.
    assert t["image4_like_big_foreground_images"] == 1   # only 'big'
    assert t["image6_like_wide_array_images"] == 1       # only 'wide'
    assert t["boundary_touching_panel_boxes"] >= 1       # 'edge' (plus possibly 'big')
    # the 0-panel crack-only test image is counted
    assert res["splits"]["test"]["images"] == 1
    assert res["splits"]["test"]["images_with_panel"] == 0
    # example lists are populated
    assert res["image4_like_examples_top"][0][1:] == ("train", "big.jpg")
    assert res["image6_like_examples_top"][0][1:] == ("train", "wide.jpg")
