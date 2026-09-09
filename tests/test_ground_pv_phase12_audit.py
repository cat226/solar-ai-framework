"""tests/test_ground_pv_phase12_audit.py — the Phase 12 dataset-acquisition
audit helpers (domain stats + internal-scene-diversity), synthetic fixture."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.audit_phase12_candidates import (
    _aspect_bucket,
    _internal_scene_diversity,
    domain_stats,
)
from training.detection.ground_pv.image_fingerprint import fingerprint_tree


def _img(path: Path, size, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size)
    w, h = size
    for x in range(w):
        for y in range(h):
            im.putpixel((x, y), ((x * 5 + seed) % 256, (y * 3 + seed) % 256, (x + y + seed) % 256))
    im.save(path)


def _lbl(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_aspect_bucket():
    assert _aspect_bucket(573, 1280) == "portrait"
    assert _aspect_bucket(1280, 573) == "landscape"
    assert _aspect_bucket(640, 640) == "square"


def test_domain_stats_bbox_and_polygon(tmp_path):
    d = tmp_path / "cand"
    _img(d / "train/images/a.jpg", (573, 1280), 1)
    _lbl(d / "train/labels/a.txt", "0 0.5 0.5 1.0 1.0\n")            # bbox, large, boundary-touching
    _img(d / "train/images/b.jpg", (1280, 573), 2)
    _lbl(d / "train/labels/b.txt", "0 0.2 0.2 0.1 0.1 0.4 0.2 0.4 0.5 0.2 0.5\n")  # polygon -> small bbox
    _img(d / "train/images/c.jpg", (640, 640), 3)
    _lbl(d / "train/labels/c.txt", "")                                # no boxes

    st = domain_stats(d)
    assert st["images"] == 3
    assert st["aspect"] == {"portrait": 1, "landscape": 1, "square": 1}
    assert st["panel_boxes"] == 2
    assert st["polygon_label_images"] == 1
    assert st["boxes_per_image"]["images_with_zero"] == 1
    assert st["box_area_fraction"]["gt_0_5"] == 1                     # the 0.9x0.9 box
    assert st["boundary_touching_boxes"] == 1


def test_internal_scene_diversity_detects_duplicates(tmp_path):
    d = tmp_path / "dup"
    # 3 identical images + 2 distinct
    for i in range(3):
        _img(d / f"images/same{i}.jpg", (64, 64), 0)
    _img(d / "images/x.jpg", (64, 64), 50)
    _img(d / "images/y.jpg", (64, 64), 120)
    fp = fingerprint_tree(d, label="dup")
    div = _internal_scene_diversity(fp)
    assert div["images"] == 5
    assert div["distinct_scene_clusters"] == 3          # {same*}, {x}, {y}
    assert div["redundancy_fraction"] > 0.3
