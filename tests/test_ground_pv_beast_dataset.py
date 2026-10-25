"""tests/test_ground_pv_beast_dataset.py — Phase 13B beast dataset builder:
candidate scene grouping, polygon->bbox conversion, split isolation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from training.detection.ground_pv.build_beast_dataset import (
    CANDIDATE_CLASS,
    _capture_key,
    _poly_to_bbox,
    build,
)


def test_capture_key_groups_video_and_photo_events():
    assert _capture_key("20240430_133150_mp4-0007_jpg.rf.abc.jpg") == ("20240430_133150_mp4", "video", 7)
    assert _capture_key("20240430_133150_mp4-0235_jpg.rf.def.jpg") == ("20240430_133150_mp4", "video", 235)
    assert _capture_key("20240430_132725_jpg.rf.abc.jpg") == ("20240430_132725", "photo", 0)
    assert _capture_key("random_name.jpg")[1] == "photo"


def test_poly_to_bbox_tight_clamped_and_rejects_bad():
    # a triangle polygon -> its tight bbox
    bb = _poly_to_bbox([0.2, 0.2, 0.6, 0.3, 0.4, 0.7])
    assert bb == pytest.approx((0.4, 0.45, 0.4, 0.5))
    # already-bbox passthrough
    assert _poly_to_bbox([0.5, 0.5, 0.2, 0.2]) == pytest.approx((0.5, 0.5, 0.2, 0.2))
    # out-of-range coords clamped to [0,1]
    cx, cy, w, h = _poly_to_bbox([-0.1, 0.1, 0.4, 0.2, 0.3, 0.6])
    assert 0 <= cx <= 1 and 0 <= cy <= 1 and w <= 1.0 and h <= 1.0
    assert cx == pytest.approx(0.2)  # x clamped: (0.0 + 0.4)/2
    # rejects
    assert _poly_to_bbox([0.5, 0.5, 0.5, 0.5001, 0.5, 0.5]) is None      # ~zero area
    assert _poly_to_bbox([0.1, 0.2, 0.3]) is None                        # odd count
    assert _poly_to_bbox([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]) is None  # scene-spanning


def _img(p: Path, size=(64, 64)) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (0, 0, 0)).save(p)


def _lbl(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _susan(root: Path) -> Path:
    for split, n in (("train", 5), ("valid", 3), ("test", 2)):
        for i in range(n):
            _img(root / split / "images" / f"s{i}.jpg")
            _lbl(root / split / "labels" / f"s{i}.txt", "0 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.1 0.1\n")
    (root / "data.yaml").write_text("names:\n  0: panel\n  1: cracked\n  2: dusty\n", encoding="utf-8")
    return root


def _candidate(root: Path) -> Path:
    # one video sequence (10 frames) + 4 burst photos (2 timestamps x 2) + 1 empty
    for f in range(10):
        _img(root / "train" / "images" / f"20240430_133150_mp4-{f:04d}_jpg.rf.h{f}.jpg")
        _lbl(root / "train" / "labels" / f"20240430_133150_mp4-{f:04d}_jpg.rf.h{f}.txt",
             "0 0.2 0.2 0.15 0.2 0.4 0.3 0.35 0.5\n")  # polygon
    for t, k in (("132725", "a"), ("132727", "b")):
        for j in range(2):
            _img(root / "valid" / "images" / f"20240430_{t}_{j}_jpg.rf.{k}{j}.jpg")
            _lbl(root / "valid" / "labels" / f"20240430_{t}_{j}_jpg.rf.{k}{j}.txt", "0 0.5 0.5 0.3 0.3\n")
    _img(root / "test" / "images" / "20240430_140000_jpg.rf.empty.jpg")
    _lbl(root / "test" / "labels" / "20240430_140000_jpg.rf.empty.txt", "")
    (root / "data.yaml").write_text("names:\n- solarpanel\nnc: 1\n", encoding="utf-8")
    return root


def test_build_beast_dataset(tmp_path):
    susan = _susan(tmp_path / "susan")
    cand = _candidate(tmp_path / "cand")
    out = tmp_path / "beast"
    rep = build(susan, cand, out, video_stride=4, max_candidate=100, overwrite=True)

    # empty-label candidate image dropped; video subsampled every 4th (10 -> 3);
    # burst photos 1 per capture-second (2)
    assert rep.candidate_empty_dropped == 1
    assert rep.candidate_video_frames_selected == 3
    assert rep.candidate_photo_frames_selected == 2
    assert rep.candidate_selected == 5
    assert rep.polygons_converted == 5 and rep.polygons_rejected == 0

    # candidate materialised into TRAIN only
    assert (out / "candidate" / "train" / "images").is_dir()
    cand_imgs = list((out / "candidate" / "train" / "images").glob("*"))
    assert len(cand_imgs) == 5
    for lp in (out / "candidate" / "train" / "labels").glob("*.txt"):
        for ln in lp.read_text().splitlines():
            p = ln.split()
            assert len(p) == 5 and int(p[0]) == CANDIDATE_CLASS
            assert all(0.0 <= float(v) <= 1.0 for v in p[1:])

    # data yamls
    da = yaml.safe_load((out / "data_A.yaml").read_text())
    dbc = yaml.safe_load((out / "data_BC.yaml").read_text())
    assert da["val"].endswith("valid/images") or da["val"] == "valid/images"
    assert isinstance(dbc["train"], list) and len(dbc["train"]) == 2       # susan + candidate
    assert "valid" in dbc["val"] and "test" in dbc["test"]                 # Susan valid/test only

    # manifest: every candidate row is one family, train split, grade C
    man = json.loads((out / "beast_dataset_manifest.json").read_text())
    fams = {r["family_id"] for r in man["manifest_rows"]}
    assert fams == {"cand_ubwtz_scene1"}
    assert all(r["split"] == "train" and r["provenance_grade"] == "C" for r in man["manifest_rows"])
    assert man["leakage_controls"]["candidate_split"].startswith("train only")

    # Susan untouched
    assert len(list((susan / "train" / "images").glob("*"))) == 5


def test_build_refuses_existing_without_overwrite(tmp_path):
    susan = _susan(tmp_path / "susan")
    cand = _candidate(tmp_path / "cand")
    out = tmp_path / "beast"
    build(susan, cand, out, overwrite=True)
    with pytest.raises(FileExistsError):
        build(susan, cand, out, overwrite=False)
