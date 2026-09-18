"""Phase 19 tools: SAM-proposal filtering, review-decision -> YOLO label building,
and the scene-disjoint rooftop split."""
import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


render = _load("render_panel_proposal_review")
build_labels = _load("build_panel_labels_from_review")
build_ds = _load("build_rooftop_panel_dataset")


def _cand(bbox, area=0.1, solidity=0.95, hsv=(110.0, 120.0, 100.0)):
    return {"bbox_norm": bbox, "area_frac": area, "solidity": solidity, "mean_hsv": list(hsv),
            "polygon_norm": [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]}


def test_filter_drops_non_blue_tiny_and_low_solidity():
    cands = [
        _cand([0.0, 0.0, 0.4, 0.4]),                                   # good
        _cand([0.5, 0.5, 0.9, 0.9], hsv=(20.0, 120.0, 100.0)),         # not blue (roof)
        _cand([0.1, 0.1, 0.12, 0.12], area=0.0005),                    # too small
        _cand([0.6, 0.0, 0.9, 0.3], solidity=0.5),                     # ragged mask
    ]
    out = render.filter_candidates(cands)
    assert len(out) == 1 and out[0]["id"] == 0


def test_filter_dedups_near_identical_masks_and_relaxed_saturation():
    a = _cand([0.0, 0.0, 0.4, 0.4], area=0.16)
    b = _cand([0.005, 0.005, 0.4, 0.4], area=0.158)
    assert len(render.filter_candidates([a, b])) == 1
    glare = _cand([0.5, 0.5, 0.9, 0.9], hsv=(105.0, 10.0, 200.0))
    assert render.filter_candidates([glare]) == []
    assert len(render.filter_candidates([glare], min_sat=6.0)) == 1


def test_build_labels_writes_yolo_and_never_marks_human_verified(tmp_path):
    filt = tmp_path / "f"
    filt.mkdir()
    (filt / "imgA.filtered.json").write_text(json.dumps({"candidates": [dict(_cand([0.0, 0.0, 0.5, 0.5]), id=0)]}))
    counts, log = build_labels.build({"imgA": {"keep": [0], "add": [[0.5, 0.5, 1.0, 1.0]]}}, filt, tmp_path / "ann")
    assert counts == {"imgA": 2} and log == {}
    lines = (tmp_path / "ann" / "imgA.txt").read_text().splitlines()
    assert lines[0].startswith("0 0.250000 0.250000 0.500000 0.500000")
    assert build_labels.STATUS == "annotated_ai_assisted_unverified"


def test_build_labels_rejects_unknown_id_and_degenerate_box(tmp_path):
    filt = tmp_path / "f"
    filt.mkdir()
    (filt / "imgA.filtered.json").write_text(json.dumps({"candidates": []}))
    with pytest.raises(SystemExit):
        build_labels.build({"imgA": {"keep": [3]}}, filt, tmp_path / "a1")
    with pytest.raises(SystemExit):
        build_labels.build({"imgA": {"add": [[0.1, 0.1, 0.105, 0.5]]}}, filt, tmp_path / "a2")


def test_scene_split_is_disjoint_and_covers_all_scenes():
    scenes = {f"scene{i:04d}" for i in range(9)} | {"scene0015"}
    by_split = {}
    for s in scenes:
        by_split.setdefault(build_ds.split_of(s), set()).add(s)
    assert set(by_split) == {"train", "valid", "test"}
    assert sum(len(v) for v in by_split.values()) == len(scenes)  # no scene in two splits
    assert not (by_split["test"] & by_split["train"])
