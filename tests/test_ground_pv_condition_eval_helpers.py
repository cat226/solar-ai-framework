"""tests/test_ground_pv_condition_eval_helpers.py — pure helpers used by the
condition-model evaluation + two-stage real-world inference scripts (no model
/ no GPU needed)."""
from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from scripts.evaluate_condition_model import (
    CONDITION_ORDER,
    _class_names,
    _dataset_stats,
    _family_counts,
    _split_dirs,
)
from scripts.infer_condition_on_panels import _pad_box


def _img(p: Path, seed: int = 0) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), (seed % 256, (seed * 2) % 256, 40)).save(p)


def _mini_condition(tmp_path: Path) -> Path:
    d = tmp_path / "condition"
    rows = [("train", "a.jpg", "0 0.5 0.5 0.3 0.3\n", "CLEAN", "fam1"),
            ("train", "b.jpg", "4 0.5 0.5 0.2 0.2\n", "ELECTRICAL_DAMAGE", "fam2"),
            ("test", "c.jpg", "1 0.5 0.5 0.3 0.3\n2 0.2 0.2 0.1 0.1\n", "DUSTY;BIRD_DROP", "fam3"),
            ("test", "e.jpg", "4 0.5 0.5 0.2 0.2\n", "ELECTRICAL_DAMAGE", "fam4")]
    for split, name, lbl, _u, _f in rows:
        _img(d / "images" / split / "images" / name, hash(name) % 200)
        (d / "images" / split / "labels").mkdir(parents=True, exist_ok=True)
        (d / "images" / split / "labels" / f"{Path(name).stem}.txt").write_text(lbl, encoding="utf-8")
    (d / "data.yaml").write_text(yaml.safe_dump({
        "path": "datasets/ground_pv_research/condition",   # deliberately stale/relative
        "train": "images/train/images", "val": "images/train/images", "test": "images/test/images",
        "names": {0: "CLEAN", 1: "DUSTY", 2: "BIRD_DROP", 3: "PHYSICAL_DAMAGE", 4: "ELECTRICAL_DAMAGE"},
    }), encoding="utf-8")
    (d / "splits").mkdir()
    (d / "splits" / "m.csv").write_text(
        "derived_split,family_id,unified_classes\n"
        + "\n".join(f"{s},{f},{u}" for s, _n, _l, u, f in rows) + "\n", encoding="utf-8")
    return d


def test_split_dirs_recovers_from_stale_relative_path(tmp_path):
    d = _mini_condition(tmp_path)
    dirs = _split_dirs(d / "data.yaml", "test")
    assert dirs is not None
    images_dir, labels_dir = dirs
    assert images_dir.is_dir() and images_dir.name == "images"
    assert labels_dir.name == "labels" and labels_dir.is_dir()


def test_dataset_stats_counts_instances_and_images(tmp_path):
    d = _mini_condition(tmp_path)
    names = _class_names(d / "data.yaml")
    st = _dataset_stats(d / "data.yaml", "test", names)
    assert st["num_images"] == 2
    assert st["instances_per_class"]["DUSTY"] == 1
    assert st["instances_per_class"]["BIRD_DROP"] == 1
    assert st["instances_per_class"]["ELECTRICAL_DAMAGE"] == 1
    assert st["images_containing_class"]["ELECTRICAL_DAMAGE"] == 1
    assert list(st["instances_per_class"].keys()) == list(CONDITION_ORDER)


def test_family_counts_are_unique_per_split_and_class(tmp_path):
    d = _mini_condition(tmp_path)
    fc = _family_counts(d / "splits" / "m.csv")
    assert fc["test"]["ELECTRICAL_DAMAGE"] == 1
    assert fc["test"]["DUSTY"] == 1
    assert fc["train"]["ELECTRICAL_DAMAGE"] == 1


def test_pad_box_clamps_to_image_bounds():
    # centred box, 10% pad, well inside
    assert _pad_box(40, 40, 60, 60, 0.1, 100, 100) == (38, 38, 62, 62)
    # pad past the edges -> clamped to [0, W/H]
    assert _pad_box(2, 2, 98, 98, 0.5, 100, 100) == (0, 0, 100, 100)
