"""tests/test_ground_pv_remediate.py — the derived leakage-clean 3-class
research dataset built from the raw Roboflow dataset.

A small synthetic raw dataset is constructed in tmp_path with the same
pathologies the real audit found: one source photo re-exported across
official splits (basename family), a polygon (YOLO-seg) annotation, a
bird_drop-only-defect image, and an exact byte-duplicate spanning splits.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from training.detection.ground_pv.class_map import FIRST_EXPERIMENT_CLASS_NAMES
from training.detection.ground_pv.remediate import remediate, source_basename

# raw source taxonomy: 0 bird_drop / 1 cracked / 2 dusty / 3 panel
_RAW_NAMES = {0: "bird_drop", 1: "cracked", 2: "dusty", 3: "panel"}


def _img(path: Path, seed: int, size=(96, 96)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = abs(hash(seed))
    Image.new("RGB", size, (h % 200 + 20, (h // 200) % 200 + 20, (h // 7) % 200 + 20)).save(path)


def _build_raw(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({
        "names": ["bird_drop", "cracked", "dusty", "panel"], "nc": 4,
        "train": "../train/images", "val": "../valid/images", "test": "../test/images",
    }))
    spec = {
        # one source photo "burst001" re-exported into all three official splits
        "train": [
            ("burst001_jpg.rf.aaaa1111.jpg", ["3 0.5 0.5 0.9 0.9", "1 0.3 0.3 0.1 0.1"]),
            ("burst001_jpg.rf.aaaa2222.jpg", ["3 0.52 0.5 0.9 0.9", "1 0.31 0.3 0.1 0.1"]),
            ("solo_train_jpg.rf.bbbb1111.jpg", ["3 0.5 0.5 0.8 0.8"]),
            # polygon (YOLO-seg) dusty annotation + panel bbox
            ("poly001_jpg.rf.cccc1111.jpg", ["3 0.5 0.5 0.9 0.9",
                                             "2 0.10 0.10 0.50 0.10 0.30 0.60"]),
            # bird_drop-only defect image (plus a panel box)
            ("bird001_jpg.rf.dddd1111.jpg", ["3 0.5 0.5 0.9 0.9", "0 0.4 0.4 0.1 0.1"]),
            ("dup_src_jpg.rf.eeee1111.jpg", ["3 0.5 0.5 0.7 0.7"]),
        ],
        "valid": [
            ("burst001_jpg.rf.aaaa3333.jpg", ["3 0.48 0.5 0.9 0.9", "1 0.29 0.3 0.1 0.1"]),
            ("solo_valid_jpg.rf.ffff1111.jpg", ["3 0.5 0.5 0.8 0.8", "2 0.6 0.6 0.2 0.2"]),
        ],
        "test": [
            ("burst001_jpg.rf.aaaa4444.jpg", ["3 0.5 0.51 0.9 0.9", "1 0.3 0.31 0.1 0.1"]),
            ("solo_test_jpg.rf.gggg1111.jpg", ["3 0.5 0.5 0.8 0.8"]),
            # exact byte-duplicate of train/dup_src (written identically below)
            ("dup_src_jpg.rf.hhhh1111.jpg", ["3 0.5 0.5 0.7 0.7"]),
        ],
    }
    for split, items in spec.items():
        for i, (name, lines) in enumerate(items):
            _img(root / split / "images" / name, seed=hash((split, name)))
            (root / split / "labels").mkdir(parents=True, exist_ok=True)
            (root / split / "labels" / f"{Path(name).stem}.txt").write_text("\n".join(lines))
    # make the two dup_src images byte-identical
    src = root / "train/images/dup_src_jpg.rf.eeee1111.jpg"
    (root / "test/images/dup_src_jpg.rf.hhhh1111.jpg").write_bytes(src.read_bytes())
    return root


def _dir_digest(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(d).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


@pytest.fixture
def remediated(tmp_path):
    raw = _build_raw(tmp_path / "raw")
    before = _dir_digest(raw)
    rep = remediate(
        raw, tmp_path / "yolo", tmp_path / "splits", tmp_path / "reports",
        ratios={"train": 0.6, "valid": 0.2, "test": 0.2}, seed=42, overwrite=True,
    )
    return {"raw": raw, "before": before, "rep": rep,
            "yolo": tmp_path / "yolo", "splits": tmp_path / "splits", "reports": tmp_path / "reports"}


def test_raw_dataset_is_unchanged(remediated):
    assert _dir_digest(remediated["raw"]) == remediated["before"]


def test_no_family_crosses_splits(remediated):
    rep = remediated["rep"]
    assert rep.no_group_crosses_splits is True
    rows = list(csv.DictReader((remediated["yolo"] / "TRACEABILITY.csv").open(encoding="utf-8")))
    fam_splits: dict[str, set] = {}
    for r in rows:
        fam_splits.setdefault(r["family_id"], set()).add(r["derived_split"])
    assert all(len(v) == 1 for v in fam_splits.values())
    # the burst001 family's 4 copies all land in ONE split
    burst = [r for r in rows if r["source_basename"] == "burst001"]
    assert len(burst) == 4
    assert len({r["derived_split"] for r in burst}) == 1


def test_no_invalid_yolo_boxes_in_derived(remediated):
    for lf in (remediated["yolo"]).rglob("labels/*.txt"):
        for line in lf.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            assert len(parts) == 5, (lf, line)
            cid = int(parts[0]); vals = [float(x) for x in parts[1:]]
            assert 0 <= cid <= 2
            assert all(0.0 <= v <= 1.0 for v in vals)
            assert vals[2] > 0 and vals[3] > 0


def test_only_panel_cracked_dusty_present(remediated):
    seen = set()
    for lf in (remediated["yolo"]).rglob("labels/*.txt"):
        for line in lf.read_text().splitlines():
            if line.strip():
                seen.add(int(line.split()[0]))
    assert seen <= {0, 1, 2}
    data = yaml.safe_load((remediated["yolo"] / "data.yaml").read_text())
    assert {int(k): v for k, v in data["names"].items()} == FIRST_EXPERIMENT_CLASS_NAMES
    # bird_drop was excluded, count reported
    assert remediated["rep"].bird_drop_annotations_excluded == 1
    assert remediated["rep"].bird_drop_images_affected == 1
    # the bird image is kept, with only its panel box
    bird = list((remediated["yolo"]).rglob("bird001*.txt"))
    assert len(bird) == 1
    assert bird[0].read_text().strip().split("\n") == ["0 0.500000 0.500000 0.900000 0.900000"]


def test_polygon_converted_to_box(remediated):
    assert remediated["rep"].polygon_to_box_conversions == 1
    poly = list((remediated["yolo"]).rglob("poly001*.txt"))[0].read_text().splitlines()
    # panel bbox + converted dusty box (class 2)
    classes = sorted(int(l.split()[0]) for l in poly)
    assert classes == [0, 2]
    dusty = [l for l in poly if l.startswith("2 ")][0].split()
    cx, cy, w, h = map(float, dusty[1:])
    assert abs(cx - 0.30) < 1e-4 and abs(cy - 0.35) < 1e-4
    assert abs(w - 0.40) < 1e-4 and abs(h - 0.50) < 1e-4


def test_exact_duplicate_removed(remediated):
    rep = remediated["rep"]
    assert rep.exact_duplicates_removed == 1
    assert rep.derived_image_count == rep.raw_image_count - 1
    dup = list((remediated["yolo"]).rglob("dup_src*.jpg"))
    assert len(dup) == 1  # only one copy survived


def test_source_to_derived_traceability(remediated):
    raw, yolo = remediated["raw"], remediated["yolo"]
    rows = list(csv.DictReader((yolo / "TRACEABILITY.csv").open(encoding="utf-8")))
    derived_imgs = {p.name for p in yolo.rglob("images/*") if p.is_file()}
    assert len(rows) == len(derived_imgs)
    for r in rows:
        assert r["derived_filename"] in derived_imgs
        src = raw / r["source_path"]
        assert src.is_file(), r["source_path"]
        assert hashlib.sha256(src.read_bytes()).hexdigest() == r["sha256"]
        assert r["source_official_split"] in {"train", "valid", "test"}
        assert r["family_id"].startswith("fam")
        # the derived image is a byte copy of its source
        got = next(p for p in yolo.rglob(f"images/{r['derived_filename']}"))
        assert got.read_bytes() == src.read_bytes()


def test_split_manifests_are_internally_consistent(remediated):
    yolo, splits, rep = remediated["yolo"], remediated["splits"], remediated["rep"]
    clean = list(csv.DictReader((splits / "leakage_clean_split/manifest.csv").open(encoding="utf-8")))
    trace = list(csv.DictReader((yolo / "TRACEABILITY.csv").open(encoding="utf-8")))
    assert clean == trace  # same rows in both places

    # official manifest covers every raw image exactly once
    official = list(csv.DictReader((splits / "official_split/manifest.csv").open(encoding="utf-8")))
    raw_imgs = [p for p in remediated["raw"].rglob("images/*") if p.is_file()]
    assert len(official) == len(raw_imgs)
    assert {r["official_split"] for r in official} <= {"train", "valid", "test"}

    # per-split counts in the report match the manifest and the files on disk
    for s in ("train", "valid", "test"):
        n_rows = sum(1 for r in clean if r["derived_split"] == s)
        n_files = sum(1 for _ in (yolo / s / "images").glob("*"))
        assert n_rows == n_files == rep.final_split_counts[s]
        n_lbls = sum(1 for _ in (yolo / s / "labels").glob("*.txt"))
        assert n_lbls == n_files

    # unique family counts add up and are disjoint across splits
    fam_by_split = {s: {r["family_id"] for r in clean if r["derived_split"] == s}
                    for s in ("train", "valid", "test")}
    assert fam_by_split["train"].isdisjoint(fam_by_split["valid"])
    assert fam_by_split["train"].isdisjoint(fam_by_split["test"])
    assert fam_by_split["valid"].isdisjoint(fam_by_split["test"])
    for s in ("train", "valid", "test"):
        assert rep.unique_groups_per_split[s] == len(fam_by_split[s])


def test_report_files_written(remediated):
    reports = remediated["reports"]
    assert (reports / "remediation_report.md").is_file()
    payload = json.loads((reports / "remediation_report.json").read_text())
    for key in ("raw_image_count", "derived_image_count", "images_removed_total",
                "exact_duplicates_removed", "near_duplicate_groups", "final_split_counts",
                "class_counts", "unique_groups_per_split", "no_group_crosses_splits",
                "polygon_to_box_conversions", "bird_drop_annotations_excluded"):
        assert key in payload
    assert (remediated["yolo"] / "REMEDIATION.json").is_file()
