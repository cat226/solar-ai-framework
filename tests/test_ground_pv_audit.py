"""tests/test_ground_pv_audit.py — the ground-PV dataset audit on small
synthetic YOLO datasets built in tmp_path (no network, no real images
required beyond tiny generated PNGs)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

from training.detection.ground_pv.audit import run_audit
from training.detection.ground_pv.normalize import normalize_dataset

_SOURCE_NAMES = {0: "panel", 1: "bird_drop", 2: "cracked", 3: "dusty"}


def _make_dataset(root: Path, spec: dict[str, list[tuple[str, list[str], tuple[int, int] | None]]],
                  names: dict[int, str] = _SOURCE_NAMES) -> Path:
    """spec: split -> list of (stem, label_lines, image_size_or_None).
    image_size None => write a deliberately corrupt image file."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({
        "path": str(root), "train": "train/images", "val": "valid/images", "test": "test/images",
        "names": names,
    }))
    for split, items in spec.items():
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for stem, lines, size in items:
            if size is None:
                (img_dir / f"{stem}.jpg").write_bytes(b"not a real jpeg")
            else:
                # distinct per (split, stem) so unrelated images are not
                # flagged as duplicates of each other
                h = abs(hash((split, stem)))
                color = (h % 200 + 20, (h // 200) % 200 + 20, (h // 40000) % 200 + 20)
                Image.new("RGB", size, color).save(img_dir / f"{stem}.jpg")
            if lines is not None:
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
    return root


def test_clean_dataset_passes(tmp_path):
    root = _make_dataset(tmp_path / "ds", {
        "train": [("a", ["0 0.5 0.5 0.4 0.4", "1 0.2 0.2 0.1 0.1"], (640, 480)),
                  ("b", ["0 0.5 0.5 0.9 0.9"], (640, 480))],
        "valid": [("c", ["0 0.5 0.5 0.5 0.5", "3 0.5 0.5 0.2 0.2"], (800, 600))],
        "test":  [("d", ["0 0.4 0.4 0.3 0.3", "2 0.4 0.4 0.15 0.15"], (1024, 768))],
    })
    result = run_audit(root)
    assert result.fail_closed_reasons == []
    assert result.as_dict()["passed"] is True
    assert result.class_instance_counts == {"panel": 4, "bird_drop": 1, "cracked": 1, "dusty": 1}
    assert result.images_per_class["panel"] == 4
    assert result.multi_class_images == 3


def test_missing_split_fails_closed(tmp_path):
    root = _make_dataset(tmp_path / "ds", {
        "train": [("a", ["0 0.5 0.5 0.4 0.4"], (640, 480))],
        "valid": [("c", ["0 0.5 0.5 0.5 0.5"], (640, 480))],
    })
    result = run_audit(root)
    assert any("missing split: test" in r for r in result.fail_closed_reasons)


def test_invalid_class_id_and_bad_coords_are_hard_errors(tmp_path):
    root = _make_dataset(tmp_path / "ds", {
        "train": [("a", ["9 0.5 0.5 0.4 0.4"], (64, 64))],
        "valid": [("c", ["0 1.4 0.5 0.5 0.5"], (64, 64))],
        "test":  [("d", ["0 0.5 0.5 0.3 0.3"], (64, 64))],
    })
    result = run_audit(root)
    kinds = {i.kind for i in result.label_issues}
    assert "invalid_class_id" in kinds
    assert "coord_out_of_range" in kinds
    assert any("hard annotation error" in r for r in result.fail_closed_reasons)


def test_cross_split_exact_duplicate_is_leakage_fail_closed(tmp_path):
    root = tmp_path / "ds"
    _make_dataset(root, {
        "train": [("a", ["0 0.5 0.5 0.4 0.4"], (128, 128))],
        "valid": [("c", ["0 0.5 0.5 0.4 0.4"], (128, 128))],
        "test":  [("d", ["0 0.5 0.5 0.4 0.4"], (128, 128))],
    })
    # make train/a and valid/c byte-identical
    (root / "valid/images/c.jpg").write_bytes((root / "train/images/a.jpg").read_bytes())
    result = run_audit(root)
    assert result.cross_split_exact_leakage
    assert any("leakage" in r for r in result.fail_closed_reasons)


def test_corruption_over_threshold_fails_closed(tmp_path):
    root = _make_dataset(tmp_path / "ds", {
        "train": [("a", ["0 0.5 0.5 0.4 0.4"], None), ("b", ["0 0.5 0.5 0.4 0.4"], None)],
        "valid": [("c", ["0 0.5 0.5 0.4 0.4"], (64, 64))],
        "test":  [("d", ["0 0.5 0.5 0.4 0.4"], (64, 64))],
    })
    result = run_audit(root, max_corruption_fraction=0.05)
    assert len(result.corrupt_images) == 2
    assert any("corrupt" in r for r in result.fail_closed_reasons)


def test_defect_only_and_panel_only_images_are_counted(tmp_path):
    root = _make_dataset(tmp_path / "ds", {
        "train": [("panelonly", ["0 0.5 0.5 0.4 0.4"], (64, 64)),
                  ("defectonly", ["1 0.5 0.5 0.2 0.2"], (64, 64))],
        "valid": [("c", ["0 0.5 0.5 0.5 0.5"], (64, 64))],
        "test":  [("d", ["0 0.5 0.5 0.3 0.3"], (64, 64))],
    })
    result = run_audit(root)
    assert any(m.endswith("defectonly.jpg") for m in result.images_defect_only_no_panel)
    assert any(m.endswith("panelonly.jpg") for m in result.images_panel_no_defect)


def test_polygon_segmentation_lines_are_counted_not_called_malformed(tmp_path):
    # default _SOURCE_NAMES = identity order: 0 panel / 1 bird_drop / 2 cracked / 3 dusty
    # class 2 (cracked) as a triangle polygon; class 0 (panel) as a bbox
    poly = "2 0.1 0.1 0.5 0.1 0.3 0.6"
    root = _make_dataset(tmp_path / "ds", {
        "train": [("a", [poly, "0 0.5 0.5 0.9 0.9"], (64, 64))],
        "valid": [("c", ["0 0.5 0.5 0.5 0.5"], (64, 64))],
        "test":  [("d", ["0 0.4 0.4 0.3 0.3"], (64, 64))],
    })
    result = run_audit(root)
    kinds = {i.kind for i in result.label_issues}
    assert "malformed_line" not in kinds
    assert result.annotation_format["polygon_lines"] == 1
    assert result.annotation_format["bbox_lines"] == 3
    assert result.annotation_format["files_mixed_bbox_and_polygon"] == 1
    assert result.annotation_format["is_pure_detection_dataset"] is False
    # cracked polygon IS counted as an instance
    assert result.class_instance_counts["cracked"] == 1
    assert result.class_instance_counts_polygon["cracked"] == 1
    assert result.class_instance_counts_bbox["cracked"] == 0
    # mixed format is fail-closed
    assert any("polygon" in r for r in result.fail_closed_reasons)


def test_normalize_converts_polygons_to_bboxes(tmp_path):
    poly = "2 0.10 0.10 0.50 0.10 0.30 0.60"  # xs 0.1..0.5, ys 0.1..0.6; class 2 = cracked
    root = _make_dataset(tmp_path / "ds", {
        "train": [("a", [poly], (64, 64))],
        "valid": [("c", ["0 0.5 0.5 0.5 0.5"], (64, 64))],
        "test":  [("d", ["0 0.4 0.4 0.3 0.3"], (64, 64))],
    })
    out = tmp_path / "yolo"
    meta = normalize_dataset(root, out)
    assert meta["polygons_converted_to_bbox"] == 1
    line = (out / "train/labels/a.txt").read_text().split()
    assert line[0] == "2"  # identity remap: cracked stays research id 2
    cx, cy, w, h = map(float, line[1:])
    assert abs(cx - 0.30) < 1e-4 and abs(cy - 0.35) < 1e-4
    assert abs(w - 0.40) < 1e-4 and abs(h - 0.50) < 1e-4
    # re-audit the normalized copy: now a pure-detection dataset
    r2 = run_audit(out)
    assert r2.annotation_format["is_pure_detection_dataset"] is True


def test_write_reports_caps_verbose_lists_in_committed_json(tmp_path):
    from training.detection.ground_pv.audit import _SAMPLE_CAP, write_reports
    root = _make_dataset(tmp_path / "ds", {
        "train": [(f"a{i}", ["0 0.5 0.5 0.4 0.4"], (64, 64)) for i in range(3)],
        "valid": [("c", ["0 0.5 0.5 0.5 0.5"], (64, 64))],
        "test":  [("d", ["0 0.4 0.4 0.3 0.3"], (64, 64))],
    })
    result = run_audit(root)
    # inject an oversized list to exercise the cap
    result.near_duplicate_pairs = [{"a": f"x{i}", "b": f"y{i}", "hamming": 1} for i in range(_SAMPLE_CAP + 50)]
    reports = tmp_path / "reports"
    json_path, md_path = write_reports(result, reports)
    import json as _json
    capped = _json.loads(json_path.read_text())
    assert len(capped["near_duplicate_pairs"]) == _SAMPLE_CAP
    assert capped["near_duplicate_pairs__truncated"]["total"] == _SAMPLE_CAP + 50
    full = _json.loads((reports / "dataset_audit_full.json").read_text())
    assert len(full["near_duplicate_pairs"]) == _SAMPLE_CAP + 50
    assert json_path.stat().st_size < (reports / "dataset_audit_full.json").stat().st_size


def test_normalize_applies_remap_and_writes_research_yaml(tmp_path):
    # source in alphabetical order -> non-identity remap
    root = _make_dataset(tmp_path / "ds", {
        "train": [("a", ["3 0.5 0.5 0.4 0.4", "0 0.2 0.2 0.1 0.1"], (64, 64))],
        "valid": [("c", ["3 0.5 0.5 0.5 0.5"], (64, 64))],
        "test":  [("d", ["3 0.4 0.4 0.3 0.3"], (64, 64))],
    }, names={0: "bird_drop", 1: "cracked", 2: "dusty", 3: "panel"})
    out = tmp_path / "yolo"
    meta = normalize_dataset(root, out)
    assert meta["remap_is_identity"] is False
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == {0: "panel", 1: "bird_drop", 2: "cracked", 3: "dusty"}
    # source id 3 (panel) -> research id 0; source id 0 (bird_drop) -> research id 1
    lines = (out / "train/labels/a.txt").read_text().split()
    assert lines[0] == "0" and lines[5] == "1"
    # re-audit the normalized copy -> clean
    assert run_audit(out).as_dict()["passed"] is True
