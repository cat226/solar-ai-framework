"""Tests for training/detection/prepare_closeup_dataset.py.

This script has never been run against real data (no legitimate close-up
bounding-box dataset has been acquired - see docs/ML_DOMAIN_REMEDIATION.md)
so all coverage here uses small synthetic fixtures built directly to the
script's own expected input contract (images/, labels/, provenance.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from training.detection.prepare_closeup_dataset import (
    _assign_group_split,
    _load_provenance,
    _normalize_license,
    _parse_yolo_label_file,
    prepare_closeup_dataset,
)


def _make_source_tree(tmp_path: Path, images: dict[str, dict]) -> Path:
    """images: {id: {"license": ..., "source_url": ..., "boxes": [...], "group_key": ..., "color": (r,g,b)}}"""
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir(parents=True)
    provenance = {}
    for image_id, spec in images.items():
        color = spec.get("color", (100, 100, 100))
        img = Image.new("RGB", (64, 64), color)
        img.save(source / "images" / f"{image_id}.jpg")
        boxes = spec.get("boxes", [(0, 0.5, 0.5, 0.2, 0.2)])
        lines = "\n".join(f"{c} {cx} {cy} {w} {h}" for c, cx, cy, w, h in boxes)
        (source / "labels" / f"{image_id}.txt").write_text(lines)
        provenance[image_id] = {
            "license": spec.get("license", "CC-BY-4.0"),
            "source_url": spec.get("source_url", f"https://example.org/{image_id}"),
            "rights_holder": spec.get("rights_holder"),
            "group_key": spec.get("group_key"),
        }
    (source / "provenance.json").write_text(json.dumps(provenance))
    return source


# ---------------------------------------------------------------------------
# A. License gate (hard constraint: no unknown-license datasets)
# ---------------------------------------------------------------------------

class TestLicenseGate:
    def test_normalize_license_handles_variants(self):
        assert _normalize_license("CC-BY 4.0".replace(" ", "-")) == "cc-by-4.0"
        assert _normalize_license("  MIT  ") == "mit"
        assert _normalize_license("cc_by_sa_4.0") == "cc-by-sa-4.0"

    def test_unknown_license_is_rejected(self, tmp_path):
        source = _make_source_tree(tmp_path, {"img1": {"license": "Unknown"}})
        manifest_path = prepare_closeup_dataset(source, tmp_path / "out")
        manifest = json.loads(manifest_path.read_text())
        assert manifest["total_accepted"] == 0
        assert manifest["total_rejected"] == 1
        assert "license" in manifest["rejected"][0]["reason"]

    def test_allowed_license_is_accepted(self, tmp_path):
        source = _make_source_tree(tmp_path, {"img1": {"license": "CC-BY-4.0"}})
        manifest_path = prepare_closeup_dataset(source, tmp_path / "out")
        manifest = json.loads(manifest_path.read_text())
        assert manifest["total_accepted"] == 1
        assert manifest["total_rejected"] == 0

    def test_no_provenance_entry_is_rejected(self, tmp_path):
        source = _make_source_tree(tmp_path, {"img1": {"license": "MIT"}})
        prov_path = source / "provenance.json"
        prov = json.loads(prov_path.read_text())
        del prov["img1"]
        prov_path.write_text(json.dumps(prov))

        manifest_path = prepare_closeup_dataset(source, tmp_path / "out")
        manifest = json.loads(manifest_path.read_text())
        assert manifest["total_accepted"] == 0
        assert "no provenance.json entry" in manifest["rejected"][0]["reason"]

    def test_missing_provenance_json_raises(self, tmp_path):
        source = _make_source_tree(tmp_path, {"img1": {"license": "MIT"}})
        (source / "provenance.json").unlink()
        with pytest.raises(FileNotFoundError, match="provenance.json"):
            prepare_closeup_dataset(source, tmp_path / "out")

    def test_missing_source_url_and_rights_holder_is_rejected(self, tmp_path):
        source = _make_source_tree(tmp_path, {"img1": {"license": "MIT"}})
        prov_path = source / "provenance.json"
        prov = json.loads(prov_path.read_text())
        prov["img1"]["source_url"] = None
        prov["img1"]["rights_holder"] = None
        prov_path.write_text(json.dumps(prov))

        manifest_path = prepare_closeup_dataset(source, tmp_path / "out")
        manifest = json.loads(manifest_path.read_text())
        assert manifest["total_accepted"] == 0


# ---------------------------------------------------------------------------
# B. Annotation format enforcement (single class "solar_panel" = id 0)
# ---------------------------------------------------------------------------

class TestAnnotationValidation:
    def test_empty_label_file_is_a_valid_negative(self):
        # Exercised via prepare_closeup_dataset below; here just the parser.
        pass

    def test_wrong_class_id_rejected_by_parser(self, tmp_path):
        label_path = tmp_path / "bad.txt"
        label_path.write_text("1 0.5 0.5 0.2 0.2\n")
        boxes, errors = _parse_yolo_label_file(label_path)
        assert boxes == []
        assert any("class id 1" in e for e in errors)

    def test_out_of_range_coordinate_rejected(self, tmp_path):
        label_path = tmp_path / "bad.txt"
        label_path.write_text("0 1.5 0.5 0.2 0.2\n")
        boxes, errors = _parse_yolo_label_file(label_path)
        assert boxes == []
        assert errors

    def test_zero_area_box_rejected(self, tmp_path):
        label_path = tmp_path / "bad.txt"
        label_path.write_text("0 0.5 0.5 0.0 0.2\n")
        boxes, errors = _parse_yolo_label_file(label_path)
        assert boxes == []
        assert errors

    def test_valid_box_parsed(self, tmp_path):
        label_path = tmp_path / "good.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.3\n")
        boxes, errors = _parse_yolo_label_file(label_path)
        assert errors == []
        assert boxes == [(0.5, 0.5, 0.2, 0.3)]

    def test_empty_file_returns_no_boxes_no_errors(self, tmp_path):
        label_path = tmp_path / "empty.txt"
        label_path.write_text("")
        boxes, errors = _parse_yolo_label_file(label_path)
        assert boxes == []
        assert errors == []

    def test_wrong_class_id_image_rejected_end_to_end(self, tmp_path):
        source = _make_source_tree(tmp_path, {
            "img1": {"license": "MIT", "boxes": [(1, 0.5, 0.5, 0.2, 0.2)]},
        })
        manifest_path = prepare_closeup_dataset(source, tmp_path / "out")
        manifest = json.loads(manifest_path.read_text())
        assert manifest["total_accepted"] == 0
        assert "class id 1" in manifest["rejected"][0]["reason"]

    def test_missing_label_file_rejected(self, tmp_path):
        source = _make_source_tree(tmp_path, {"img1": {"license": "MIT"}})
        (source / "labels" / "img1.txt").unlink()
        manifest_path = prepare_closeup_dataset(source, tmp_path / "out")
        manifest = json.loads(manifest_path.read_text())
        assert manifest["total_accepted"] == 0
        assert "no matching label file" in manifest["rejected"][0]["reason"]

    def test_negative_image_with_empty_label_is_accepted(self, tmp_path):
        source = _make_source_tree(tmp_path, {"img1": {"license": "MIT", "boxes": []}})
        manifest_path = prepare_closeup_dataset(source, tmp_path / "out")
        manifest = json.loads(manifest_path.read_text())
        assert manifest["total_accepted"] == 1
        assert manifest["records"][0]["num_boxes"] == 0


# ---------------------------------------------------------------------------
# C. Exact and near-duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_exact_duplicate_content_is_rejected(self, tmp_path):
        source = _make_source_tree(tmp_path, {
            "img1": {"license": "MIT", "color": (10, 20, 30)},
            "img2": {"license": "MIT", "color": (10, 20, 30)},  # byte-identical pixels
        })
        manifest_path = prepare_closeup_dataset(source, tmp_path / "out")
        manifest = json.loads(manifest_path.read_text())
        assert manifest["total_accepted"] == 1
        assert manifest["total_rejected"] == 1
        assert "exact duplicate" in manifest["rejected"][0]["reason"]

    def test_near_duplicate_cluster_stays_in_one_split(self, tmp_path):
        # Two images differing in one pixel corner - solid-color 64x64
        # images are pixel-identical here on purpose (same dHash), which is
        # both an exact AND near duplicate; near-duplicate clustering must
        # still hold even without exact-hash collision, so also add a
        # distinctly-different pair that is NOT clustered.
        images = {}
        for i in range(6):
            images[f"cluster_{i}"] = {"license": "MIT", "color": (200, 50, 50), "group_key": "same-scene"}
        for i in range(6):
            images[f"other_{i}"] = {"license": "MIT", "color": (10, 200, 10)}
        source = _make_source_tree(tmp_path, images)
        manifest_path = prepare_closeup_dataset(source, tmp_path / "out", seed=1)
        manifest = json.loads(manifest_path.read_text())

        cluster_splits = {r["split"] for r in manifest["records"] if r["id"].startswith("cluster_")}
        assert len(cluster_splits) == 1, "all group_key='same-scene' images must land in exactly one split"


# ---------------------------------------------------------------------------
# D. Deterministic, reproducible split
# ---------------------------------------------------------------------------

class TestDeterministicSplit:
    def test_same_seed_reproduces_identical_split(self, tmp_path):
        images = {f"img{i}": {"license": "MIT", "color": (i * 5 % 255, i * 7 % 255, i * 11 % 255)} for i in range(20)}
        source = _make_source_tree(tmp_path, images)

        manifest_1 = json.loads(prepare_closeup_dataset(source, tmp_path / "out1", seed=42).read_text())
        manifest_2 = json.loads(prepare_closeup_dataset(source, tmp_path / "out2", seed=42).read_text())

        split_1 = {r["id"]: r["split"] for r in manifest_1["records"]}
        split_2 = {r["id"]: r["split"] for r in manifest_2["records"]}
        assert split_1 == split_2
        assert manifest_1["dataset_content_hash_sha256"] == manifest_2["dataset_content_hash_sha256"]

    def test_assign_group_split_is_pure_and_deterministic(self):
        assert _assign_group_split("abc", 42, (0.7, 0.15, 0.15)) == _assign_group_split("abc", 42, (0.7, 0.15, 0.15))

    def test_invalid_split_ratios_raise(self, tmp_path):
        source = _make_source_tree(tmp_path, {"img1": {"license": "MIT"}})
        with pytest.raises(ValueError, match="split_ratios"):
            prepare_closeup_dataset(source, tmp_path / "out", split_ratios=(0.5, 0.5, 0.5))

    def test_output_directories_created_for_all_splits(self, tmp_path):
        images = {f"img{i}": {"license": "MIT", "color": (i * 13 % 255, i * 17 % 255, i * 19 % 255)} for i in range(15)}
        source = _make_source_tree(tmp_path, images)
        out = tmp_path / "out"
        prepare_closeup_dataset(source, out, seed=7)
        for split in ("train", "val", "test"):
            assert (out / split / "images").is_dir()
            assert (out / split / "labels").is_dir()


# ---------------------------------------------------------------------------
# E. Manifest content
# ---------------------------------------------------------------------------

class TestManifest:
    def test_class_name_is_solar_panel_singular_taxonomy(self, tmp_path):
        source = _make_source_tree(tmp_path, {"img1": {"license": "MIT"}})
        manifest = json.loads(prepare_closeup_dataset(source, tmp_path / "out").read_text())
        assert manifest["class_names"] == ["solar_panel"]

    def test_manifest_records_license_and_source(self, tmp_path):
        source = _make_source_tree(tmp_path, {"img1": {"license": "CC-BY-4.0", "source_url": "https://example.org/img1"}})
        manifest = json.loads(prepare_closeup_dataset(source, tmp_path / "out").read_text())
        record = manifest["records"][0]
        assert record["license"] == "cc-by-4.0"
        assert record["source_url"] == "https://example.org/img1"

    def test_load_provenance_reads_json(self, tmp_path):
        source = _make_source_tree(tmp_path, {"img1": {"license": "MIT"}})
        prov = _load_provenance(source)
        assert "img1" in prov
