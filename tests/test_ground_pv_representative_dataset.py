"""tests/test_ground_pv_representative_dataset.py — Phase 14B tooling for the
representative deployment-domain panel dataset.

Grouping/split tests construct `SourceManifestEntry` objects directly with
controlled `dhash16` hex strings (known Hamming distances) rather than real
near-duplicate images, so the union-find/threshold logic is tested exactly
and deterministically without depending on PIL's actual perceptual-hash
output for handcrafted test images. Image-metadata-extraction tests use
real small PNGs written to a tmp_path (fast, no network, no GPU).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from training.detection.ground_pv.representative_dataset import (
    DEFAULT_SPLIT_RATIOS,
    SourceManifestEntry,
    assign_scene_isolated_splits,
    build_readiness_checklist,
    build_source_manifest,
    check_overlap_and_size,
    compute_count_error_stats,
    compute_duplicate_leakage_report,
    compute_panel_count_stats,
    compute_resolution_and_aspect_stats,
    extract_image_metadata,
    group_into_scenes,
    load_source_manifest,
    save_source_manifest,
    save_split_manifest,
    validate_representative_annotations,
    verify_no_scene_crosses_splits,
)


def _make_entry(image_id, dhash16_hex, sha256="s" + "0" * 63, w=1280, h=720, **kw):
    return SourceManifestEntry(
        image_id=image_id, source_family_id=image_id, original_filename=f"{image_id}.jpg",
        source_path=f"/x/{image_id}.jpg", width=w, height=h, aspect_ratio=round(w / h, 4),
        sha256=sha256, capture_source="first_party", scene_id=image_id,
        provenance_status="first_party", dhash16=dhash16_hex, **kw,
    )


def _hex_from_bits(bits: str) -> str:
    """256-bit hex string from a 256-char '0'/'1' string (for controlled Hamming distances)."""
    assert len(bits) == 256
    return f"{int(bits, 2):064x}"


_BASE_BITS = "0" * 256


def _flip_n_bits(n: int) -> str:
    bits = list(_BASE_BITS)
    for i in range(n):
        bits[i] = "1"
    return _hex_from_bits("".join(bits))


# ---------------------------------------------------------------------------
# Image metadata extraction (real files)
# ---------------------------------------------------------------------------


class TestImageMetadataExtraction:
    def test_extract_metadata_basic_fields(self, tmp_path):
        img_path = tmp_path / "a.jpg"
        Image.new("RGB", (800, 600), (10, 20, 30)).save(img_path)
        meta = extract_image_metadata(img_path)
        assert meta.width == 800
        assert meta.height == 600
        assert meta.aspect_ratio == pytest.approx(800 / 600, rel=1e-3)
        assert len(meta.sha256) == 64
        assert len(meta.dhash16) == 32  # 256 bits = 32 bytes

    def test_extract_metadata_raises_on_corrupt_file(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not an image")
        with pytest.raises(Exception):
            extract_image_metadata(bad)

    def test_build_source_manifest_scans_directory(self, tmp_path):
        for i in range(3):
            Image.new("RGB", (640 + i, 480), (i, i, i)).save(tmp_path / f"img{i}.jpg")
        entries, corrupt = build_source_manifest(tmp_path, capture_source="first_party")
        assert len(entries) == 3
        assert corrupt == []
        assert all(e.scene_id == e.image_id for e in entries)  # singleton until grouped
        assert all(e.source_family_id == e.image_id for e in entries)
        assert all(e.annotation_status == "pending" for e in entries)
        assert all(e.provenance_status == "first_party" for e in entries)

    def test_build_source_manifest_reports_corrupt_without_dropping_silently(self, tmp_path):
        Image.new("RGB", (640, 480), (1, 1, 1)).save(tmp_path / "good.jpg")
        (tmp_path / "corrupt.jpg").write_bytes(b"garbage")
        entries, corrupt = build_source_manifest(tmp_path, capture_source="first_party")
        assert len(entries) == 1
        assert len(corrupt) == 1
        assert "corrupt.jpg" in corrupt[0]["path"]

    def test_build_source_manifest_rejects_invalid_capture_source(self, tmp_path):
        with pytest.raises(ValueError):
            build_source_manifest(tmp_path, capture_source="made_up")

    def test_manifest_save_load_roundtrip(self, tmp_path):
        Image.new("RGB", (640, 480), (5, 5, 5)).save(tmp_path / "img.jpg")
        entries, _ = build_source_manifest(tmp_path, capture_source="first_party")
        out = tmp_path / "manifest.json"
        save_source_manifest(entries, out)
        loaded = load_source_manifest(out)
        assert len(loaded) == 1
        assert loaded[0].image_id == entries[0].image_id
        assert loaded[0].sha256 == entries[0].sha256


# ---------------------------------------------------------------------------
# Scene / source-family grouping
# ---------------------------------------------------------------------------


class TestSceneGrouping:
    def test_exact_duplicate_shas_grouped_into_one_scene(self):
        entries = [
            _make_entry("img1", _flip_n_bits(0), sha256="a" * 64),
            _make_entry("img2", _flip_n_bits(200), sha256="a" * 64),  # same sha despite different dhash
        ]
        grouped, near_pairs = group_into_scenes(entries)
        assert grouped[0].scene_id == grouped[1].scene_id

    def test_near_duplicate_dhash_within_threshold_grouped(self):
        # Hamming distance 3 (< default threshold 8) -> same scene
        entries = [
            _make_entry("img1", _flip_n_bits(0), sha256="a" * 64),
            _make_entry("img2", _flip_n_bits(3), sha256="b" * 64),
        ]
        grouped, near_pairs = group_into_scenes(entries)
        assert grouped[0].scene_id == grouped[1].scene_id
        assert len(near_pairs) == 1
        assert near_pairs[0]["dhash16_hamming"] == 3

    def test_dissimilar_images_stay_separate_scenes(self):
        # Hamming distance 128 (half the bits differ) -> clearly different scenes
        entries = [
            _make_entry("img1", _flip_n_bits(0), sha256="a" * 64),
            _make_entry("img2", _flip_n_bits(128), sha256="b" * 64),
        ]
        grouped, near_pairs = group_into_scenes(entries)
        assert grouped[0].scene_id != grouped[1].scene_id
        assert near_pairs == []

    def test_transitive_grouping_via_chain(self):
        # img1~img2 (dist 3), img2~img3 (dist 3 from img2's hash) -> all one scene
        h1 = _flip_n_bits(0)
        h2 = _flip_n_bits(3)
        h3 = _flip_n_bits(6)
        entries = [
            _make_entry("img1", h1, sha256="a" * 64),
            _make_entry("img2", h2, sha256="b" * 64),
            _make_entry("img3", h3, sha256="c" * 64),
        ]
        grouped, _ = group_into_scenes(entries)
        scene_ids = {e.scene_id for e in grouped}
        assert len(scene_ids) == 1

    def test_custom_threshold_narrows_grouping(self):
        entries = [
            _make_entry("img1", _flip_n_bits(0), sha256="a" * 64),
            _make_entry("img2", _flip_n_bits(5), sha256="b" * 64),
        ]
        grouped_default, _ = group_into_scenes(list(entries), dhash16_merge_hamming=8)
        assert grouped_default[0].scene_id == grouped_default[1].scene_id
        entries2 = [
            _make_entry("img1", _flip_n_bits(0), sha256="a" * 64),
            _make_entry("img2", _flip_n_bits(5), sha256="b" * 64),
        ]
        grouped_strict, _ = group_into_scenes(entries2, dhash16_merge_hamming=2)
        assert grouped_strict[0].scene_id != grouped_strict[1].scene_id

    def test_empty_entries_list(self):
        grouped, near_pairs = group_into_scenes([])
        assert grouped == []
        assert near_pairs == []

    def test_single_entry_is_its_own_scene(self):
        entries = [_make_entry("img1", _flip_n_bits(0))]
        grouped, near_pairs = group_into_scenes(entries)
        assert len(grouped) == 1
        assert near_pairs == []


# ---------------------------------------------------------------------------
# Scene-isolated, deterministic split assignment
# ---------------------------------------------------------------------------


class TestSplitAssignment:
    def _many_scenes(self, n_scenes=20, images_per_scene=3):
        entries = []
        for s in range(n_scenes):
            for i in range(images_per_scene):
                e = _make_entry(f"s{s}_img{i}", _flip_n_bits(0), sha256=f"scene{s}img{i}" + "0" * 40)
                e.scene_id = f"scene{s:03d}"
                e.source_family_id = f"scene{s:03d}"
                entries.append(e)
        return entries

    def test_no_scene_ever_crosses_splits(self):
        entries = self._many_scenes()
        result = assign_scene_isolated_splits(entries, seed=42)
        check = verify_no_scene_crosses_splits(entries, result.image_to_split)
        assert check["ok"] is True
        assert check["crossing_scenes"] == {}

    def test_verify_detects_a_forced_crossing(self):
        entries = self._many_scenes(n_scenes=2, images_per_scene=2)
        # Deliberately split the SAME scene across train/test (forbidden).
        # _many_scenes() image_ids are "s{s}_img{i}" with scene_id "scene{s:03d}".
        bad_split = {
            "s0_img0": "train", "s0_img1": "test",
            "s1_img0": "train", "s1_img1": "train",
        }
        check = verify_no_scene_crosses_splits(entries, bad_split)
        assert check["ok"] is False
        assert "scene000" in check["crossing_scenes"]

    def test_split_is_deterministic_across_runs(self):
        entries_a = self._many_scenes()
        entries_b = self._many_scenes()
        result_a = assign_scene_isolated_splits(entries_a, seed=42)
        result_b = assign_scene_isolated_splits(entries_b, seed=42)
        assert result_a.image_to_split == result_b.image_to_split

    def test_split_is_deterministic_under_input_reordering(self):
        import random
        entries_a = self._many_scenes()
        entries_b = list(entries_a)
        random.shuffle(entries_b)
        result_a = assign_scene_isolated_splits(entries_a, seed=42)
        result_b = assign_scene_isolated_splits(entries_b, seed=42)
        assert result_a.scene_to_split == result_b.scene_to_split

    def test_split_ratios_approximately_respected(self):
        entries = self._many_scenes(n_scenes=30, images_per_scene=2)
        result = assign_scene_isolated_splits(entries, seed=42)
        counts = {"train": 0, "valid": 0, "test": 0}
        for split in result.image_to_split.values():
            counts[split] += 1
        total = sum(counts.values())
        assert counts["train"] / total == pytest.approx(DEFAULT_SPLIT_RATIOS["train"], abs=0.15)

    def test_save_split_manifest_writes_csv(self, tmp_path):
        entries = self._many_scenes(n_scenes=3, images_per_scene=2)
        result = assign_scene_isolated_splits(entries, seed=42)
        out = save_split_manifest(entries, result.image_to_split, tmp_path / "split.csv")
        text = out.read_text(encoding="utf-8")
        assert "image_id,scene_id,source_family_id,split" in text
        assert text.count("\n") == len(entries) + 1  # header + one row per image


# ---------------------------------------------------------------------------
# Annotation validation (extends validate_yolo_annotations.py)
# ---------------------------------------------------------------------------


class TestAnnotationValidation:
    def test_overlap_diagnostic_flags_near_duplicate_boxes(self):
        # cx/cy offset by 0.005 against w/h=0.2 -> IoU ~0.906, above the 0.85 default threshold
        boxes = [
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "line": 1},
            {"class_id": 0, "cx": 0.505, "cy": 0.505, "w": 0.2, "h": 0.2, "line": 2},  # near-identical
        ]
        issues = check_overlap_and_size(boxes)
        assert any("excessive overlap" in i for i in issues)

    def test_adjacent_non_overlapping_panels_not_flagged(self):
        boxes = [
            {"class_id": 0, "cx": 0.2, "cy": 0.5, "w": 0.2, "h": 0.3, "line": 1},
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.3, "line": 2},
        ]
        issues = check_overlap_and_size(boxes)
        assert issues == []

    def test_tiny_box_flagged(self):
        boxes = [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.01, "h": 0.01, "line": 1}]
        issues = check_overlap_and_size(boxes, min_relative_area=0.001)
        assert any("suspiciously tiny" in i for i in issues)

    def test_huge_box_flagged(self):
        boxes = [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.99, "h": 0.99, "line": 1}]
        issues = check_overlap_and_size(boxes, max_relative_area=0.9)
        assert any("suspiciously huge" in i for i in issues)

    def test_normal_boxes_no_diagnostics(self):
        boxes = [
            {"class_id": 0, "cx": 0.25, "cy": 0.25, "w": 0.2, "h": 0.2, "line": 1},
            {"class_id": 0, "cx": 0.75, "cy": 0.75, "w": 0.2, "h": 0.2, "line": 2},
        ]
        issues = check_overlap_and_size(boxes)
        assert issues == []

    def _write_fixture(self, tmp_path, name, w, h, label_lines):
        Image.new("RGB", (w, h), (0, 0, 0)).save(tmp_path / "images" / f"{name}.jpg")
        (tmp_path / "labels" / f"{name}.txt").write_text("\n".join(label_lines), encoding="utf-8")

    def test_full_directory_validation_integration(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        # one clean multi-panel image (one physical panel = one box)
        self._write_fixture(tmp_path, "clean", 1280, 720, [
            "0 0.1 0.1 0.15 0.15", "0 0.4 0.1 0.15 0.15", "0 0.7 0.1 0.15 0.15",
        ])
        # a zero-panel image (legitimate empty label file, e.g. no panel visible)
        self._write_fixture(tmp_path, "zero_panel", 1280, 720, [])
        # a malformed line
        self._write_fixture(tmp_path, "malformed", 1280, 720, ["0 bad data here"])
        # a near-duplicate-box mistake
        self._write_fixture(tmp_path, "dup_box", 1280, 720, [
            "0 0.5 0.5 0.2 0.2", "0 0.505 0.505 0.2 0.2",
        ])

        report = validate_representative_annotations(tmp_path / "images", tmp_path / "labels")
        assert report["total_images"] == 4
        assert report["images_with_zero_annotations_count"] == 1
        assert "zero_panel.jpg" in report["images_with_zero_annotations"]
        assert report["files_with_errors_count"] == 1  # malformed
        assert report["images_with_overlap_or_size_diagnostics_count"] == 1
        assert "dup_box.jpg" in report["overlap_and_size_diagnostics"]

    def test_zero_panel_image_is_not_an_error(self, tmp_path):
        """A legitimate empty label file (no panel visible / all fragments
        ambiguous per the partial-visibility policy) must validate cleanly,
        never be reported as a mechanical error."""
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        self._write_fixture(tmp_path, "empty_scene", 1280, 720, [])
        report = validate_representative_annotations(tmp_path / "images", tmp_path / "labels")
        assert report["files_with_errors_count"] == 0
        assert report["images_with_zero_annotations_count"] == 1

    def test_missing_label_file_is_reported_not_silently_skipped(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        Image.new("RGB", (640, 480), (0, 0, 0)).save(tmp_path / "images" / "no_label.jpg")
        report = validate_representative_annotations(tmp_path / "images", tmp_path / "labels")
        assert report["missing_label_count"] == 1
        assert "no_label.jpg" in report["missing_label_files"]


# ---------------------------------------------------------------------------
# Panel-count / resolution / aspect-ratio statistics
# ---------------------------------------------------------------------------


class TestPanelCountStats:
    def test_stats_from_panel_count_if_known(self):
        entries = [
            _make_entry("a", _flip_n_bits(0), panel_count_if_known=4),
            _make_entry("b", _flip_n_bits(0), panel_count_if_known=4),
            _make_entry("c", _flip_n_bits(0), panel_count_if_known=6),
            _make_entry("d", _flip_n_bits(0), panel_count_if_known=8),
        ]
        stats = compute_panel_count_stats(entries)
        assert stats["images_by_panel_count"] == {4: 2, 6: 1, 8: 1}
        assert stats["target_counts_covered"] == [4, 6, 8]
        assert stats["target_counts_missing"] == []

    def test_stats_from_label_files_require_human_verified_status(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "a.txt").write_text("0 0.1 0.1 0.1 0.1\n0 0.5 0.5 0.1 0.1", encoding="utf-8")
        (labels_dir / "b.txt").write_text("0 0.1 0.1 0.1 0.1", encoding="utf-8")
        entries = [
            SourceManifestEntry(
                image_id="img1", source_family_id="img1", original_filename="a.jpg",
                source_path="x", width=100, height=100, aspect_ratio=1.0, sha256="a" * 64,
                capture_source="first_party", scene_id="img1",
                annotation_status="annotated_human_verified",
            ),
            SourceManifestEntry(
                image_id="img2", source_family_id="img2", original_filename="b.jpg",
                source_path="x", width=100, height=100, aspect_ratio=1.0, sha256="b" * 64,
                capture_source="first_party", scene_id="img2",
                annotation_status="annotated_human_verified",
            ),
        ]
        stats = compute_panel_count_stats(entries, labels_dir=labels_dir)
        assert stats["images_by_panel_count"] == {1: 1, 2: 1}

    def test_unverified_label_stub_is_not_counted_as_zero_panels(self, tmp_path):
        """An empty placeholder .txt (written by --write-empty-label-stubs
        for a human to fill in) must NOT be read back as a verified
        zero-panel annotation just because the file exists on disk."""
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "a.txt").write_text("", encoding="utf-8")  # empty stub, not yet annotated
        entries = [
            SourceManifestEntry(
                image_id="img1", source_family_id="img1", original_filename="a.jpg",
                source_path="x", width=100, height=100, aspect_ratio=1.0, sha256="a" * 64,
                capture_source="first_party", scene_id="img1",
                annotation_status="pending",  # NOT verified
            ),
        ]
        stats = compute_panel_count_stats(entries, labels_dir=labels_dir)
        assert stats["images_with_known_panel_count"] == 0
        assert stats["images_by_panel_count"] == {}

    def test_images_without_known_count_excluded_not_assumed(self):
        entries = [
            _make_entry("a", _flip_n_bits(0), panel_count_if_known=None),
            _make_entry("b", _flip_n_bits(0), panel_count_if_known=4),
        ]
        stats = compute_panel_count_stats(entries)
        assert stats["images_with_known_panel_count"] == 1
        assert stats["images_total"] == 2

    def test_missing_target_counts_reported(self):
        entries = [_make_entry("a", _flip_n_bits(0), panel_count_if_known=4)]
        stats = compute_panel_count_stats(entries)
        assert stats["target_counts_missing"] == [6, 8]

    def test_resolution_and_aspect_stats(self):
        entries = [
            _make_entry("a", _flip_n_bits(0), w=573, h=1280),   # portrait
            _make_entry("b", _flip_n_bits(0), w=1280, h=573),   # landscape
            _make_entry("c", _flip_n_bits(0), w=640, h=640),    # square
        ]
        stats = compute_resolution_and_aspect_stats(entries)
        assert stats["count"] == 3
        assert stats["orientation_counts"]["portrait"] == 1
        assert stats["orientation_counts"]["landscape"] == 1
        assert stats["orientation_counts"]["square"] == 1

    def test_resolution_stats_empty_entries(self):
        assert compute_resolution_and_aspect_stats([]) == {"count": 0}


# ---------------------------------------------------------------------------
# Count-error evaluation protocol
# ---------------------------------------------------------------------------


class TestDuplicateLeakageReport:
    def test_no_duplicates_clean_report(self):
        entries = [_make_entry("a", _flip_n_bits(0), sha256="a" * 64),
                   _make_entry("b", _flip_n_bits(200), sha256="b" * 64)]
        report = compute_duplicate_leakage_report(entries, [], {"a": "train", "b": "test"})
        assert report["exact_duplicate_groups"] == 0
        assert report["cross_split_exact_duplicate_groups"] == 0
        assert report["cross_split_near_duplicate_pairs"] == 0

    def test_exact_duplicate_within_same_split_not_flagged_as_crossing(self):
        entries = [_make_entry("a", _flip_n_bits(0), sha256="dup" + "0" * 61),
                   _make_entry("b", _flip_n_bits(200), sha256="dup" + "0" * 61)]
        report = compute_duplicate_leakage_report(entries, [], {"a": "train", "b": "train"})
        assert report["exact_duplicate_groups"] == 1
        assert report["cross_split_exact_duplicate_groups"] == 0

    def test_exact_duplicate_crossing_splits_flagged(self):
        entries = [_make_entry("a", _flip_n_bits(0), sha256="dup" + "0" * 61),
                   _make_entry("b", _flip_n_bits(200), sha256="dup" + "0" * 61)]
        report = compute_duplicate_leakage_report(entries, [], {"a": "train", "b": "test"})
        assert report["cross_split_exact_duplicate_groups"] == 1

    def test_near_duplicate_crossing_splits_flagged(self):
        entries = [_make_entry("a", _flip_n_bits(0), sha256="a" * 64),
                   _make_entry("b", _flip_n_bits(3), sha256="b" * 64)]
        near_pairs = [{"a": "a", "b": "b", "dhash16_hamming": 3}]
        report = compute_duplicate_leakage_report(entries, near_pairs, {"a": "train", "b": "valid"})
        assert report["cross_split_near_duplicate_pairs"] == 1
        report_same_split = compute_duplicate_leakage_report(entries, near_pairs, {"a": "train", "b": "train"})
        assert report_same_split["cross_split_near_duplicate_pairs"] == 0


class TestReadinessChecklist:
    def _minimal_entries(self, n=6, provenance="first_party", annotation="pending"):
        entries = []
        for i in range(n):
            e = _make_entry(f"img{i}", _flip_n_bits(0), sha256=f"e{i}" + "0" * 62)
            e.provenance_status = provenance
            e.annotation_status = annotation
            e.scene_id = f"scene{i}"
            entries.append(e)
        return entries

    def test_small_unannotated_dataset_is_not_ready(self):
        entries = self._minimal_entries(n=6)
        image_to_split = {e.image_id: "train" for e in entries}
        split_check = {"ok": True, "crossing_scenes": {}}
        dup_report = compute_duplicate_leakage_report(entries, [], image_to_split)
        panel_stats = compute_panel_count_stats(entries)
        checklist = build_readiness_checklist(
            entries=entries, image_to_split=image_to_split, split_check=split_check,
            duplicate_report=dup_report, panel_count_stats=panel_stats, annotation_report=None,
        )
        assert checklist["ready_for_phase_14c"] is False
        assert checklist["num_images"] == 6
        assert any("100" in r for r in checklist["blocking_reasons"])
        assert any("annotations" in r for r in checklist["blocking_reasons"])

    def test_scene_crossing_blocks_readiness(self):
        entries = self._minimal_entries(n=120, annotation="annotated_human_verified")
        image_to_split = {e.image_id: "train" for e in entries}
        bad_split_check = {"ok": False, "crossing_scenes": {"scene0": ["train", "test"]}}
        dup_report = compute_duplicate_leakage_report(entries, [], image_to_split)
        panel_stats = {"target_counts_covered": [4, 6, 8]}
        checklist = build_readiness_checklist(
            entries=entries, image_to_split=image_to_split, split_check=bad_split_check,
            duplicate_report=dup_report, panel_count_stats=panel_stats, annotation_report={"ready_for_next_stage": True},
        )
        assert checklist["ready_for_phase_14c"] is False
        assert checklist["train_valid_test_scene_isolated"] is False

    def test_missing_target_panel_counts_blocks_readiness(self):
        entries = self._minimal_entries(n=120, annotation="annotated_human_verified")
        image_to_split = {e.image_id: "train" for e in entries}
        split_check = {"ok": True, "crossing_scenes": {}}
        dup_report = compute_duplicate_leakage_report(entries, [], image_to_split)
        panel_stats = {"target_counts_covered": [4, 6]}  # missing 8
        checklist = build_readiness_checklist(
            entries=entries, image_to_split=image_to_split, split_check=split_check,
            duplicate_report=dup_report, panel_count_stats=panel_stats, annotation_report={"ready_for_next_stage": True},
        )
        assert checklist["target_4_6_8_panel_scenes_represented"] is False
        assert checklist["ready_for_phase_14c"] is False

    def test_large_foreground_and_touching_panel_coverage_never_guessed(self):
        entries = self._minimal_entries(n=120, annotation="annotated_human_verified")
        image_to_split = {e.image_id: "train" for e in entries}
        split_check = {"ok": True, "crossing_scenes": {}}
        dup_report = compute_duplicate_leakage_report(entries, [], image_to_split)
        panel_stats = {"target_counts_covered": [4, 6, 8]}
        checklist = build_readiness_checklist(
            entries=entries, image_to_split=image_to_split, split_check=split_check,
            duplicate_report=dup_report, panel_count_stats=panel_stats, annotation_report={"ready_for_next_stage": True},
        )
        # these require human confirmation — must never silently become True
        assert checklist["large_foreground_panels_represented"] == "unknown_needs_manual_tagging"
        assert checklist["touching_grouped_panels_represented"] == "unknown_needs_manual_tagging"
        assert checklist["ready_for_phase_14c"] is False  # blocked by the two unknowns

    def test_no_test_split_blocks_readiness(self):
        entries = self._minimal_entries(n=120, annotation="annotated_human_verified")
        image_to_split = {e.image_id: "train" for e in entries}  # no test split at all
        split_check = {"ok": True, "crossing_scenes": {}}
        dup_report = compute_duplicate_leakage_report(entries, [], image_to_split)
        panel_stats = {"target_counts_covered": [4, 6, 8]}
        checklist = build_readiness_checklist(
            entries=entries, image_to_split=image_to_split, split_check=split_check,
            duplicate_report=dup_report, panel_count_stats=panel_stats, annotation_report={"ready_for_next_stage": True},
        )
        assert checklist["genuine_independent_holdout"] is False
        assert checklist["ready_for_phase_14c"] is False


class TestCountErrorEvaluation:
    def test_exact_match(self):
        stats = compute_count_error_stats({"a": 4, "b": 6}, {"a": 4, "b": 6})
        assert stats["exact_count_accuracy"] == 1.0
        assert stats["mean_absolute_count_error"] == 0.0

    def test_overcount_and_undercount(self):
        stats = compute_count_error_stats({"a": 5, "b": 5}, {"a": 4, "b": 8})
        assert stats["overcount_rate"] == 0.5
        assert stats["undercount_rate"] == 0.5
        assert stats["exact_count_accuracy"] == 0.0
        assert stats["mean_absolute_count_error"] == pytest.approx((1 + 3) / 2)

    def test_signed_mean_error_shows_bias(self):
        # consistently undercounting by 1
        stats = compute_count_error_stats({"a": 3, "b": 5}, {"a": 4, "b": 6})
        assert stats["mean_count_error"] == pytest.approx(-1.0)

    def test_missing_predictions_or_ground_truth_not_fabricated(self):
        stats = compute_count_error_stats({"a": 4, "c": 8}, {"a": 4, "b": 6})
        assert stats["num_images_scored"] == 1  # only "a" is in both
        assert stats["num_predictions_without_ground_truth"] == 1  # "c"
        assert stats["num_ground_truth_without_prediction"] == 1  # "b"

    def test_empty_inputs(self):
        stats = compute_count_error_stats({}, {})
        assert stats["num_images_scored"] == 0
        assert stats["exact_count_accuracy"] is None
        assert stats["mean_absolute_count_error"] is None
