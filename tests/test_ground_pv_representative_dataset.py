"""tests/test_ground_pv_representative_dataset.py — Phase 14B / 14B.1 tooling
for the representative deployment-domain panel dataset (manifest, scene
grouping, scene-isolated splitting, annotation validation, batch import,
difficult-case tags, and the readiness checklist / "Final Decision Rule").

Grouping/split tests construct `SourceManifestEntry` objects directly with
controlled `dhash16` hex strings (known Hamming distances) rather than real
near-duplicate images, so the union-find/threshold logic is tested exactly
and deterministically without depending on PIL's actual perceptual-hash
output for handcrafted test images. Image-metadata-extraction and batch-
import tests use real small PNGs written to a tmp_path (fast, no network,
no GPU).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from training.detection.ground_pv.representative_dataset import (
    DEFAULT_SPLIT_RATIOS,
    DIFFICULT_CASE_TAGS,
    SourceManifestEntry,
    apply_difficult_case_tags,
    assign_scene_isolated_splits,
    audit_panel_count_gt_disagreements,
    build_readiness_checklist,
    build_source_manifest,
    check_overlap_and_size,
    compute_count_error_stats,
    compute_duplicate_leakage_report,
    compute_panel_count_stats,
    compute_resolution_and_aspect_stats,
    extract_image_metadata,
    group_into_scenes,
    import_batch_into_manifest,
    load_source_manifest,
    save_source_manifest,
    save_split_manifest,
    sync_panel_count_gt_from_annotations,
    validate_representative_annotations,
    verify_no_scene_crosses_splits,
)


def _make_entry(image_id, dhash16_hex, sha256="s" + "0" * 63, w=1280, h=720, **kw):
    defaults = dict(
        image_id=image_id, source_family_id=image_id, original_filename=f"{image_id}.jpg",
        source_path=f"/x/{image_id}.jpg", width=w, height=h, aspect_ratio=round(w / h, 4),
        sha256=sha256, source_type="first_party", scene_id=image_id,
        provenance_status="first_party", dhash16=dhash16_hex,
    )
    defaults.update(kw)
    return SourceManifestEntry(**defaults)


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
        entries, corrupt = build_source_manifest(tmp_path, source_type="first_party")
        assert len(entries) == 3
        assert corrupt == []
        assert all(e.scene_id == e.image_id for e in entries)  # singleton until grouped
        assert all(e.source_family_id == e.image_id for e in entries)
        assert all(e.annotation_status == "pending" for e in entries)
        assert all(e.provenance_status == "first_party" for e in entries)
        assert all(e.scene_status == "needs_review" for e in entries)

    def test_build_source_manifest_reports_corrupt_without_dropping_silently(self, tmp_path):
        Image.new("RGB", (640, 480), (1, 1, 1)).save(tmp_path / "good.jpg")
        (tmp_path / "corrupt.jpg").write_bytes(b"garbage")
        entries, corrupt = build_source_manifest(tmp_path, source_type="first_party")
        assert len(entries) == 1
        assert len(corrupt) == 1
        assert "corrupt.jpg" in corrupt[0]["path"]

    def test_build_source_manifest_rejects_invalid_source_type(self, tmp_path):
        with pytest.raises(ValueError):
            build_source_manifest(tmp_path, source_type="made_up")

    def test_manifest_save_load_roundtrip(self, tmp_path):
        Image.new("RGB", (640, 480), (5, 5, 5)).save(tmp_path / "img.jpg")
        entries, _ = build_source_manifest(tmp_path, source_type="first_party")
        out = tmp_path / "manifest.json"
        save_source_manifest(entries, out)
        loaded = load_source_manifest(out)
        assert len(loaded) == 1
        assert loaded[0].image_id == entries[0].image_id
        assert loaded[0].sha256 == entries[0].sha256
        assert loaded[0].dhash16 == entries[0].dhash16  # regression: dhash16 was silently dropped before
        assert loaded[0].source_type == "first_party"
        assert loaded[0].scene_status == "needs_review"


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
        assert grouped[0].scene_status == "auto_grouped"

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
        assert grouped[0].scene_status == "auto_grouped"

    def test_dissimilar_images_stay_separate_scenes(self):
        # Hamming distance 128 (half the bits differ) -> clearly different scenes
        entries = [
            _make_entry("img1", _flip_n_bits(0), sha256="a" * 64),
            _make_entry("img2", _flip_n_bits(128), sha256="b" * 64),
        ]
        grouped, near_pairs = group_into_scenes(entries)
        assert grouped[0].scene_id != grouped[1].scene_id
        assert near_pairs == []
        assert grouped[0].scene_status == "needs_review"
        assert grouped[1].scene_status == "needs_review"

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

    def test_single_entry_is_its_own_scene_needs_review(self):
        entries = [_make_entry("img1", _flip_n_bits(0))]
        grouped, near_pairs = group_into_scenes(entries)
        assert len(grouped) == 1
        assert near_pairs == []
        assert grouped[0].scene_status == "needs_review"

    def test_manually_assigned_scenes_are_seeded_and_preserved(self):
        # Two images explicitly assigned the same scene by a human importer,
        # with dHashes far apart (would NOT auto-group) — the manual
        # assignment must still hold.
        entries = [
            _make_entry("img1", _flip_n_bits(0), sha256="a" * 64,
                        scene_id="human_scene", scene_status="manually_assigned"),
            _make_entry("img2", _flip_n_bits(128), sha256="b" * 64,
                        scene_id="human_scene", scene_status="manually_assigned"),
        ]
        grouped, _ = group_into_scenes(entries)
        assert grouped[0].scene_id == grouped[1].scene_id
        assert grouped[0].scene_status == "manually_assigned"
        assert grouped[1].scene_status == "manually_assigned"

    def test_manual_assignment_status_propagates_to_auto_merged_additions(self):
        # img1+img2 manually assigned; img3 is a near-duplicate of img1 but
        # was NOT manually assigned — once merged, the whole group should
        # read as manually_assigned (a human confirmed part of it).
        entries = [
            _make_entry("img1", _flip_n_bits(0), sha256="a" * 64,
                        scene_id="human_scene", scene_status="manually_assigned"),
            _make_entry("img2", _flip_n_bits(200), sha256="b" * 64,
                        scene_id="human_scene", scene_status="manually_assigned"),
            _make_entry("img3", _flip_n_bits(3), sha256="c" * 64),  # near-dup of img1, unassigned
        ]
        grouped, _ = group_into_scenes(entries)
        scene_ids = {e.scene_id for e in grouped}
        assert len(scene_ids) == 1
        assert all(e.scene_status == "manually_assigned" for e in grouped)


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
# panel_count_gt sync + disagreement audit
# ---------------------------------------------------------------------------


class TestPanelCountGtSync:
    def test_sync_sets_panel_count_gt_from_verified_boxes(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "a.txt").write_text("0 0.1 0.1 0.1 0.1\n0 0.5 0.5 0.1 0.1", encoding="utf-8")
        entries = [
            SourceManifestEntry(
                image_id="img1", source_family_id="img1", original_filename="a.jpg",
                source_path="x", width=100, height=100, aspect_ratio=1.0, sha256="a" * 64,
                source_type="first_party", scene_id="img1",
                annotation_status="annotated_human_verified", panel_count_gt=None,
            ),
        ]
        disagreements = sync_panel_count_gt_from_annotations(entries, labels_dir)
        assert disagreements == []
        assert entries[0].panel_count_gt == 2

    def test_sync_never_touches_unverified_entries(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "a.txt").write_text("0 0.1 0.1 0.1 0.1", encoding="utf-8")
        entries = [
            SourceManifestEntry(
                image_id="img1", source_family_id="img1", original_filename="a.jpg",
                source_path="x", width=100, height=100, aspect_ratio=1.0, sha256="a" * 64,
                source_type="first_party", scene_id="img1",
                annotation_status="pending", panel_count_gt=None,  # NOT verified
            ),
        ]
        sync_panel_count_gt_from_annotations(entries, labels_dir)
        assert entries[0].panel_count_gt is None  # untouched

    def test_sync_reports_disagreement_with_stale_stored_value(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "a.txt").write_text("0 0.1 0.1 0.1 0.1\n0 0.5 0.5 0.1 0.1", encoding="utf-8")
        entries = [
            SourceManifestEntry(
                image_id="img1", source_family_id="img1", original_filename="a.jpg",
                source_path="x", width=100, height=100, aspect_ratio=1.0, sha256="a" * 64,
                source_type="first_party", scene_id="img1",
                annotation_status="annotated_human_verified", panel_count_gt=99,  # stale/wrong
            ),
        ]
        disagreements = sync_panel_count_gt_from_annotations(entries, labels_dir)
        assert len(disagreements) == 1
        assert disagreements[0]["stored_panel_count_gt"] == 99
        assert disagreements[0]["actual_verified_box_count"] == 2
        assert entries[0].panel_count_gt == 2  # corrected

    def test_audit_disagreements_does_not_mutate(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "a.txt").write_text("0 0.1 0.1 0.1 0.1", encoding="utf-8")
        entries = [
            SourceManifestEntry(
                image_id="img1", source_family_id="img1", original_filename="a.jpg",
                source_path="x", width=100, height=100, aspect_ratio=1.0, sha256="a" * 64,
                source_type="first_party", scene_id="img1",
                annotation_status="annotated_human_verified", panel_count_gt=5,  # wrong
            ),
        ]
        disagreements = audit_panel_count_gt_disagreements(entries, labels_dir)
        assert len(disagreements) == 1
        assert entries[0].panel_count_gt == 5  # NOT mutated by the read-only audit


# ---------------------------------------------------------------------------
# Difficult-case tags
# ---------------------------------------------------------------------------


class TestDifficultCaseTags:
    def test_apply_tags_to_matching_scene(self):
        entries = [
            _make_entry("img1", _flip_n_bits(0), scene_id="scene0"),
            _make_entry("img2", _flip_n_bits(0), scene_id="scene0"),
            _make_entry("img3", _flip_n_bits(0), scene_id="scene1"),
        ]
        apply_difficult_case_tags(entries, {"scene0": ["large_foreground_panel", "strong_perspective"]})
        assert set(entries[0].difficult_case_tags) == {"large_foreground_panel", "strong_perspective"}
        assert set(entries[1].difficult_case_tags) == {"large_foreground_panel", "strong_perspective"}
        assert entries[2].difficult_case_tags == []

    def test_unrecognized_tag_rejected(self):
        entries = [_make_entry("img1", _flip_n_bits(0), scene_id="scene0")]
        with pytest.raises(ValueError):
            apply_difficult_case_tags(entries, {"scene0": ["not_a_real_tag"]})
        assert entries[0].difficult_case_tags == []  # untouched on failure

    def test_tags_merge_not_overwrite(self):
        entries = [_make_entry("img1", _flip_n_bits(0), scene_id="scene0",
                               difficult_case_tags=["occlusion"])]
        apply_difficult_case_tags(entries, {"scene0": ["low_light"]})
        assert set(entries[0].difficult_case_tags) == {"occlusion", "low_light"}

    def test_all_documented_tags_are_valid(self):
        for tag in DIFFICULT_CASE_TAGS:
            entries = [_make_entry("img1", _flip_n_bits(0), scene_id="scene0")]
            apply_difficult_case_tags(entries, {"scene0": [tag]})  # must not raise


# ---------------------------------------------------------------------------
# Panel-count / resolution / aspect-ratio statistics
# ---------------------------------------------------------------------------


class TestPanelCountStats:
    def test_stats_from_panel_count_gt_when_verified(self):
        entries = [
            _make_entry("a", _flip_n_bits(0), panel_count_gt=4, annotation_status="annotated_human_verified"),
            _make_entry("b", _flip_n_bits(0), panel_count_gt=4, annotation_status="annotated_human_verified"),
            _make_entry("c", _flip_n_bits(0), panel_count_gt=6, annotation_status="annotated_human_verified"),
            _make_entry("d", _flip_n_bits(0), panel_count_gt=8, annotation_status="annotated_human_verified"),
        ]
        stats = compute_panel_count_stats(entries)
        assert stats["images_by_panel_count"] == {4: 2, 6: 1, 8: 1}
        assert stats["target_counts_covered"] == [4, 6, 8]
        assert stats["target_counts_missing"] == []

    def test_unverified_panel_count_gt_never_counted(self):
        """A panel_count_gt value on an entry that is NOT
        annotated_human_verified must be ignored entirely (e.g. a stale
        value left over from before annotation_status was reset to
        pending/rejected)."""
        entries = [_make_entry("a", _flip_n_bits(0), panel_count_gt=4, annotation_status="pending")]
        stats = compute_panel_count_stats(entries)
        assert stats["images_with_known_panel_count"] == 0

    def test_stats_from_label_files_require_human_verified_status(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "a.txt").write_text("0 0.1 0.1 0.1 0.1\n0 0.5 0.5 0.1 0.1", encoding="utf-8")
        (labels_dir / "b.txt").write_text("0 0.1 0.1 0.1 0.1", encoding="utf-8")
        entries = [
            SourceManifestEntry(
                image_id="img1", source_family_id="img1", original_filename="a.jpg",
                source_path="x", width=100, height=100, aspect_ratio=1.0, sha256="a" * 64,
                source_type="first_party", scene_id="img1",
                annotation_status="annotated_human_verified",
            ),
            SourceManifestEntry(
                image_id="img2", source_family_id="img2", original_filename="b.jpg",
                source_path="x", width=100, height=100, aspect_ratio=1.0, sha256="b" * 64,
                source_type="first_party", scene_id="img2",
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
                source_type="first_party", scene_id="img1",
                annotation_status="pending",  # NOT verified
            ),
        ]
        stats = compute_panel_count_stats(entries, labels_dir=labels_dir)
        assert stats["images_with_known_panel_count"] == 0
        assert stats["images_by_panel_count"] == {}

    def test_images_without_known_count_excluded_not_assumed(self):
        entries = [
            _make_entry("a", _flip_n_bits(0), panel_count_gt=None, annotation_status="annotated_human_verified"),
            _make_entry("b", _flip_n_bits(0), panel_count_gt=4, annotation_status="annotated_human_verified"),
        ]
        stats = compute_panel_count_stats(entries)
        assert stats["images_with_known_panel_count"] == 1
        assert stats["images_total"] == 2

    def test_missing_target_counts_reported(self):
        entries = [_make_entry("a", _flip_n_bits(0), panel_count_gt=4, annotation_status="annotated_human_verified")]
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
# Duplicate / leakage report
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


# ---------------------------------------------------------------------------
# Readiness checklist (Final Decision Rule, tightened for Phase 14B.1)
# ---------------------------------------------------------------------------


class TestReadinessChecklist:
    def _verified_scene(self, idx, panel_count, tags=None, split="train"):
        """One verified, non-reference, non-needs_review scene (1 image)."""
        e = _make_entry(
            f"img{idx}", _flip_n_bits(0), sha256=f"e{idx}" + "0" * 62,
            annotation_status="annotated_human_verified", panel_count_gt=panel_count,
            scene_status="auto_grouped", difficult_case_tags=list(tags or []),
        )
        e.scene_id = f"scene{idx}"
        return e, split

    def _build(self, n_each=34):
        """Build enough verified scenes to plausibly hit the 100 minimum,
        with 4/6/8 coverage and both difficult-case tags represented."""
        entries = []
        splits = {}
        idx = 0
        counts = [4, 6, 8]
        tags_by_i = [["large_foreground_panel"], ["touching_panels"], []]
        for i in range(n_each):
            for j, count in enumerate(counts):
                e, split = self._verified_scene(idx, count, tags=tags_by_i[j] if i == 0 else None)
                entries.append(e)
                splits[e.image_id] = split
                idx += 1
        # add a genuine test-split scene too
        e, _ = self._verified_scene(idx, 4, split="test")
        entries.append(e)
        splits[e.image_id] = "test"
        return entries, splits

    def test_small_unannotated_dataset_is_not_ready(self):
        entries = [_make_entry(f"img{i}", _flip_n_bits(0)) for i in range(6)]
        image_to_split = {e.image_id: "train" for e in entries}
        split_check = {"ok": True, "crossing_scenes": {}}
        dup_report = compute_duplicate_leakage_report(entries, [], image_to_split)
        panel_stats = compute_panel_count_stats(entries)
        checklist = build_readiness_checklist(
            entries=entries, image_to_split=image_to_split, split_check=split_check,
            duplicate_report=dup_report, panel_count_stats=panel_stats, annotation_report=None,
        )
        assert checklist["ready_for_phase_14c"] is False
        assert checklist["verified_scene_count"] == 0
        assert any("verified independent scene" in r for r in checklist["blocking_reasons"])
        assert any("annotations" in r for r in checklist["blocking_reasons"])

    def test_fully_populated_dataset_passes_every_gate(self):
        entries, image_to_split = self._build()
        split_check = verify_no_scene_crosses_splits(entries, image_to_split)
        dup_report = compute_duplicate_leakage_report(entries, [], image_to_split)
        panel_stats = compute_panel_count_stats(entries)
        checklist = build_readiness_checklist(
            entries=entries, image_to_split=image_to_split, split_check=split_check,
            duplicate_report=dup_report, panel_count_stats=panel_stats,
            annotation_report={"ready_for_next_stage": True},
        )
        assert checklist["verified_scene_count"] >= 100
        assert checklist["has_verified_4_panel_scene"] is True
        assert checklist["has_verified_6_panel_scene"] is True
        assert checklist["has_verified_8_panel_scene"] is True
        assert checklist["large_foreground_panels_represented"] is True
        assert checklist["touching_grouped_panels_represented"] is True
        assert checklist["genuine_independent_holdout"] is True
        assert checklist["ready_for_phase_14c"] is True
        assert checklist["blocking_reasons"] == []

    def test_needs_review_scenes_never_count_toward_the_minimum(self):
        entries, image_to_split = self._build()
        # flip everything to needs_review — none of it should count anymore
        for e in entries:
            e.scene_status = "needs_review"
        split_check = verify_no_scene_crosses_splits(entries, image_to_split)
        dup_report = compute_duplicate_leakage_report(entries, [], image_to_split)
        panel_stats = compute_panel_count_stats(entries)
        checklist = build_readiness_checklist(
            entries=entries, image_to_split=image_to_split, split_check=split_check,
            duplicate_report=dup_report, panel_count_stats=panel_stats,
            annotation_report={"ready_for_next_stage": True},
        )
        assert checklist["verified_scene_count"] == 0
        assert checklist["ready_for_phase_14c"] is False

    def test_reference_split_scenes_excluded_from_the_minimum(self):
        """The six-owner-photo qualitative reference set must never count
        toward the >=100-scene training-readiness requirement, even if
        (hypothetically) annotated and scene-resolved."""
        entries, image_to_split = self._build()
        for e in entries:
            image_to_split[e.image_id] = "reference"
        split_check = {"ok": True, "crossing_scenes": {}}
        dup_report = compute_duplicate_leakage_report(entries, [], image_to_split)
        panel_stats = compute_panel_count_stats(entries)
        checklist = build_readiness_checklist(
            entries=entries, image_to_split=image_to_split, split_check=split_check,
            duplicate_report=dup_report, panel_count_stats=panel_stats,
            annotation_report={"ready_for_next_stage": True},
        )
        assert checklist["verified_scene_count"] == 0
        assert checklist["ready_for_phase_14c"] is False

    def test_scene_crossing_blocks_readiness(self):
        entries, image_to_split = self._build()
        bad_split_check = {"ok": False, "crossing_scenes": {"scene0": ["train", "test"]}}
        dup_report = compute_duplicate_leakage_report(entries, [], image_to_split)
        panel_stats = compute_panel_count_stats(entries)
        checklist = build_readiness_checklist(
            entries=entries, image_to_split=image_to_split, split_check=bad_split_check,
            duplicate_report=dup_report, panel_count_stats=panel_stats,
            annotation_report={"ready_for_next_stage": True},
        )
        assert checklist["ready_for_phase_14c"] is False
        assert checklist["train_valid_test_scene_isolated"] is False

    def test_missing_one_of_4_6_8_blocks_readiness(self):
        entries, image_to_split = self._build()
        # drop every verified 8-panel scene
        for e in entries:
            if e.panel_count_gt == 8:
                e.panel_count_gt = 4
        split_check = verify_no_scene_crosses_splits(entries, image_to_split)
        dup_report = compute_duplicate_leakage_report(entries, [], image_to_split)
        panel_stats = compute_panel_count_stats(entries)
        checklist = build_readiness_checklist(
            entries=entries, image_to_split=image_to_split, split_check=split_check,
            duplicate_report=dup_report, panel_count_stats=panel_stats,
            annotation_report={"ready_for_next_stage": True},
        )
        assert checklist["has_verified_8_panel_scene"] is False
        assert checklist["ready_for_phase_14c"] is False

    def test_difficult_case_coverage_is_a_real_boolean_not_a_guess(self):
        entries, image_to_split = self._build()
        for e in entries:
            e.difficult_case_tags = []  # strip all tags
        split_check = verify_no_scene_crosses_splits(entries, image_to_split)
        dup_report = compute_duplicate_leakage_report(entries, [], image_to_split)
        panel_stats = compute_panel_count_stats(entries)
        checklist = build_readiness_checklist(
            entries=entries, image_to_split=image_to_split, split_check=split_check,
            duplicate_report=dup_report, panel_count_stats=panel_stats,
            annotation_report={"ready_for_next_stage": True},
        )
        # must be a real False, never the old "unknown_needs_manual_tagging" sentinel
        assert checklist["large_foreground_panels_represented"] is False
        assert checklist["touching_grouped_panels_represented"] is False
        assert checklist["ready_for_phase_14c"] is False

    def test_no_test_split_blocks_readiness(self):
        entries, image_to_split = self._build()
        for image_id in list(image_to_split):
            if image_to_split[image_id] == "test":
                image_to_split[image_id] = "train"
        split_check = verify_no_scene_crosses_splits(entries, image_to_split)
        dup_report = compute_duplicate_leakage_report(entries, [], image_to_split)
        panel_stats = compute_panel_count_stats(entries)
        checklist = build_readiness_checklist(
            entries=entries, image_to_split=image_to_split, split_check=split_check,
            duplicate_report=dup_report, panel_count_stats=panel_stats,
            annotation_report={"ready_for_next_stage": True},
        )
        assert checklist["genuine_independent_holdout"] is False
        assert checklist["ready_for_phase_14c"] is False


# ---------------------------------------------------------------------------
# Count-error evaluation protocol
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Batch import (scripts/import_first_party_panels.py's engine)
# ---------------------------------------------------------------------------


class TestBatchImport:
    def _source_dir_with_images(self, tmp_path, names_colors):
        src = tmp_path / "source"
        src.mkdir(parents=True, exist_ok=True)
        for name, color in names_colors:
            Image.new("RGB", (640, 480), color).save(src / name)
        return src

    def test_first_batch_assigns_ids_from_one(self, tmp_path):
        src = self._source_dir_with_images(tmp_path, [("a.jpg", (1, 1, 1)), ("b.jpg", (2, 2, 2))])
        raw_root = tmp_path / "raw"
        result = import_batch_into_manifest(
            src, existing_entries=[], raw_dest_root=raw_root, batch_id="batch_001",
        )
        assert len(result.imported) == 2
        assert {e.image_id for e in result.imported} == {"img_0001", "img_0002"}
        assert result.rejected_duplicates == []
        assert result.corrupt == []

    def test_deterministic_ids_continue_across_batches(self, tmp_path):
        src1 = self._source_dir_with_images(tmp_path, [("a.jpg", (1, 1, 1))])
        raw_root = tmp_path / "raw"
        r1 = import_batch_into_manifest(src1, existing_entries=[], raw_dest_root=raw_root, batch_id="batch_001")
        src2 = self._source_dir_with_images(tmp_path / "src2", [("c.jpg", (3, 3, 3))])
        r2 = import_batch_into_manifest(
            src2, existing_entries=r1.all_entries, raw_dest_root=raw_root, batch_id="batch_002",
        )
        assert r2.imported[0].image_id == "img_0002"  # continues from batch_001's img_0001

    def test_exact_duplicate_within_batch_rejected(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        Image.new("RGB", (640, 480), (7, 7, 7)).save(src / "a.jpg")
        Image.new("RGB", (640, 480), (7, 7, 7)).save(src / "a_copy.jpg")  # byte-identical content
        raw_root = tmp_path / "raw"
        result = import_batch_into_manifest(src, existing_entries=[], raw_dest_root=raw_root, batch_id="batch_001")
        assert len(result.imported) == 1
        assert len(result.rejected_duplicates) == 1
        assert "same import batch" in result.rejected_duplicates[0]["reason"]

    def test_exact_duplicate_against_existing_manifest_rejected(self, tmp_path):
        src1 = self._source_dir_with_images(tmp_path, [("a.jpg", (9, 9, 9))])
        raw_root = tmp_path / "raw"
        r1 = import_batch_into_manifest(src1, existing_entries=[], raw_dest_root=raw_root, batch_id="batch_001")

        src2 = tmp_path / "src2"
        src2.mkdir()
        Image.new("RGB", (640, 480), (9, 9, 9)).save(src2 / "same_content_different_name.jpg")
        r2 = import_batch_into_manifest(
            src2, existing_entries=r1.all_entries, raw_dest_root=raw_root, batch_id="batch_002",
        )
        assert len(r2.imported) == 0
        assert len(r2.rejected_duplicates) == 1
        assert "already-imported" in r2.rejected_duplicates[0]["reason"]

    def test_originals_never_modified(self, tmp_path):
        src = self._source_dir_with_images(tmp_path, [("a.jpg", (4, 4, 4))])
        original_bytes = (src / "a.jpg").read_bytes()
        raw_root = tmp_path / "raw"
        import_batch_into_manifest(src, existing_entries=[], raw_dest_root=raw_root, batch_id="batch_001")
        assert (src / "a.jpg").read_bytes() == original_bytes

    def test_copy_never_silently_overwrites_a_different_file(self, tmp_path):
        src = self._source_dir_with_images(tmp_path, [("a.jpg", (5, 5, 5))])
        raw_root = tmp_path / "raw"
        (raw_root / "batch_001").mkdir(parents=True)
        # a DIFFERENT file already occupies the destination name
        Image.new("RGB", (10, 10), (255, 0, 0)).save(raw_root / "batch_001" / "a.jpg")
        result = import_batch_into_manifest(src, existing_entries=[], raw_dest_root=raw_root, batch_id="batch_001")
        # the pre-existing different file must survive untouched
        assert Image.open(raw_root / "batch_001" / "a.jpg").size == (10, 10)
        # the new image was copied under a disambiguated name instead
        assert result.imported[0].relative_path == "batch_001/a__dup2.jpg"

    def test_no_format_conversion_on_ingestion(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        Image.new("RGB", (640, 480), (6, 6, 6)).save(src / "a.png")
        raw_root = tmp_path / "raw"
        result = import_batch_into_manifest(src, existing_entries=[], raw_dest_root=raw_root, batch_id="batch_001")
        dest = raw_root / "batch_001" / "a.png"
        assert dest.is_file()
        assert Image.open(dest).format == "PNG"  # not silently re-saved as JPEG

    def test_default_scene_assignment_is_needs_review(self, tmp_path):
        src = self._source_dir_with_images(tmp_path, [("a.jpg", (1, 2, 3))])
        raw_root = tmp_path / "raw"
        result = import_batch_into_manifest(src, existing_entries=[], raw_dest_root=raw_root, batch_id="batch_001")
        assert result.imported[0].scene_status == "needs_review"
        assert result.imported[0].scene_id == result.imported[0].image_id  # singleton

    def test_explicit_scene_id_for_whole_batch(self, tmp_path):
        src = self._source_dir_with_images(tmp_path, [("a.jpg", (1, 2, 3)), ("b.jpg", (4, 5, 6))])
        raw_root = tmp_path / "raw"
        result = import_batch_into_manifest(
            src, existing_entries=[], raw_dest_root=raw_root, batch_id="batch_001",
            scene_id_for_batch="scene_005",
        )
        assert all(e.scene_id == "scene_005" for e in result.imported)
        assert all(e.scene_status == "manually_assigned" for e in result.imported)

    def test_explicit_scene_map_per_file(self, tmp_path):
        src = self._source_dir_with_images(tmp_path, [("a.jpg", (1, 2, 3)), ("b.jpg", (4, 5, 6))])
        raw_root = tmp_path / "raw"
        result = import_batch_into_manifest(
            src, existing_entries=[], raw_dest_root=raw_root, batch_id="batch_001",
            scene_map={"a.jpg": "scene_001", "b.jpg": "scene_002"},
        )
        by_name = {e.original_filename: e for e in result.imported}
        assert by_name["a.jpg"].scene_id == "scene_001"
        assert by_name["b.jpg"].scene_id == "scene_002"
        assert by_name["a.jpg"].scene_status == "manually_assigned"

    def test_scene_id_and_scene_map_mutually_exclusive(self, tmp_path):
        src = self._source_dir_with_images(tmp_path, [("a.jpg", (1, 2, 3))])
        raw_root = tmp_path / "raw"
        with pytest.raises(ValueError):
            import_batch_into_manifest(
                src, existing_entries=[], raw_dest_root=raw_root, batch_id="batch_001",
                scene_id_for_batch="scene_005", scene_map={"a.jpg": "scene_001"},
            )

    def test_batch_metadata_recorded(self, tmp_path):
        src = self._source_dir_with_images(tmp_path, [("a.jpg", (1, 2, 3))])
        raw_root = tmp_path / "raw"
        result = import_batch_into_manifest(
            src, existing_entries=[], raw_dest_root=raw_root, batch_id="batch_007",
            camera_id="cam_A", capture_timestamp="2026-09-11T10:00:00",
        )
        e = result.imported[0]
        assert e.capture_batch_id == "batch_007"
        assert e.camera_id == "cam_A"
        assert e.capture_timestamp == "2026-09-11T10:00:00"
        assert e.annotation_status == "pending"  # never pre-annotated on import

    def test_corrupt_file_reported_not_silently_dropped(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        Image.new("RGB", (640, 480), (1, 1, 1)).save(src / "good.jpg")
        (src / "bad.jpg").write_bytes(b"not an image")
        raw_root = tmp_path / "raw"
        result = import_batch_into_manifest(src, existing_entries=[], raw_dest_root=raw_root, batch_id="batch_001")
        assert len(result.imported) == 1
        assert len(result.corrupt) == 1

    def test_supports_jpg_png_webp(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        Image.new("RGB", (100, 100), (1, 1, 1)).save(src / "a.jpg")
        Image.new("RGB", (100, 100), (2, 2, 2)).save(src / "b.png")
        Image.new("RGB", (100, 100), (3, 3, 3)).save(src / "c.webp")
        raw_root = tmp_path / "raw"
        result = import_batch_into_manifest(src, existing_entries=[], raw_dest_root=raw_root, batch_id="batch_001")
        assert len(result.imported) == 3
