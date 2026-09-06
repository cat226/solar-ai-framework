"""Tests for training/detection/prepare_mendeley_panel_dataset.py.

Synthetic fixtures built directly to the raw Mendeley release's own
4-class format (bird_drop=0, cracked=1, dusty=2, panel=3) - no real
dataset files are needed to exercise the audit/conversion logic.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from training.detection.prepare_mendeley_panel_dataset import (
    PANEL_CLASS_ID,
    TARGET_CLASS_ID,
    audit_raw_dataset,
    audit_raw_label_file,
    convert_to_single_class,
    parse_raw_label_line,
    provenance_category,
)


def _make_raw_split(root: Path, split: str, records: dict[str, str]) -> None:
    """records: {stem: label_file_content}"""
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for stem, content in records.items():
        Image.new("RGB", (100, 100), (50, 60, 70)).save(images_dir / f"{stem}.jpg")
        (labels_dir / f"{stem}.txt").write_text(content)


# ---------------------------------------------------------------------------
# A. parse_raw_label_line
# ---------------------------------------------------------------------------

class TestParseRawLabelLine:
    def test_valid_panel_line(self):
        box, error = parse_raw_label_line("3 0.5 0.5 0.2 0.3", 1)
        assert error is None
        assert box == (3, 0.5, 0.5, 0.2, 0.3)

    def test_valid_bird_drop_line(self):
        box, error = parse_raw_label_line("0 0.1 0.1 0.02 0.02", 1)
        assert error is None
        assert box[0] == 0

    def test_unknown_class_id_rejected(self):
        box, error = parse_raw_label_line("9 0.5 0.5 0.1 0.1", 1)
        assert box is None
        assert "unknown class id 9" in error

    def test_wrong_field_count_rejected(self):
        box, error = parse_raw_label_line("3 0.5 0.5 0.1", 1)
        assert box is None
        assert "expected 5 fields" in error

    def test_zero_area_rejected(self):
        box, error = parse_raw_label_line("3 0.5 0.5 0 0.1", 1)
        assert box is None
        assert "zero/negative-area" in error

    def test_box_edge_outside_image_rejected(self):
        box, error = parse_raw_label_line("3 0.05 0.5 0.5 0.1", 1)
        assert box is None
        assert "outside image" in error

    def test_touching_edge_exactly_allowed(self):
        box, error = parse_raw_label_line("3 0.1 0.5 0.2 0.1", 1)
        assert error is None


# ---------------------------------------------------------------------------
# B. audit_raw_label_file
# ---------------------------------------------------------------------------

class TestAuditRawLabelFile:
    def test_counts_classes_correctly(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 0.1 0.1 0.02 0.02\n3 0.5 0.5 0.3 0.3\n2 0.6 0.6 0.1 0.1\n")
        result = audit_raw_label_file(p)
        assert result["class_counts"] == {0: 1, 1: 0, 2: 1, 3: 1}
        assert result["panel_box_count"] == 1
        assert result["errors"] == []

    def test_empty_file_is_valid_zero_boxes(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("")
        result = audit_raw_label_file(p)
        assert result["panel_box_count"] == 0
        assert result["errors"] == []

    def test_malformed_line_reported_not_silently_dropped(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("3 0.5 0.5 0.2 0.2\n7 bad line here\n")
        result = audit_raw_label_file(p)
        assert result["panel_box_count"] == 1
        assert len(result["errors"]) == 1

    def test_high_iou_panel_pair_flagged(self, tmp_path):
        p = tmp_path / "a.txt"
        # Two nearly identical panel boxes - a likely duplicate annotation.
        p.write_text("3 0.5 0.5 0.4 0.4\n3 0.51 0.5 0.4 0.4\n")
        result = audit_raw_label_file(p)
        assert result["panel_box_count"] == 2
        assert len(result["high_iou_panel_pairs"]) == 1

    def test_distinct_panel_boxes_not_flagged(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("3 0.2 0.2 0.1 0.1\n3 0.8 0.8 0.1 0.1\n")
        result = audit_raw_label_file(p)
        assert result["high_iou_panel_pairs"] == []


# ---------------------------------------------------------------------------
# C. audit_raw_dataset
# ---------------------------------------------------------------------------

class TestAuditRawDataset:
    def test_aggregates_across_all_three_splits(self, tmp_path):
        _make_raw_split(tmp_path, "train", {
            "1": "3 0.5 0.5 0.2 0.2\n",
            "2": "0 0.1 0.1 0.02 0.02\n",  # no panel box
        })
        _make_raw_split(tmp_path, "val", {"3": "3 0.5 0.5 0.3 0.3\n3 0.2 0.2 0.1 0.1\n"})
        _make_raw_split(tmp_path, "test", {"4": ""})

        audit = audit_raw_dataset(tmp_path)
        assert audit["per_split"]["train"]["total_label_files"] == 2
        assert audit["per_split"]["val"]["total_label_files"] == 1
        assert audit["per_split"]["test"]["total_label_files"] == 1
        assert audit["images_with_panel_box_count"] == 2  # stem 1 and stem 3
        assert audit["total_class_counts"]["panel"] == 3  # 1 + 2

    def test_images_without_panel_box_excluded_from_panel_list(self, tmp_path):
        _make_raw_split(tmp_path, "train", {"1": "0 0.1 0.1 0.02 0.02\n"})
        audit = audit_raw_dataset(tmp_path)
        assert audit["images_with_panel_box_count"] == 0

    def test_malformed_lines_counted_globally(self, tmp_path):
        _make_raw_split(tmp_path, "train", {"1": "3 0.5 0.5 0.2 0.2\nbadline\n"})
        audit = audit_raw_dataset(tmp_path)
        assert audit["total_malformed_lines"] == 1

    def test_missing_split_directory_skipped_gracefully(self, tmp_path):
        _make_raw_split(tmp_path, "train", {"1": "3 0.5 0.5 0.2 0.2\n"})
        # val/test directories never created.
        audit = audit_raw_dataset(tmp_path)
        assert "val" not in audit["per_split"]
        assert audit["images_with_panel_box_count"] == 1


# ---------------------------------------------------------------------------
# D. convert_to_single_class
# ---------------------------------------------------------------------------

class TestConvertToSingleClass:
    def test_only_images_with_panel_boxes_are_converted(self, tmp_path):
        raw = tmp_path / "raw"
        _make_raw_split(raw, "train", {
            "1": "3 0.5 0.5 0.2 0.2\n",
            "2": "0 0.1 0.1 0.02 0.02\n",  # no panel -> excluded
        })
        out = tmp_path / "out"
        stats = convert_to_single_class(raw, out)
        assert stats["images_converted"] == 1
        assert (out / "images" / "1.jpg").is_file()
        assert not (out / "images" / "2.jpg").exists()

    def test_output_labels_use_only_class_zero(self, tmp_path):
        raw = tmp_path / "raw"
        _make_raw_split(raw, "train", {
            "1": "0 0.1 0.1 0.02 0.02\n3 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.05 0.05\n3 0.7 0.7 0.1 0.1\n",
        })
        out = tmp_path / "out"
        convert_to_single_class(raw, out)
        label_text = (out / "labels" / "1.txt").read_text().strip()
        lines = label_text.splitlines()
        assert len(lines) == 2  # only the 2 panel boxes retained
        for line in lines:
            assert line.startswith(f"{TARGET_CLASS_ID} ")

    def test_non_panel_boxes_never_appear_in_output(self, tmp_path):
        raw = tmp_path / "raw"
        _make_raw_split(raw, "train", {
            "1": "0 0.1 0.1 0.02 0.02\n1 0.2 0.2 0.02 0.02\n2 0.3 0.3 0.02 0.02\n3 0.5 0.5 0.2 0.2\n",
        })
        out = tmp_path / "out"
        convert_to_single_class(raw, out)
        label_text = (out / "labels" / "1.txt").read_text().strip()
        assert len(label_text.splitlines()) == 1

    def test_box_coordinates_preserved_exactly(self, tmp_path):
        raw = tmp_path / "raw"
        _make_raw_split(raw, "train", {"1": "3 0.295354 0.538105 0.365639 0.309829\n"})
        out = tmp_path / "out"
        convert_to_single_class(raw, out)
        line = (out / "labels" / "1.txt").read_text().strip()
        parts = line.split()
        assert parts[0] == "0"
        assert float(parts[1]) == pytest.approx(0.295354, abs=1e-5)
        assert float(parts[4]) == pytest.approx(0.309829, abs=1e-5)

    def test_provenance_json_has_correct_fields_for_every_converted_image(self, tmp_path):
        raw = tmp_path / "raw"
        _make_raw_split(raw, "train", {"1": "3 0.5 0.5 0.2 0.2\n"})
        out = tmp_path / "out"
        convert_to_single_class(raw, out)
        import json
        prov = json.loads((out / "provenance.json").read_text())
        assert "1" in prov
        assert prov["1"]["license"] == "CC-BY-4.0"
        assert "10.17632/8x7wjsr94v.1" in prov["1"]["source_url"]

    def test_conversion_stats_report_excluded_class_totals(self, tmp_path):
        raw = tmp_path / "raw"
        _make_raw_split(raw, "train", {
            "1": "0 0.1 0.1 0.02 0.02\n0 0.2 0.2 0.02 0.02\n3 0.5 0.5 0.2 0.2\n",
        })
        out = tmp_path / "out"
        stats = convert_to_single_class(raw, out)
        assert stats["excluded_class_totals"]["bird_drop"] == 2
        assert stats["total_retained_panel_boxes"] == 1

    def test_never_modifies_the_raw_source(self, tmp_path):
        raw = tmp_path / "raw"
        _make_raw_split(raw, "train", {"1": "3 0.5 0.5 0.2 0.2\n"})
        raw_label = raw / "train" / "labels" / "1.txt"
        before = raw_label.read_text()
        out = tmp_path / "out"
        convert_to_single_class(raw, out)
        after = raw_label.read_text()
        assert before == after

    def test_zero_panel_images_across_whole_dataset_converts_nothing(self, tmp_path):
        raw = tmp_path / "raw"
        _make_raw_split(raw, "train", {"1": "0 0.1 0.1 0.02 0.02\n"})
        out = tmp_path / "out"
        stats = convert_to_single_class(raw, out)
        assert stats["images_converted"] == 0
        assert stats["total_retained_panel_boxes"] == 0

    def test_multi_panel_image_retains_all_panel_boxes(self, tmp_path):
        raw = tmp_path / "raw"
        _make_raw_split(raw, "train", {
            "1": "3 0.2 0.2 0.1 0.1\n3 0.5 0.5 0.1 0.1\n3 0.8 0.8 0.1 0.1\n",
        })
        out = tmp_path / "out"
        stats = convert_to_single_class(raw, out)
        assert stats["total_retained_panel_boxes"] == 3
        assert len((out / "labels" / "1.txt").read_text().strip().splitlines()) == 3


# ---------------------------------------------------------------------------
# E. provenance_category / provenance filtering
# ---------------------------------------------------------------------------

class TestProvenanceCategory:
    def test_numeric_stem_is_native(self):
        assert provenance_category("1108") == "numeric_native"

    def test_a_prefixed_stem_is_confirmed_contaminated(self):
        assert provenance_category("a406") == "confirmed_contaminated"

    def test_oip_prefixed_stem_is_confirmed_contaminated(self):
        assert provenance_category("OIP-1") == "confirmed_contaminated"

    def test_b_prefixed_stem_is_unverified(self):
        assert provenance_category("b10") == "unverified_prefixed"


class TestConvertToSingleClassProvenanceFilter:
    def test_default_filter_excludes_non_numeric_stems(self, tmp_path):
        raw = tmp_path / "raw"
        _make_raw_split(raw, "train", {
            "1": "3 0.5 0.5 0.2 0.2\n",       # numeric_native -> kept
            "a406": "3 0.5 0.5 0.2 0.2\n",    # confirmed_contaminated -> excluded
            "b10": "3 0.5 0.5 0.2 0.2\n",     # unverified_prefixed -> excluded
        })
        out = tmp_path / "out"
        stats = convert_to_single_class(raw, out)
        assert stats["images_converted"] == 1
        assert (out / "images" / "1.jpg").is_file()
        assert not (out / "images" / "a406.jpg").exists()
        assert not (out / "images" / "b10.jpg").exists()
        assert stats["images_skipped_provenance_filter"] == 2
        assert set(stats["skipped_provenance_filter_stems"]) == {"a406", "b10"}

    def test_filter_disabled_with_none_includes_everything(self, tmp_path):
        raw = tmp_path / "raw"
        _make_raw_split(raw, "train", {
            "1": "3 0.5 0.5 0.2 0.2\n",
            "a406": "3 0.5 0.5 0.2 0.2\n",
        })
        out = tmp_path / "out"
        stats = convert_to_single_class(raw, out, allowed_provenance_categories=None)
        assert stats["images_converted"] == 2
        assert stats["images_skipped_provenance_filter"] == 0

    def test_custom_allowed_set_can_include_unverified(self, tmp_path):
        raw = tmp_path / "raw"
        _make_raw_split(raw, "train", {
            "1": "3 0.5 0.5 0.2 0.2\n",
            "b10": "3 0.5 0.5 0.2 0.2\n",
            "a406": "3 0.5 0.5 0.2 0.2\n",
        })
        out = tmp_path / "out"
        stats = convert_to_single_class(
            raw, out, allowed_provenance_categories={"numeric_native", "unverified_prefixed"}
        )
        assert stats["images_converted"] == 2
        assert (out / "images" / "b10.jpg").is_file()
        assert not (out / "images" / "a406.jpg").exists()

    def test_provenance_json_notes_include_category(self, tmp_path):
        raw = tmp_path / "raw"
        _make_raw_split(raw, "train", {"1": "3 0.5 0.5 0.2 0.2\n"})
        out = tmp_path / "out"
        convert_to_single_class(raw, out)
        import json
        prov = json.loads((out / "provenance.json").read_text())
        assert "provenance_category=numeric_native" in prov["1"]["notes"]


class TestManualSiteGroupKey:
    def test_known_site_a_stem_gets_group_key(self, tmp_path):
        raw = tmp_path / "raw"
        _make_raw_split(raw, "train", {"1108": "3 0.5 0.5 0.2 0.2\n"})
        out = tmp_path / "out"
        convert_to_single_class(raw, out)
        import json
        prov = json.loads((out / "provenance.json").read_text())
        assert prov["1108"]["group_key"] == "mendeley_site_A"

    def test_unlisted_numeric_stem_has_no_group_key(self, tmp_path):
        raw = tmp_path / "raw"
        _make_raw_split(raw, "train", {"999999": "3 0.5 0.5 0.2 0.2\n"})
        out = tmp_path / "out"
        convert_to_single_class(raw, out)
        import json
        prov = json.loads((out / "provenance.json").read_text())
        assert prov["999999"]["group_key"] is None


class TestAuditRawDatasetProvenanceBreakdown:
    def test_provenance_category_counts_present(self, tmp_path):
        _make_raw_split(tmp_path, "train", {
            "1": "3 0.5 0.5 0.2 0.2\n",
            "a406": "3 0.5 0.5 0.2 0.2\n3 0.2 0.2 0.1 0.1\n",
        })
        audit = audit_raw_dataset(tmp_path)
        assert audit["provenance_category_image_counts"]["numeric_native"] == 1
        assert audit["provenance_category_image_counts"]["confirmed_contaminated"] == 1
        assert audit["provenance_category_panel_box_counts"]["confirmed_contaminated"] == 2
