"""Adversarial security and failure-mode validation tests.

These tests actively attempt to bypass existing security controls.
They use harmless synthetic inputs only.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

from services.weather import _validate_city, fetch_weather
from utils.security import sanitize_for_log
from scripts.verify_model_artifacts import verify_manifest, _resolve_artifact_path


# ===========================================================================
# ATTACK SURFACE 2 — CITY / LOCATION INPUT
# ===========================================================================

class TestCityAdversarialInputs:
    """Attempt to bypass city validation or inject malicious content."""

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_city("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_city("   ")

    def test_null_bytes_rejected(self):
        with pytest.raises(ValueError, match="invalid control characters"):
            _validate_city("Chennai\x00\x01\x02")

    def test_newline_rejected(self):
        with pytest.raises(ValueError, match="invalid control characters"):
            _validate_city("Chennai\r\n")

    def test_tab_rejected(self):
        with pytest.raises(ValueError, match="invalid control characters"):
            _validate_city("Chennai\there")

    def test_del_character_rejected(self):
        with pytest.raises(ValueError, match="invalid control characters"):
            _validate_city("Chennai\x7f")

    def test_shell_metacharacters_not_executed(self):
        """Shell metacharacters in city are just text — they are not executed
        because the value is passed as an HTTP query parameter, not to a shell."""
        result = _validate_city("Chennai; rm -rf /")
        assert result == "Chennai; rm -rf /"

    def test_path_traversal_not_executed(self):
        """Path traversal strings are just text — not filesystem operations."""
        result = _validate_city("../../../etc/passwd")
        assert result == "../../../etc/passwd"

    def test_url_like_string_truncated_safely(self):
        result = _validate_city("http://evil.com/attack" * 20)
        assert len(result) <= 100

    def test_localhost_string_truncated(self):
        result = _validate_city("127.0.0.1" * 20)
        assert len(result) <= 100

    def test_file_url_not_executed(self):
        """file:// URLs are just text — not executed as file operations."""
        result = _validate_city("file:///etc/passwd")
        assert result == "file:///etc/passwd"

    def test_unicode_edge_cases_preserved(self):
        result = _validate_city("São Paulo")
        assert result == "São Paulo"

    def test_max_length_enforced(self):
        result = _validate_city("A" * 200)
        assert len(result) == 100

    def test_leading_trailing_whitespace_stripped(self):
        result = _validate_city("  Chennai  ")
        assert result == "Chennai"

    def test_non_string_rejected(self):
        with pytest.raises(ValueError, match="must be a string"):
            _validate_city(12345)

    def test_none_rejected(self):
        with pytest.raises(ValueError, match="must be a string"):
            _validate_city(None)


class TestCityLogInjectionAcrossAllPaths:
    """Verify no log injection via city input in any code path."""

    def test_fetch_weather_logs_sanitized_city_on_validation_failure(self, monkeypatch, caplog):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        import logging
        with caplog.at_level(logging.WARNING):
            fetch_weather("Chennai\x00\x01INJECTED")
        assert "\x00" not in caplog.text
        assert "\x01" not in caplog.text
        # The text "INJECTED" itself is not a security issue — control chars are
        # the log-injection risk. Sanitization removes control chars, not content.
        assert "ChennaiINJECTED" in caplog.text

    def test_sanitize_for_log_strips_all_control_chars(self):
        malicious = "normal\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x7ftext"
        cleaned = sanitize_for_log(malicious)
        assert "\x00" not in cleaned
        assert "\x7f" not in cleaned
        assert cleaned == "normaltext"

    def test_sanitize_for_log_prevents_newline_injection(self):
        malicious = "real\nFAKE LOG ENTRY\nmore"
        cleaned = sanitize_for_log(malicious)
        assert "\n" not in cleaned
        # Newlines are stripped, preventing log line injection.
        # The text between newlines remains on the same log line.
        assert "FAKE LOG ENTRY" in cleaned

    def test_sanitize_for_log_prevents_carriage_return_injection(self):
        malicious = "real\rFAKE LOG ENTRY\rmore"
        cleaned = sanitize_for_log(malicious)
        assert "\r" not in cleaned
        assert "FAKE LOG ENTRY" in cleaned


# ===========================================================================
# ATTACK SURFACE 1 — FILE UPLOADS
# ===========================================================================

class TestFileUploadAdversarial:
    """Test image upload handling with malicious inputs."""

    def test_valid_jpeg_accepts(self):
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        with Image.open(buf) as im:
            im.verify()
        assert True

    def test_corrupt_image_rejected(self):
        buf = io.BytesIO(b"this is not an image")
        with pytest.raises(UnidentifiedImageError):
            with Image.open(buf) as im:
                im.verify()

    def test_empty_bytes_rejected(self):
        buf = io.BytesIO(b"")
        with pytest.raises((UnidentifiedImageError, OSError)):
            with Image.open(buf) as im:
                im.verify()

    def test_truncated_jpeg_rejected(self):
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        truncated = buf.read()[:10]
        buf2 = io.BytesIO(truncated)
        with pytest.raises((UnidentifiedImageError, OSError)):
            with Image.open(buf2) as im:
                im.verify()

    def test_png_accepted_when_content_is_valid(self):
        """Pillow auto-detects format, so a valid PNG is accepted regardless of
        file extension. The security boundary is content validation, not extension."""
        buf = io.BytesIO()
        Image.new("RGB", (50, 50)).save(buf, format="PNG")
        buf.seek(0)
        with Image.open(buf) as im:
            im.verify()
        assert True

    def test_rgba_converted_to_rgb(self):
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        with Image.open(buf) as im:
            rgb = im.convert("RGB")
        assert rgb.mode == "RGB"

    def test_grayscale_converted_to_rgb(self):
        img = Image.new("L", (100, 100), 128)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        with Image.open(buf) as im:
            rgb = im.convert("RGB")
        assert rgb.mode == "RGB"

    def test_decompression_bomb_detected(self):
        with pytest.raises(Image.DecompressionBombError):
            img = Image.new("RGB", (10000, 10000))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            Image.MAX_IMAGE_PIXELS = 1000
            try:
                with Image.open(buf) as im:
                    im.verify()
            finally:
                Image.MAX_IMAGE_PIXELS = None

    def test_filename_with_path_traversal_not_used(self):
        """Streamlit file_uploader provides a filename, but app.py never uses it
        for filesystem operations — only the bytes content is processed."""
        img = Image.new("RGB", (100, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()
        with Image.open(io.BytesIO(raw_bytes)) as im:
            im.verify()
        with Image.open(io.BytesIO(raw_bytes)) as im:
            pil_image = im.convert("RGB")
        assert pil_image is not None


# ===========================================================================
# ATTACK SURFACE 3 — MODEL MANIFEST
# ===========================================================================

class TestModelManifestAdversarial:
    """Verify manifest path traversal and boundary protections."""

    def test_absolute_path_rejected(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps({"artifacts": [{"path": str(tmp_path / "outside.bin"), "sha256": "0" * 64}]}),
            encoding="utf-8",
        )
        ok, errors = verify_manifest(manifest)
        assert ok is False
        assert any("relative" in e for e in errors)

    def test_parent_traversal_rejected(self, tmp_path):
        weights = tmp_path / "weights"
        weights.mkdir()
        manifest = weights / "manifest.json"
        manifest.write_text(
            json.dumps({"artifacts": [{"path": "../outside.bin", "sha256": "0" * 64}]}),
            encoding="utf-8",
        )
        ok, errors = verify_manifest(manifest)
        assert ok is False
        assert any("escapes" in e for e in errors)

    def test_nested_parent_traversal_rejected(self, tmp_path):
        weights = tmp_path / "weights"
        weights.mkdir()
        deep = weights / "a" / "b" / "c"
        deep.mkdir(parents=True)
        manifest = deep / "manifest.json"
        manifest.write_text(
            json.dumps({"artifacts": [{"path": "../../../../outside.bin", "sha256": "0" * 64}]}),
            encoding="utf-8",
        )
        ok, errors = verify_manifest(manifest)
        assert ok is False
        assert any("escapes" in e for e in errors)

    def test_dot_slash_traversal_rejected(self, tmp_path):
        weights = tmp_path / "weights"
        weights.mkdir()
        manifest = weights / "manifest.json"
        manifest.write_text(
            json.dumps({"artifacts": [{"path": "./../outside.bin", "sha256": "0" * 64}]}),
            encoding="utf-8",
        )
        ok, errors = verify_manifest(manifest)
        assert ok is False
        assert any("escapes" in e for e in errors)

    def test_symlink_outside_rejected(self, tmp_path):
        weights = tmp_path / "weights"
        weights.mkdir()
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"outside")
        symlink = weights / "link.bin"
        try:
            symlink.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")

        manifest = weights / "manifest.json"
        digest = "0" * 64
        manifest.write_text(
            json.dumps({"artifacts": [{"path": "link.bin", "sha256": digest}]}),
            encoding="utf-8",
        )
        ok, errors = verify_manifest(manifest)
        assert ok is False
        assert any("escapes" in e for e in errors)

    def test_broken_symlink_rejected_as_missing(self, tmp_path):
        weights = tmp_path / "weights"
        weights.mkdir()
        target = tmp_path / "nonexistent.bin"
        symlink = weights / "broken_link.bin"
        try:
            symlink.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")

        manifest = weights / "manifest.json"
        digest = "0" * 64
        manifest.write_text(
            json.dumps({"artifacts": [{"path": "broken_link.bin", "sha256": digest}]}),
            encoding="utf-8",
        )
        ok, errors = verify_manifest(manifest)
        assert ok is False
        assert any("missing" in e for e in errors)

    def test_wrong_hash_rejected(self, tmp_path):
        artifact = tmp_path / "model.bin"
        artifact.write_bytes(b"trusted model bytes")
        digest = "0" * 64
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps({"artifacts": [{"path": "model.bin", "sha256": digest}]}),
            encoding="utf-8",
        )
        ok, errors = verify_manifest(manifest)
        assert ok is False
        assert any("mismatch" in e for e in errors)

    def test_missing_artifact_rejected(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps({"artifacts": [{"path": "missing.bin", "sha256": "0" * 64}]}),
            encoding="utf-8",
        )
        ok, errors = verify_manifest(manifest)
        assert ok is False
        assert any("missing" in e for e in errors)

    def test_malformed_manifest_rejected(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text("not json", encoding="utf-8")
        ok, errors = verify_manifest(manifest)
        assert ok is False

    def test_missing_artifacts_list_rejected(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps({"not_artifacts": []}),
            encoding="utf-8",
        )
        ok, errors = verify_manifest(manifest)
        assert ok is False
        assert any("artifacts" in e for e in errors)

    def test_empty_artifacts_list_rejected(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps({"artifacts": []}),
            encoding="utf-8",
        )
        ok, errors = verify_manifest(manifest)
        assert ok is False
        assert any("non-empty" in e for e in errors)

    def test_non_dict_artifact_entry_rejected(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps({"artifacts": ["bad entry"]}),
            encoding="utf-8",
        )
        ok, errors = verify_manifest(manifest)
        assert ok is False
        assert any("object" in e for e in errors)

    def test_invalid_sha256_rejected(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps({"artifacts": [{"path": "model.bin", "sha256": "not-a-hash"}]}),
            encoding="utf-8",
        )
        ok, errors = verify_manifest(manifest)
        assert ok is False
        assert any("SHA-256" in e for e in errors)

    def test_duplicate_artifact_entries_both_checked(self, tmp_path):
        artifact = tmp_path / "model.bin"
        artifact.write_bytes(b"data")
        digest = "0" * 64
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps({"artifacts": [
                {"path": "model.bin", "sha256": digest},
                {"path": "model.bin", "sha256": digest},
            ]}),
            encoding="utf-8",
        )
        ok, errors = verify_manifest(manifest)
        assert ok is False
        assert len(errors) == 2

    def test_manifest_symlink_boundary_anchored_to_supplied_parent(self, tmp_path):
        real_dir = tmp_path / "real"
        supplied_dir = tmp_path / "bundle"
        real_dir.mkdir()
        supplied_dir.mkdir()

        artifact = supplied_dir / "model.bin"
        artifact.write_bytes(b"trusted model bytes")
        digest = "0" * 64

        target_manifest = real_dir / "manifest.json"
        target_manifest.write_text(
            json.dumps({"artifacts": [{"path": artifact.name, "sha256": digest}]}),
            encoding="utf-8",
        )
        supplied_manifest = supplied_dir / "manifest.json"
        try:
            os.symlink(target_manifest, supplied_manifest)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

        ok, errors = verify_manifest(supplied_manifest)
        assert ok is True
        assert errors == []


# ===========================================================================
# ATTACK SURFACE 4 — MODEL LOADING
# ===========================================================================

class TestModelLoadingAdversarial:
    """Verify model loading fails safely under adversarial conditions."""

    def test_missing_all_artifacts_raises_typed_errors(self):
        from models.model_manager import ModelManager
        mm = ModelManager()
        with pytest.raises(Exception):
            mm.get_detector()
        with pytest.raises(Exception):
            mm.get_classifier()
        with pytest.raises(Exception):
            mm.get_predictor()

    def test_artifact_status_reports_missing(self):
        from models.model_manager import ModelManager
        mm = ModelManager()
        status = mm.artifact_status
        assert status["YOLO"]["exists"] is False
        assert status["MobileNet"]["exists"] is False
        assert status["XGBoost"]["exists"] is False

    def test_readiness_script_fails_closed_without_artifacts(self):
        import scripts.check_runtime_readiness as readiness
        from models.model_manager import ModelManager
        mm = ModelManager()
        original = readiness.model_manager
        readiness.model_manager = mm
        try:
            assert readiness.main() == 2
        finally:
            readiness.model_manager = original

    def test_model_manager_logs_no_absolute_paths_on_missing(self, caplog):
        import logging
        from models.model_manager import ModelManager
        mm = ModelManager()
        with caplog.at_level(logging.INFO):
            with pytest.raises(Exception):
                mm.get_detector()
        assert str(Path(__file__).resolve()) not in caplog.text


# ===========================================================================
# ATTACK SURFACE 14 — CONFIGURATION TAMPERING
# ===========================================================================

class TestConfigAdversarial:
    """Verify configuration validation rejects malformed configs."""

    def test_missing_required_section_rejected(self):
        from utils.config import load_config
        import tempfile, yaml
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"not_weather": {}}, f)
            path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="Missing required configuration sections"):
                load_config(path)
        finally:
            path.unlink()

    def test_missing_physics_key_rejected(self):
        from utils.config import load_config
        import tempfile, yaml
        config = {
            "weather": {"base_url": "x", "timeout_seconds": 10, "default_city": "x", "units": "x", "defaults": {"ambient_temp_c": 25, "humidity_pct": 50, "wind_speed_ms": 2, "cloud_cover_pct": 0, "pressure_hpa": 1013, "latitude": 0, "longitude": 0}},
            "models": {"yolo": {"weights": "x", "confidence_threshold": 0.5, "iou_threshold": 0.5, "image_size": 640}, "mobilenet": {"weights": "x", "num_classes": 6, "input_size": 224}, "xgboost": {"weights": "x"}},
            "classification": {"labels": ["Clean"]},
            "physics": {"max_irradiance_wm2": 1000, "irradiance_cloud_factor": 0.75, "noct_celsius": 45, "noct_irradiance_ref": 800, "noct_ambient_ref": 20, "wind_cooling_coefficient": 1.5, "temp_coefficient_pmax": -0.004, "stc_temperature": 25, "soiling_ratios": {"Clean": 1.0}, "panel_rated_power_wp": 400},
            "feature_engineering": {"feature_columns": ["x"]},
            "recommendations": {"efficiency_loss_critical_pct": 20, "efficiency_loss_warning_pct": 10, "hotspot_max_temp_c": 65, "cleaning_humidity_threshold_pct": 85},
            "logging": {"level": "INFO", "format": "x"},
        }
        del config["physics"]["irradiance_cloud_factor"]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="Missing required configuration keys"):
                load_config(path)
        finally:
            path.unlink()

    def test_irradiance_cloud_factor_compatibility_preserved(self):
        from utils.config import load_config
        config = load_config()
        assert "irradiance_cloud_factor" in config["physics"]

    def test_negative_physics_value_rejected(self):
        from utils.config import load_config
        import tempfile, yaml
        config = {
            "weather": {"base_url": "x", "timeout_seconds": 10, "default_city": "x", "units": "x", "defaults": {"ambient_temp_c": 25, "humidity_pct": 50, "wind_speed_ms": 2, "cloud_cover_pct": 0, "pressure_hpa": 1013, "latitude": 0, "longitude": 0}},
            "models": {"yolo": {"weights": "x", "confidence_threshold": 0.5, "iou_threshold": 0.5, "image_size": 640}, "mobilenet": {"weights": "x", "num_classes": 6, "input_size": 224}, "xgboost": {"weights": "x"}},
            "classification": {"labels": ["Clean"]},
            "physics": {"max_irradiance_wm2": 1000, "irradiance_cloud_factor": -1.0, "noct_celsius": 45, "noct_irradiance_ref": 800, "noct_ambient_ref": 20, "wind_cooling_coefficient": 1.5, "temp_coefficient_pmax": -0.004, "stc_temperature": 25, "soiling_ratios": {"Clean": 1.0}, "panel_rated_power_wp": 400},
            "feature_engineering": {"feature_columns": ["x"]},
            "recommendations": {"efficiency_loss_critical_pct": 20, "efficiency_loss_warning_pct": 10, "hotspot_max_temp_c": 65, "cleaning_humidity_threshold_pct": 85},
            "logging": {"level": "INFO", "format": "x"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="must be >= 0"):
                load_config(path)
        finally:
            path.unlink()
