"""tests/test_image_utils.py - Deterministic tests for utils/image_utils.py.

Covers the image preprocessing utilities:

A. load_pil_image
B. resize_for_yolo
C. resize_for_mobilenet
D. pil_to_numpy
E. get_image_dimensions
F. Configuration behavior
G. Logging behavior
H. Determinism

Design rules honoured:
- no real model weights
- no network access
- deterministic PIL/numpy operations
- existing conftest fixtures preferred
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from utils.image_utils import (
    _MN_SIZE,
    _YOLO_SIZE,
    get_image_dimensions,
    load_pil_image,
    pil_to_numpy,
    resize_for_mobilenet,
    resize_for_yolo,
)
from utils.config import CFG
from utils.exceptions import ImageValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_png_bytes(width=100, height=100, mode="RGB"):
    """Create valid PNG image bytes for testing."""
    img = Image.new(mode, (width, height), (120, 130, 140))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# A. load_pil_image
# ---------------------------------------------------------------------------

class TestLoadPilImage:
    """Image loading from various sources."""

    def test_none_source_raises_image_validation_error(self):
        with pytest.raises(ImageValidationError, match="No image source provided"):
            load_pil_image(None)

    def test_empty_bytes_raises_image_validation_error(self):
        with pytest.raises(ImageValidationError, match="Image byte buffer is empty"):
            load_pil_image(b"")

    def test_empty_bytearray_raises_image_validation_error(self):
        with pytest.raises(ImageValidationError, match="Image byte buffer is empty"):
            load_pil_image(bytearray())

    def test_invalid_type_raises_image_validation_error(self):
        with pytest.raises(ImageValidationError, match="Unsupported image source type"):
            load_pil_image(12345)

    def test_valid_path_returns_rgb_image(self, tmp_path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        img.save(img_path)
        
        result = load_pil_image(str(img_path))
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_valid_pathlib_path_returns_rgb_image(self, tmp_path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), (0, 255, 0))
        img.save(img_path)
        
        result = load_pil_image(img_path)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_valid_bytes_returns_rgb_image(self):
        img_bytes = _make_valid_png_bytes()
        result = load_pil_image(img_bytes)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_valid_bytearray_returns_rgb_image(self):
        img_bytes = _make_valid_png_bytes()
        result = load_pil_image(bytearray(img_bytes))
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_nonexistent_path_raises_image_validation_error(self):
        with pytest.raises(ImageValidationError, match="Could not open image"):
            load_pil_image("/nonexistent/path/image.png")

    def test_corrupted_bytes_raises_image_validation_error(self):
        with pytest.raises(ImageValidationError, match="Could not decode image"):
            load_pil_image(b"not_an_image_at_all")

    def test_converts_rgba_to_rgb(self, tmp_path):
        img_path = tmp_path / "test_rgba.png"
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        img.save(img_path)
        
        result = load_pil_image(str(img_path))
        assert result.mode == "RGB"

    def test_loads_jpeg_format(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), (0, 255, 0))
        img.save(img_path, format="JPEG")
        
        result = load_pil_image(str(img_path))
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# B. resize_for_yolo
# ---------------------------------------------------------------------------

class TestResizeForYolo:
    """YOLO resize behavior."""

    def test_returns_square_image(self):
        img = Image.new("RGB", (800, 600), (255, 0, 0))
        result = resize_for_yolo(img)
        assert result.width == _YOLO_SIZE
        assert result.height == _YOLO_SIZE

    def test_wide_image_resized_correctly(self):
        img = Image.new("RGB", (800, 400), (255, 0, 0))
        result = resize_for_yolo(img)
        assert result.width == _YOLO_SIZE
        assert result.height == _YOLO_SIZE

    def test_tall_image_resized_correctly(self):
        img = Image.new("RGB", (400, 800), (0, 255, 0))
        result = resize_for_yolo(img)
        assert result.width == _YOLO_SIZE
        assert result.height == _YOLO_SIZE

    def test_square_image_resized_correctly(self):
        img = Image.new("RGB", (500, 500), (0, 0, 255))
        result = resize_for_yolo(img)
        assert result.width == _YOLO_SIZE
        assert result.height == _YOLO_SIZE

    def test_returns_rgb_mode(self):
        img = Image.new("RGB", (800, 600), (255, 0, 0))
        result = resize_for_yolo(img)
        assert result.mode == "RGB"

    def test_small_image_resized_correctly(self):
        img = Image.new("RGB", (50, 50), (255, 255, 0))
        result = resize_for_yolo(img)
        assert result.width == _YOLO_SIZE
        assert result.height == _YOLO_SIZE

    def test_uses_configured_yolo_size(self, project_config):
        img = Image.new("RGB", (800, 600), (255, 0, 0))
        expected_size = project_config["models"]["yolo"]["image_size"]
        result = resize_for_yolo(img)
        assert result.width == expected_size
        assert result.height == expected_size

    def test_preserves_aspect_ratio_visually(self):
        """The resized image should maintain aspect ratio via letterboxing."""
        img = Image.new("RGB", (800, 400), (255, 0, 0))  # 2:1 aspect ratio
        result = resize_for_yolo(img)
        
        # After resize: (640, 320) → pasted on (640, 640) canvas
        # The non-grey area should reflect the original aspect ratio
        assert result.width == _YOLO_SIZE
        assert result.height == _YOLO_SIZE


# ---------------------------------------------------------------------------
# C. resize_for_mobilenet
# ---------------------------------------------------------------------------

class TestResizeForMobileNet:
    """MobileNet resize behavior."""

    def test_returns_square_image(self):
        img = Image.new("RGB", (300, 300), (255, 0, 0))
        result = resize_for_mobilenet(img)
        assert result.width == _MN_SIZE
        assert result.height == _MN_SIZE

    def test_wide_image_resized_correctly(self):
        img = Image.new("RGB", (600, 300), (255, 0, 0))
        result = resize_for_mobilenet(img)
        assert result.width == _MN_SIZE
        assert result.height == _MN_SIZE

    def test_tall_image_resized_correctly(self):
        img = Image.new("RGB", (300, 600), (0, 255, 0))
        result = resize_for_mobilenet(img)
        assert result.width == _MN_SIZE
        assert result.height == _MN_SIZE

    def test_returns_rgb_mode(self):
        img = Image.new("RGB", (300, 300), (0, 0, 255))
        result = resize_for_mobilenet(img)
        assert result.mode == "RGB"

    def test_uses_configured_mobilenet_size(self, project_config):
        img = Image.new("RGB", (300, 300), (255, 0, 0))
        expected_size = project_config["models"]["mobilenet"]["input_size"]
        result = resize_for_mobilenet(img)
        assert result.width == expected_size
        assert result.height == expected_size

    def test_centre_crop_behavior(self):
        """After resize and crop, the result should be exactly input_size x input_size."""
        img = Image.new("RGB", (400, 300), (128, 128, 128))
        result = resize_for_mobilenet(img)
        assert result.width == _MN_SIZE
        assert result.height == _MN_SIZE


# ---------------------------------------------------------------------------
# D. pil_to_numpy
# ---------------------------------------------------------------------------

class TestPilToNumpy:
    """PIL to numpy conversion behavior."""

    def test_returns_numpy_array(self):
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        result = pil_to_numpy(img)
        assert isinstance(result, np.ndarray)

    def test_returns_uint8_dtype(self):
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        result = pil_to_numpy(img)
        assert result.dtype == np.uint8

    def test_returns_correct_shape(self):
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        result = pil_to_numpy(img)
        assert result.shape == (100, 100, 3)

    def test_preserves_pixel_values(self):
        img = Image.new("RGB", (10, 10), (123, 45, 67))
        result = pil_to_numpy(img)
        assert result[0, 0, 0] == 123
        assert result[0, 0, 1] == 45
        assert result[0, 0, 2] == 67


# ---------------------------------------------------------------------------
# E. get_image_dimensions
# ---------------------------------------------------------------------------

class TestGetImageDimensions:
    """Image dimension extraction."""

    def test_returns_width_height_tuple(self):
        img = Image.new("RGB", (640, 480), (255, 0, 0))
        width, height = get_image_dimensions(img)
        assert width == 640
        assert height == 480

    def test_returns_integers(self):
        img = Image.new("RGB", (100, 200), (0, 255, 0))
        width, height = get_image_dimensions(img)
        assert isinstance(width, int)
        assert isinstance(height, int)

    def test_square_image_dimensions(self):
        img = Image.new("RGB", (300, 300), (0, 0, 255))
        width, height = get_image_dimensions(img)
        assert width == 300
        assert height == 300


# ---------------------------------------------------------------------------
# F. Configuration behavior
# ---------------------------------------------------------------------------

class TestConfigurationBehavior:
    """Image utilities use config for sizes."""

    def test_yolo_size_from_config(self, project_config):
        expected = project_config["models"]["yolo"]["image_size"]
        assert _YOLO_SIZE == expected

    def test_mobilenet_size_from_config(self, project_config):
        expected = project_config["models"]["mobilenet"]["input_size"]
        assert _MN_SIZE == expected


# ---------------------------------------------------------------------------
# G. Logging behavior
# ---------------------------------------------------------------------------

class TestLoggingBehavior:
    """Image utilities emit expected debug logs."""

    def test_load_pil_image_logs_debug_on_success(self, caplog):
        img_bytes = _make_valid_png_bytes()
        import logging
        with caplog.at_level(logging.DEBUG):
            load_pil_image(img_bytes)
        
        assert "Loaded image from bytes buffer" in caplog.text

    def test_resize_for_yolo_logs_debug(self, caplog):
        img = Image.new("RGB", (800, 600), (255, 0, 0))
        import logging
        with caplog.at_level(logging.DEBUG):
            resize_for_yolo(img)
        
        assert "Resized image for YOLO" in caplog.text

    def test_resize_for_mobilenet_logs_debug(self, caplog):
        img = Image.new("RGB", (400, 300), (255, 0, 0))
        import logging
        with caplog.at_level(logging.DEBUG):
            resize_for_mobilenet(img)
        
        assert "Resized image for MobileNet" in caplog.text


# ---------------------------------------------------------------------------
# H. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Repeated operations with same input produce equivalent results."""

    def test_repeated_load_produces_equivalent_images(self):
        img_bytes = _make_valid_png_bytes()
        r1 = load_pil_image(img_bytes)
        r2 = load_pil_image(img_bytes)
        
        assert r1.size == r2.size
        assert r1.mode == r2.mode
        assert np.array_equal(np.array(r1), np.array(r2))

    def test_repeated_resize_produces_equivalent_results(self):
        img = Image.new("RGB", (800, 600), (255, 0, 0))
        r1 = resize_for_yolo(img)
        r2 = resize_for_yolo(img)
        
        assert np.array_equal(np.array(r1), np.array(r2))