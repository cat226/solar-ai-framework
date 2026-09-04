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
    crop_panel,
    get_image_dimensions,
    load_pil_image,
    pil_to_numpy,
    resize_for_mobilenet,
    resize_for_yolo,
    unletterbox_box,
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
# B2. unletterbox_box — must be the exact inverse of resize_for_yolo's transform
# ---------------------------------------------------------------------------

class TestUnletterboxBox:
    """Detection boxes are in letterboxed 640x640 space; this must map them
    back to the original image's coordinates, or every overlay drawn on the
    original image would be misplaced."""

    def test_wide_image_known_coordinates(self):
        # 800x400, ratio=2 -> new_w=640, new_h=320, offset_x=0, offset_y=160, scale=1.25
        box = unletterbox_box((100.0, 210.0, 300.0, 410.0), (800, 400))
        assert box == pytest.approx((125.0, 62.5, 375.0, 312.5))

    def test_tall_image_known_coordinates(self):
        # 400x800, ratio=0.5 -> new_w=320, new_h=640, offset_x=160, offset_y=0, scale=1.25
        box = unletterbox_box((210.0, 100.0, 410.0, 300.0), (400, 800))
        assert box == pytest.approx((62.5, 125.0, 312.5, 375.0))

    def test_square_image_no_offset(self):
        # 640x640 original -> no scaling, no offset at all.
        box = unletterbox_box((50.0, 60.0, 200.0, 220.0), (640, 640))
        assert box == pytest.approx((50.0, 60.0, 200.0, 220.0))

    def test_clamps_to_image_bounds(self):
        # A box reaching into the grey padding region must clamp, not go negative
        # or exceed the original image size.
        box = unletterbox_box((-10.0, -10.0, 700.0, 700.0), (800, 400))
        x1, y1, x2, y2 = box
        assert 0.0 <= x1 <= 800.0
        assert 0.0 <= y1 <= 400.0
        assert 0.0 <= x2 <= 800.0
        assert 0.0 <= y2 <= 400.0

    def test_round_trip_against_real_resize_for_yolo(self):
        """A box drawn at a known fractional position on the original image,
        mapped forward through the real letterbox geometry by hand, must map
        back via unletterbox_box to (approximately) the original position -
        proving this is the true inverse of resize_for_yolo, not a
        coincidentally-similar formula."""
        orig_w, orig_h = 800, 400
        img = Image.new("RGB", (orig_w, orig_h))
        letterboxed = resize_for_yolo(img)
        assert letterboxed.size == (_YOLO_SIZE, _YOLO_SIZE)

        # A point at the original image's exact centre must letterbox to the
        # letterboxed canvas's exact centre (640x640 centre = (320, 320)),
        # regardless of the aspect-ratio padding, and unletterbox_box must
        # map that canvas centre back to the original centre.
        canvas_centre_box = (315.0, 315.0, 325.0, 325.0)
        mapped_back = unletterbox_box(canvas_centre_box, (orig_w, orig_h))
        cx = (mapped_back[0] + mapped_back[2]) / 2
        cy = (mapped_back[1] + mapped_back[3]) / 2
        assert cx == pytest.approx(orig_w / 2, abs=1.0)
        assert cy == pytest.approx(orig_h / 2, abs=1.0)


class TestCropPanel:
    """crop_panel() - unletterbox a detection box then crop the *original*
    image, the real "Panel crops" step in the YOLO -> MobileNet pipeline."""

    def test_crop_has_expected_content_at_known_position(self):
        # A distinct color block in the top-left quadrant of the original
        # image, the rest black - the crop must contain (at least mostly)
        # that color, proving it cropped the real original image region,
        # not some arbitrary letterboxed-canvas slice.
        orig_w, orig_h = 640, 640  # square, so no letterbox padding at all
        img = Image.new("RGB", (orig_w, orig_h), (0, 0, 0))
        for x in range(0, 100):
            for y in range(0, 100):
                img.putpixel((x, y), (255, 0, 0))

        crop = crop_panel(img, (0.0, 0.0, 100.0, 100.0))
        assert crop.size == (100, 100)
        assert crop.getpixel((50, 50)) == (255, 0, 0)

    def test_crop_size_matches_box_dimensions(self):
        img = Image.new("RGB", (640, 640))
        crop = crop_panel(img, (100.0, 100.0, 300.0, 250.0))
        assert crop.size == (200, 150)

    def test_degenerate_zero_area_box_still_produces_nonempty_crop(self):
        img = Image.new("RGB", (640, 640))
        crop = crop_panel(img, (100.0, 100.0, 100.0, 100.0))
        assert crop.width >= 1
        assert crop.height >= 1

    def test_box_outside_image_bounds_is_clamped(self):
        img = Image.new("RGB", (640, 640))
        crop = crop_panel(img, (-50.0, -50.0, 5000.0, 5000.0))
        assert crop.width <= 640
        assert crop.height <= 640
        assert crop.width >= 1
        assert crop.height >= 1

    def test_crop_on_non_square_original_image_uses_unletterbox_mapping(self):
        """A box near the letterboxed canvas's centre on a wide original
        image must crop from that image's own centre, not naively (which
        would be off because of the aspect-ratio padding)."""
        orig_w, orig_h = 800, 400
        img = Image.new("RGB", (orig_w, orig_h), (0, 0, 0))
        cx, cy = orig_w // 2, orig_h // 2
        for x in range(cx - 20, cx + 20):
            for y in range(cy - 20, cy + 20):
                img.putpixel((x, y), (0, 255, 0))

        # Canvas-centre box (640x640 canvas centre = (320, 320))
        crop = crop_panel(img, (310.0, 310.0, 330.0, 330.0))
        # The crop should be roughly centred on the green block.
        w, h = crop.size
        assert crop.getpixel((w // 2, h // 2)) == (0, 255, 0)


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