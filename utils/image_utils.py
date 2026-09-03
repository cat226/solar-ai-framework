"""utils/image_utils.py — Image preprocessing helpers.

Provides reusable utilities for loading, resizing, and converting images
into the formats expected by the YOLO detector and MobileNet classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image

from utils.config import CFG
from utils.exceptions import ImageValidationError
from utils.logger import get_logger

logger = get_logger(__name__)

# Shorthand aliases from config
_YOLO_SIZE: int = CFG["models"]["yolo"]["image_size"]
_MN_SIZE: int = CFG["models"]["mobilenet"]["input_size"]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_pil_image(source: Union[str, Path, bytes]) -> Image.Image:
    """Load an image from a file path or raw bytes into a PIL Image.

    Args:
        source: File path (str or Path) or raw byte content (e.g. from an
                uploaded Streamlit file buffer).

    Returns:
        PIL.Image.Image in RGB mode.

    Raises:
        ImageValidationError: If the source is empty, of an unsupported type,
            or cannot be decoded as an image.
    """
    if source is None:
        raise ImageValidationError(
            "No image source provided. Pass a file path (str/Path) or raw bytes."
        )
    if isinstance(source, (str, Path)):
        try:
            img = Image.open(str(source)).convert("RGB")
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise ImageValidationError(
                f"Could not open image at '{source}': {exc}. "
                "Provide a readable image file (e.g. PNG or JPEG)."
            ) from exc
        logger.debug("Loaded image from path: %s", source)
    elif isinstance(source, (bytes, bytearray)):
        if len(source) == 0:
            raise ImageValidationError(
                "Image byte buffer is empty. Upload a non-empty image file."
            )
        import io
        try:
            img = Image.open(io.BytesIO(bytes(source))).convert("RGB")
        except (OSError, ValueError) as exc:
            raise ImageValidationError(
                f"Could not decode image from byte buffer: {exc}. "
                "Upload a valid image file (e.g. PNG or JPEG)."
            ) from exc
        logger.debug("Loaded image from bytes buffer (%d bytes).", len(source))
    else:
        raise ImageValidationError(
            f"Unsupported image source type '{type(source).__name__}'. "
            "Pass a file path (str/Path) or raw bytes."
        )
    return img


def resize_for_yolo(img: Image.Image) -> Image.Image:
    """Resize a PIL image to the square size expected by YOLO.

    The image is resized to ``models.yolo.image_size`` × ``image_size``
    using bilinear resampling while preserving the aspect ratio via
    letter-boxing on a grey canvas.

    Args:
        img: Source PIL image (RGB).

    Returns:
        Letter-boxed PIL image of shape (image_size, image_size, 3).
    """
    target = _YOLO_SIZE
    img_ratio = img.width / img.height
    if img_ratio > 1:
        new_w, new_h = target, int(target / img_ratio)
    else:
        new_w, new_h = int(target * img_ratio), target

    resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (target, target), (114, 114, 114))
    offset_x = (target - new_w) // 2
    offset_y = (target - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    logger.debug("Resized image for YOLO: %dx%d → %dx%d (letterbox).",
                 img.width, img.height, target, target)
    return canvas


def unletterbox_box(
    box: tuple[float, float, float, float],
    original_size: Tuple[int, int],
) -> tuple[float, float, float, float]:
    """Map a detection box from :func:`resize_for_yolo`'s 640x640 letterboxed
    coordinate space back to the original image's pixel coordinates.

    YOLO detection runs on the letterboxed canvas, so
    ``DetectionResult.boxes`` coordinates are relative to that canvas, not
    the original uploaded image. Drawing them directly on the original image
    would misplace every box. This performs the exact inverse of the scale
    and offset :func:`resize_for_yolo` applied.

    Args:
        box: (x1, y1, x2, y2) in the letterboxed 640x640 canvas.
        original_size: (width, height) of the original image the letterboxed
                       canvas was produced from.

    Returns:
        (x1, y1, x2, y2) in the original image's pixel coordinates, clamped
        to the image bounds.
    """
    orig_w, orig_h = original_size
    target = _YOLO_SIZE
    img_ratio = orig_w / orig_h
    if img_ratio > 1:
        new_w, new_h = target, int(target / img_ratio)
    else:
        new_w, new_h = int(target * img_ratio), target
    offset_x = (target - new_w) // 2
    offset_y = (target - new_h) // 2
    scale_x = orig_w / new_w
    scale_y = orig_h / new_h

    x1, y1, x2, y2 = box
    orig_x1 = (x1 - offset_x) * scale_x
    orig_y1 = (y1 - offset_y) * scale_y
    orig_x2 = (x2 - offset_x) * scale_x
    orig_y2 = (y2 - offset_y) * scale_y

    return (
        max(0.0, min(orig_w, orig_x1)),
        max(0.0, min(orig_h, orig_y1)),
        max(0.0, min(orig_w, orig_x2)),
        max(0.0, min(orig_h, orig_y2)),
    )


def resize_for_mobilenet(img: Image.Image) -> Image.Image:
    """Resize and centre-crop a PIL image to MobileNet's expected input size.

    Args:
        img: Source PIL image (RGB).

    Returns:
        PIL image of shape (input_size, input_size, 3).
    """
    size = _MN_SIZE
    # Resize shortest side to `size`, then centre-crop
    ratio = size / min(img.width, img.height)
    new_w = int(img.width * ratio)
    new_h = int(img.height * ratio)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - size) // 2
    top = (new_h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    logger.debug("Resized image for MobileNet: %dx%d (centre-crop).", size, size)
    return img


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to a uint8 NumPy array of shape (H, W, 3).

    Args:
        img: PIL image in RGB mode.

    Returns:
        NumPy uint8 array with shape (H, W, 3).
    """
    return np.array(img, dtype=np.uint8)


def get_image_dimensions(img: Image.Image) -> Tuple[int, int]:
    """Return (width, height) of a PIL image.

    Args:
        img: PIL image.

    Returns:
        Tuple of (width, height) in pixels.
    """
    return img.width, img.height
