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
        ValueError: If the source type is not supported.
    """
    if isinstance(source, (str, Path)):
        img = Image.open(str(source)).convert("RGB")
        logger.debug("Loaded image from path: %s", source)
    elif isinstance(source, bytes):
        import io
        img = Image.open(io.BytesIO(source)).convert("RGB")
        logger.debug("Loaded image from bytes buffer (%d bytes).", len(source))
    else:
        raise ValueError(
            f"Unsupported source type '{type(source).__name__}'. "
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
