"""
validator.py -- Post-download image integrity verification.

Checks that a downloaded file is:
  - Not zero bytes / too small
  - Not an HTML error page
  - Openable by Pillow (actually an image)
  - Has the correct extension for its format
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, UnidentifiedImageError

import config

# Map Pillow format strings to canonical extensions
_PILLOW_FORMAT_TO_EXT: dict[str, str] = {
    "JPEG":    ".jpg",
    "PNG":     ".png",
    "GIF":     ".gif",
    "WEBP":    ".webp",
    "TIFF":    ".tiff",
    "BMP":     ".bmp",
    "ICO":     ".ico",
    "PPM":     ".ppm",
}


class ValidationError(Exception):
    """Raised when a downloaded file fails validation."""


def validate(filepath: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate a downloaded file.

    Returns:
        (ok, message, correct_extension)
        - ok:                True if the file is a valid image
        - message:           Human-readable result description
        - correct_extension: The canonical extension for the image format
                             (e.g. ".jpg"), or None on failure
    """
    path = Path(filepath)

    # 1. File must exist
    if not path.is_file():
        return False, f"File not found: {filepath}", None

    # 2. Minimum size check
    size_bytes = path.stat().st_size
    if size_bytes < config.MIN_IMAGE_BYTES:
        return False, f"File too small ({size_bytes} bytes) -- likely corrupt or an error page", None

    # 3. HTML sniff  (error pages from Wikimedia/Flickr)
    try:
        with open(filepath, "rb") as f:
            header = f.read(512)
        if b"<!DOCTYPE" in header or b"<html" in header.lower():
            return False, "Downloaded file appears to be an HTML error page", None
    except OSError as exc:
        return False, f"Could not read file header: {exc}", None

    # 4. Pillow open
    try:
        with Image.open(filepath) as img:
            img.verify()
        # Re-open to get format after verify (verify closes the file)
        with Image.open(filepath) as img:
            pillow_format = img.format or ""
            img_width, img_height = img.size
    except (UnidentifiedImageError, Exception) as exc:
        return False, f"Pillow cannot open file: {exc}", None

    # 5. Extension check
    correct_ext = _PILLOW_FORMAT_TO_EXT.get(pillow_format.upper(), path.suffix.lower())
    declared_ext = path.suffix.lower()
    if declared_ext not in config.VALID_IMAGE_EXTENSIONS:
        return False, f"Unrecognised extension: {declared_ext}", None

    msg = (
        f"Valid image: {pillow_format} {img_width}×{img_height}, "
        f"{size_bytes:,} bytes"
    )
    return True, msg, correct_ext


def is_valid_existing_image(filepath: str) -> bool:
    """Quick check -- True if the path points to an already-valid image file."""
    ok, _, _ = validate(filepath)
    return ok
