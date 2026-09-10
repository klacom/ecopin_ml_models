"""
file_manager.py -- Rename, move, and collision-check downloaded images.

Safety rules:
  - Never rename until download is fully complete (no .crdownload / .part).
  - Never silently overwrite an existing dataset file.
  - Flag duplicate filename conflicts in the return value.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import config
from logger import Logger

log = Logger.__new__(Logger)  # used for static helpers only


# -----------------------------------------------------------------------------
# Temp-file detection helpers
# -----------------------------------------------------------------------------

_TEMP_SUFFIXES = {".crdownload", ".part", ".tmp", ".download"}


def is_temp_file(path: str) -> bool:
    """Return True if the path looks like an in-progress download file."""
    return Path(path).suffix.lower() in _TEMP_SUFFIXES


def wait_for_download_to_finish(
    download_dir: str,
    expected_stem: Optional[str] = None,
    timeout: float = config.DOWNLOAD_TIMEOUT,
    poll_interval: float = 1.0,
) -> Optional[str]:
    """
    Poll `download_dir` until a non-temporary file appears (and no temp files remain).

    Args:
        download_dir:  Directory the browser is saving into.
        expected_stem: If provided, only match files whose stem starts with this string.
        timeout:       Maximum seconds to wait.
        poll_interval: Seconds between checks.

    Returns:
        Absolute path to the completed download, or None on timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        entries = list(Path(download_dir).iterdir())
        temp_files = [e for e in entries if is_temp_file(str(e))]
        real_files = [
            e for e in entries
            if e.is_file()
            and not is_temp_file(str(e))
            and e.suffix.lower() in config.VALID_IMAGE_EXTENSIONS
        ]

        if expected_stem:
            real_files = [f for f in real_files if f.stem.startswith(expected_stem)]

        if real_files and not temp_files:
            # Sort by modification time -- newest first -- in case of leftovers
            real_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(real_files[0])

        time.sleep(poll_interval)

    return None


# -----------------------------------------------------------------------------
# Core operations
# -----------------------------------------------------------------------------

class FileManagerError(Exception):
    pass


def clean_download_dir(download_dir: str) -> None:
    """Remove all files from the temp download directory before a new download."""
    dl_path = Path(download_dir)
    dl_path.mkdir(parents=True, exist_ok=True)
    for entry in dl_path.iterdir():
        if entry.is_file():
            try:
                entry.unlink()
            except OSError:
                pass


def move_and_rename(
    src_path:        str,
    destination_dir: str,
    target_filename: str,
) -> str:
    """
    Move `src_path` into `destination_dir` and rename it to `target_filename`.

    Safety checks:
    - src_path must exist and be a non-temp file.
    - destination_dir is created if it doesn't exist.
    - If destination already exists, raises FileManagerError (no silent overwrite).

    Returns the final destination path on success.
    """
    src = Path(src_path)

    if not src.is_file():
        raise FileManagerError(f"Source file does not exist: {src_path}")

    if is_temp_file(str(src)):
        raise FileManagerError(
            f"Refusing to rename a temporary download file: {src_path}"
        )

    dest_dir  = Path(destination_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / target_filename

    if dest_path.exists():
        raise FileManagerError(
            f"Destination already exists (would overwrite): {dest_path}\n"
            f"  Source: {src_path}"
        )

    shutil.move(str(src), str(dest_path))
    return str(dest_path)


def check_existing_file(dest_path: str) -> tuple[bool, str]:
    """
    Check whether a destination file already exists and is valid.

    Returns (exists_and_valid, message).
    """
    from validator import is_valid_existing_image

    if not os.path.isfile(dest_path):
        return False, "File does not exist"

    if is_valid_existing_image(dest_path):
        size = os.path.getsize(dest_path)
        return True, f"File exists and is valid ({size:,} bytes)"
    else:
        return False, "File exists but is NOT a valid image -- will re-download"


def file_hash(path: str, algorithm: str = "sha256") -> str:
    """Compute the hash of a file for optional duplicate detection."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
