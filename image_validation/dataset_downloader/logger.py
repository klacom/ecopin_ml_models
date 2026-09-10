"""
logger.py -- CSV and console logging for every download operation.
"""

from __future__ import annotations

import csv
import os
import sys
import threading
from datetime import datetime

import config

_HEADERS = ["timestamp", "excel_row", "image_id", "source", "filename", "status", "destination", "error"]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Logger:
    """
    Writes a CSV log file and prints progress to the terminal.
    Thread-safe.
    """

    def __init__(self, filepath: str = config.LOG_FILE) -> None:
        self._path  = filepath
        self._lock  = threading.Lock()
        self._write_header()

    def _write_header(self) -> None:
        write_header = not os.path.isfile(self._path)
        self._file = open(self._path, "a", newline="", encoding="utf-8")  # noqa: WPS515
        self._writer = csv.DictWriter(self._file, fieldnames=_HEADERS)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def log(
        self,
        excel_row:   int,
        image_id:    str,
        source:      str,
        filename:    str,
        status:      str,
        destination: str = "",
        error:       str = "",
    ) -> None:
        row = {
            "timestamp":   _now(),
            "excel_row":   excel_row,
            "image_id":    image_id,
            "source":      source,
            "filename":    filename,
            "status":      status,
            "destination": destination,
            "error":       error,
        }
        with self._lock:
            self._writer.writerow(row)
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._file.close()
            except Exception:
                pass

    # -- Console helpers -------------------------------------------------------

    @staticmethod
    def section(text: str) -> None:
        print(f"\n{'=' * 60}")
        print(f"  {text}")
        print(f"{'=' * 60}")

    @staticmethod
    def info(text: str) -> None:
        print(f"  {text}")

    @staticmethod
    def success(text: str) -> None:
        print(f"  [OK]  {text}")

    @staticmethod
    def warn(text: str) -> None:
        print(f"  [!!]  {text}", file=sys.stderr)

    @staticmethod
    def error(text: str) -> None:
        print(f"  [XX]  {text}", file=sys.stderr)

    @staticmethod
    def step(index: int, total: int, image_id: str, source: str) -> None:
        print(f"\n[{index}/{total}] {image_id}  ({source})")
