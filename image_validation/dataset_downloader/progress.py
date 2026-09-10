"""
progress.py -- Persistent progress tracking using a JSON file.

Records the status of every image download attempt so the tool
can resume correctly after interruption.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Optional

import config


# -----------------------------------------------------------------------------
# Status constants  (stored as plain strings in the JSON)
# -----------------------------------------------------------------------------

STATUS_PENDING                   = "pending"
STATUS_DOWNLOADING               = "downloading"
STATUS_DOWNLOADED                = "downloaded"
STATUS_ALREADY_EXISTS            = "already_exists"
STATUS_FAILED                    = "failed"
STATUS_BLOCKED                   = "blocked"
STATUS_RATE_LIMITED              = "rate_limited"
STATUS_INVALID_URL               = "invalid_url"
STATUS_INVALID_DOWNLOAD          = "invalid_download"
STATUS_MANUAL_INTERVENTION       = "manual_intervention_required"
STATUS_SKIP                      = "skip"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ProgressTracker:
    """
    Thread-safe progress tracker backed by a JSON file.

    JSON structure:
    {
        "WST_001": {
            "image_id": "WST_001",
            "excel_row": 2,
            "source_name": "Wikimedia Commons",
            "source_url": "https://...",
            "filename": "WST_001.jpg",
            "destination": "C:/.../.../WST_001.jpg",
            "status": "downloaded",
            "timestamp": "2026-09-10T16:00:00",
            "error": "",
            "retry_count": 0
        },
        ...
    }
    """

    def __init__(self, filepath: str = config.PROGRESS_FILE) -> None:
        self._path  = filepath
        self._lock  = threading.Lock()
        self._data: dict[str, dict] = {}
        self._load()

    # -- I/O ------------------------------------------------------------------

    def _load(self) -> None:
        if os.path.isfile(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                print(f"[progress] Loaded {len(self._data)} existing records from {self._path}")
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[progress] WARNING: Could not load progress file ({exc}). Starting fresh.")
                self._data = {}

    def _save(self) -> None:
        """Write the current state to disk (called after every update)."""
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            print(f"[progress] ERROR: Failed to save progress file: {exc}")

    # -- Public API ------------------------------------------------------------

    def is_done(self, image_id: str) -> bool:
        """Return True if this image has a terminal success status."""
        rec = self._data.get(image_id)
        if rec is None:
            return False
        return rec.get("status") in (STATUS_DOWNLOADED, STATUS_ALREADY_EXISTS, STATUS_SKIP)

    def get_status(self, image_id: str) -> Optional[str]:
        rec = self._data.get(image_id)
        return rec.get("status") if rec else None

    def get_retry_count(self, image_id: str) -> int:
        rec = self._data.get(image_id)
        return rec.get("retry_count", 0) if rec else 0

    def upsert(
        self,
        image_id:    str,
        excel_row:   int,
        source_name: str,
        source_url:  str,
        filename:    str,
        destination: str,
        status:      str,
        error:       str = "",
        retry_count: int = 0,
    ) -> None:
        """Create or update a record and immediately persist to disk."""
        with self._lock:
            existing = self._data.get(image_id, {})
            self._data[image_id] = {
                "image_id":    image_id,
                "excel_row":   excel_row,
                "source_name": source_name,
                "source_url":  source_url,
                "filename":    filename,
                "destination": destination,
                "status":      status,
                "timestamp":   _now(),
                "error":       error,
                "retry_count": retry_count,
            }
            self._save()

    def increment_retry(self, image_id: str) -> int:
        """Increment the retry counter and return the new value."""
        with self._lock:
            rec = self._data.get(image_id, {})
            new_count = rec.get("retry_count", 0) + 1
            if image_id in self._data:
                self._data[image_id]["retry_count"] = new_count
                self._data[image_id]["timestamp"] = _now()
            self._save()
            return new_count

    def all_records(self) -> list[dict]:
        with self._lock:
            return list(self._data.values())

    def print_summary(self) -> None:
        from collections import Counter
        counts = Counter(r.get("status", "unknown") for r in self._data.values())
        print()
        print("  Progress file summary:")
        for status, count in sorted(counts.items()):
            print(f"    {status:<35} {count:4d}")
        print()
