"""
rate_limiter.py -- Rate limiting, exponential backoff, jitter, and Retry-After handling.

Provides compliant rate limit detection, conservative download pacing with jitter,
exponential backoff, and graceful session pausing when servers request throttling.
"""

from __future__ import annotations

import math
import random
import time
from datetime import datetime
from typing import Optional

import config
from logger import Logger


class RateLimitError(Exception):
    """Raised when a server returns a throttling or rate limit response (e.g. HTTP 429 / 503)."""

    def __init__(self, message: str, retry_after: Optional[float] = None, source: str = "") -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.source = source


class RateLimitExceededError(Exception):
    """Raised when consecutive rate limits exceed the maximum safety threshold, pausing the run."""
    pass


def parse_retry_after(header_val: Optional[str]) -> Optional[float]:
    """
    Parse an HTTP Retry-After header.

    Can be:
      - Seconds integer (e.g. '120')
      - HTTP Date string (e.g. 'Wed, 21 Oct 2026 07:28:00 GMT')
    """
    if not header_val:
        return None

    header_str = str(header_val).strip()

    # Case 1: Integer seconds
    if header_str.isdigit():
        return float(header_str)

    # Case 2: HTTP date
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(header_str)
        now_dt = datetime.now(dt.tzinfo)
        diff = (dt - now_dt).total_seconds()
        return max(1.0, diff) if diff > 0 else 1.0
    except Exception:
        return None


def calculate_pacing_delay(
    min_delay: float = config.MIN_DELAY_BETWEEN_IMAGES,
    max_delay: float = config.MAX_DELAY_BETWEEN_IMAGES,
    jitter_factor: float = config.JITTER_FACTOR,
) -> float:
    """
    Calculate a randomized pacing delay with jitter between download requests.
    """
    base_delay = random.uniform(min_delay, max_delay)
    jitter = base_delay * random.uniform(-jitter_factor, jitter_factor)
    return max(0.5, base_delay + jitter)


def calculate_backoff(
    attempt: int,
    retry_after: Optional[float] = None,
    initial_backoff: float = config.RATE_LIMIT_INITIAL_BACKOFF,
    backoff_factor: float = config.RATE_LIMIT_BACKOFF_FACTOR,
    max_backoff: float = config.RATE_LIMIT_MAX_BACKOFF,
    jitter_factor: float = config.JITTER_FACTOR,
) -> float:
    """
    Calculate exponential backoff duration, incorporating Retry-After header if present.
    """
    if retry_after is not None and retry_after > 0:
        delay = retry_after
        Logger.info(f"  [rate-limit] Respecting server 'Retry-After' header: {retry_after:.1f}s")
    else:
        delay = initial_backoff * (backoff_factor ** max(0, attempt - 1))

    delay = min(delay, max_backoff)
    # Add randomized jitter (+-20%) to avoid predictable pulse patterns
    jitter = delay * random.uniform(-jitter_factor, jitter_factor)
    final_delay = max(1.0, delay + jitter)
    return final_delay


class RateLimitTracker:
    """
    Tracks consecutive rate limits per source (Wikimedia / Flickr) to prevent hammering servers.
    """

    def __init__(self, max_consecutive: int = config.MAX_CONSECUTIVE_RATE_LIMITS) -> None:
        self.max_consecutive = max_consecutive
        self._counts: dict[str, int] = {}

    def record_rate_limit(self, source: str) -> int:
        """Record a rate limit occurrence for a source and return the new consecutive count."""
        count = self._counts.get(source, 0) + 1
        self._counts[source] = count
        if count >= self.max_consecutive:
            raise RateLimitExceededError(
                f"Source '{source}' rate limited {count} consecutive times.\n"
                f"  Pausing downloader to respect server throttling. Resume later with: python main.py"
            )
        return count

    def record_success(self, source: str) -> None:
        """Reset consecutive rate limit count on successful request."""
        self._counts[source] = 0
