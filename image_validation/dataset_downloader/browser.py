"""
browser.py -- Playwright browser lifecycle management.

Provides a context manager that:
  - Launches a visible (or headless) Chromium browser
  - Configures the download directory
  - Exposes a single Page for all source handlers to use
  - Detects blocking / rate-limiting pages
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    sync_playwright,
)

import config
from logger import Logger


class BlockingDetectedError(Exception):
    """Raised when a site appears to be blocking or rate-limiting the script."""


class BrowserManager:
    """
    Manages a single Playwright browser session.

    Usage:
        with BrowserManager() as bm:
            bm.navigate("https://example.com")
            # ... interact
    """

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser:    Browser    | None = None
        self._context:    BrowserContext | None = None
        self._page:       Page        | None = None

    # -- Lifecycle -------------------------------------------------------------

    def start(self) -> None:
        import os
        os.makedirs(config.BROWSER_DOWNLOAD_DIR, exist_ok=True)
        os.makedirs(config.AUTOMATION_USER_DATA_DIR, exist_ok=True)

        self._playwright = sync_playwright().start()

        launch_kwargs = {
            "headless": config.HEADLESS,
            "downloads_path": config.BROWSER_DOWNLOAD_DIR,
            "args": [
                "--start-maximized",
                "--disable-features=DownloadBubble,DownloadBubbleV2",
            ],
        }

        browser_label = config.BROWSER_TYPE

        if getattr(config, "USE_HELIUM", False):
            helium_exe = getattr(config, "HELIUM_EXECUTABLE_PATH", "")
            if not os.path.isfile(helium_exe):
                raise FileNotFoundError(
                    f"Helium executable not found at specified path: {helium_exe}\n"
                    f"Please verify HELIUM_EXECUTABLE_PATH in config.py."
                )
            launch_kwargs["executable_path"] = helium_exe
            browser_label = "Helium (Chromium)"

        self._browser = getattr(self._playwright, config.BROWSER_TYPE).launch(**launch_kwargs)
        self._context = self._browser.new_context(
            accept_downloads=True,
            viewport=None,
        )
        self._page = self._context.new_page()

        # Set a reasonable default navigation timeout
        self._page.set_default_navigation_timeout(config.PAGE_LOAD_TIMEOUT * 1000)
        self._page.set_default_timeout(config.PAGE_LOAD_TIMEOUT * 1000)

        Logger.info(f"Browser started ({browser_label}, headless={config.HEADLESS})")

    def stop(self) -> None:
        try:
            if self._context:
                self._context.close()
        except Exception:
            pass
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        Logger.info("Browser stopped.")

    def __enter__(self) -> "BrowserManager":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # -- Page access ----------------------------------------------------------

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser not started. Use 'with BrowserManager() as bm:'")
        return self._page

    @property
    def context(self) -> BrowserContext:
        if self._context is None:
            raise RuntimeError("Browser not started.")
        return self._context

    # -- Navigation helpers ----------------------------------------------------

    def navigate(self, url: str, wait_until: str = "domcontentloaded") -> None:
        """
        Navigate to a URL and wait for the page to reach the specified state.
        Raises BlockingDetectedError if a blocking page is detected.
        """
        Logger.info(f"Opening: {url}")
        self._page.goto(url, wait_until=wait_until, timeout=config.PAGE_LOAD_TIMEOUT * 1000)
        self.check_for_blocking()

    def check_for_blocking(self) -> None:
        """
        Inspect the current page title and body for blocking / rate-limiting signals.
        Raises BlockingDetectedError if detected.
        """
        page = self._page
        try:
            title = (page.title() or "").lower()
            body_snippet = (page.inner_text("body") or "")[:2000].lower()
        except Exception:
            return  # Page may not have a body -- not necessarily blocked

        combined = title + " " + body_snippet
        for keyword in config.BLOCK_KEYWORDS:
            if keyword.lower() in combined:
                raise BlockingDetectedError(
                    f"Blocking/rate-limit detected! Keyword: '{keyword}'\n"
                    f"  Page title: {page.title()}\n"
                    f"  URL: {page.url}"
                )

    def wait_and_check(self, seconds: float) -> None:
        """Sleep for `seconds` and then check for blocking."""
        time.sleep(seconds)
        self.check_for_blocking()
