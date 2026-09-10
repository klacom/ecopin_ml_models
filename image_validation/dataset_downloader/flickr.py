"""
flickr.py -- Download handler for Flickr photo pages.

Strategy:
  1. Navigate to the Flickr photo page (e.g. https://www.flickr.com/photos/user/id/)
  2. Click the download arrow (↓) button to reveal the size menu
  3. Select "Original" (highest quality) -- or fall back to "Large" if no original
  4. Capture the download via Playwright's expect_download()
  5. Return the path to the downloaded file

Handles:
  - Pages that require login to download -> manual intervention
  - Missing download button -> manual intervention
  - CAPTCHA / blocking -> stop and alert
"""

from __future__ import annotations

import os
import time
from typing import Optional

from playwright.sync_api import Page, TimeoutError as PWTimeoutError

import config
from browser import BlockingDetectedError, BrowserManager
from logger import Logger


class FlickrDownloadError(Exception):
    pass


class FlickrManualRequired(Exception):
    pass


# CSS selectors for the download button on Flickr photo pages
_DOWNLOAD_BUTTON_SELECTORS = [
    # The download arrow icon button (data-testid attribute)
    "[data-testid='download-icon']",
    "button[title*='Download' i]",
    "a[title*='Download' i]",
    ".download-icon",
    "a[download]",
    "button[aria-label*='download' i]",
    "a.download-button",
    "button.download-button",
    "i.download-icon",
    "i.ui-icon-download",
    ".photo-engagement-bar .download",
    ".engagement-bar button:has(svg)",
    "button:has-text('Download')",
    "a:has-text('Download')",
]

# Selectors for the size options in the dropdown menu
_SIZE_OPTION_SELECTORS = [
    "a:has-text('Original')",
    "li:has-text('Original') a",
    "a[title*='Original' i]",
    "a:has-text('Large')",
    "li:has-text('Large') a",
    "a:has-text('2048')",
    "a:has-text('1600')",
    "a:has-text('1024')",
    ".download-dropdown a",
    "[data-download-url]",
]


def download_from_flickr(
    bm:           BrowserManager,
    source_url:   str,
    download_dir: str,
) -> str:
    """
    Navigate to `source_url` on Flickr and download the highest-quality image into download_dir.

    Returns the absolute path to the downloaded temp file.
    Raises:
        FlickrDownloadError    -- unrecoverable download failure
        FlickrManualRequired   -- page structure not recognised / login required
        BlockingDetectedError  -- rate limit / CAPTCHA detected
    """
    page: Page = bm.page

    Logger.info("Source: Flickr")
    Logger.info(f"URL: {source_url}")

    # -- 1. Navigate ----------------------------------------------------------
    bm.navigate(source_url, wait_until="domcontentloaded")
    time.sleep(3)  # Flickr is JS-heavy; wait for dynamic content

    # -- 2. Check for login wall ------------------------------------------------
    page_text = ""
    try:
        page_text = (page.inner_text("body") or "")[:3000].lower()
    except Exception:
        pass

    if "sign in" in page_text and "download" not in page_text:
        raise FlickrManualRequired(
            "Flickr is showing a login wall. Please sign in manually and then confirm."
        )

    # -- 3. Click the download button ------------------------------------------
    download_btn = None
    for selector in _DOWNLOAD_BUTTON_SELECTORS:
        try:
            el = page.query_selector(selector)
            if el and el.is_visible():
                download_btn = el
                Logger.info(f"  Found download button via: {selector}")
                break
        except Exception:
            continue

    if download_btn is None:
        # Try by ARIA role
        try:
            btns = page.get_by_role("button").filter(has_text="Download").all()
            if btns:
                download_btn = btns[0]
                Logger.info("  Found download button via ARIA role")
        except Exception:
            pass

    if download_btn is None:
        raise FlickrManualRequired(
            f"Could not find the download button on the Flickr page.\n"
            f"  URL: {source_url}\n"
            f"  Please download the original image manually."
        )

    # Click to open the size dropdown
    try:
        download_btn.click(timeout=10_000)
        time.sleep(1.5)
    except PWTimeoutError:
        raise FlickrDownloadError("Timed out clicking the download button.")

    # -- 4. Select size option -------------------------------------------------
    size_link = None
    for selector in _SIZE_OPTION_SELECTORS:
        try:
            el = page.query_selector(selector)
            if el and el.is_visible():
                size_link = el
                label = el.inner_text().strip() if hasattr(el, "inner_text") else selector
                Logger.info(f"  Selecting size: {label}")
                break
        except Exception:
            continue

    if size_link is None:
        # Last resort: look for direct image download URLs in href attributes
        try:
            links = page.query_selector_all("a[href*='live.staticflickr.com']")
            if links:
                size_link = links[-1]  # usually the largest
                Logger.info("  Found static Flickr CDN link as fallback")
        except Exception:
            pass

    if size_link is None:
        raise FlickrManualRequired(
            f"Could not find a size/download option on the Flickr page.\n"
            f"  URL: {source_url}\n"
            f"  This may require login or manual download."
        )

    os.makedirs(download_dir, exist_ok=True)

    # -- 5. Click the size link and capture download ---------------------------
    Logger.info("  Initiating download ...")
    try:
        with page.expect_download(timeout=config.DOWNLOAD_TIMEOUT * 1000) as dl_info:
            size_link.click()
        download = dl_info.value
    except PWTimeoutError:
        Logger.warn("  expect_download timed out -- trying direct URL approach")
        try:
            href = size_link.get_attribute("href") or ""
            if href and any(ext in href for ext in [".jpg", ".png", ".gif", ".webp"]):
                with page.expect_download(timeout=config.DOWNLOAD_TIMEOUT * 1000) as dl_info2:
                    page.goto(href)
                download = dl_info2.value
            else:
                raise FlickrDownloadError(
                    f"Download timed out and no direct image URL found.\n"
                    f"  URL: {source_url}"
                )
        except Exception as ex2:
            raise FlickrDownloadError(
                f"Download timed out after {config.DOWNLOAD_TIMEOUT}s.\n"
                f"  URL: {source_url}\n  {ex2}"
            )

    suggested_name = download.suggested_filename or "flickr_download.jpg"
    dest_path = os.path.join(download_dir, suggested_name)
    download.save_as(dest_path)
    Logger.info(f"  Saved download to temp folder: {dest_path}")

    # Settle time
    time.sleep(config.POST_DOWNLOAD_SETTLE_TIME)

    if not os.path.isfile(dest_path) or os.path.getsize(dest_path) == 0:
        raise FlickrDownloadError(
            f"Downloaded file is missing or empty: {dest_path}"
        )

    return dest_path
