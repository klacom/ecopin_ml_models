"""
wikimedia.py -- Download handler for Wikimedia Commons pages.

Strategy:
  1. Navigate to the File: page (e.g. https://commons.wikimedia.org/wiki/File:Foo.jpg)
  2. Find the "Original file" link (the full-resolution download link)
  3. Use Playwright's expect_download() context to capture the file
  4. Save it to the temp download directory
  5. Return the path to the downloaded file

Handles:
  - Blocking / CAPTCHA detection
  - Missing download links -> manual intervention
  - Unexpected page structures -> manual intervention
  - Timeout -> retry
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from playwright.sync_api import Page, TimeoutError as PWTimeoutError

import config
from browser import BlockingDetectedError, BrowserManager
from logger import Logger


class WikimediaDownloadError(Exception):
    pass


class WikimediaManualRequired(Exception):
    pass


# Selectors to find the original-file download link on a Wikimedia File: page.
# These are tried in order; the first match wins.
_ORIGINAL_FILE_SELECTORS = [
    # The standard "Original file" link text
    "a:has-text('Original file')",
    # Fallback: the fullImageLink anchor
    "#file a",
    "div.fullImageLink a",
    ".fullMedia a",
    "a.internal",
    "a[href*='/commons/']",
    "a[href*='upload.wikimedia.org']",
]


def download_from_wikimedia(
    bm:           BrowserManager,
    source_url:   str,
    download_dir: str,
) -> str:
    """
    Navigate to `source_url` on Wikimedia Commons and download the original image into download_dir.

    Returns the absolute path to the downloaded temp file.
    Raises:
        WikimediaDownloadError   -- unrecoverable download failure
        WikimediaManualRequired  -- page structure not recognised; needs human
        BlockingDetectedError    -- rate limit / CAPTCHA detected
    """
    page: Page = bm.page

    Logger.info("Source: Wikimedia Commons")
    Logger.info(f"URL: {source_url}")

    # -- 1. Navigate ----------------------------------------------------------
    bm.navigate(source_url, wait_until="domcontentloaded")
    time.sleep(2)  # let JS settle

    # -- 2. Find download link -------------------------------------------------
    download_link = None
    for selector in _ORIGINAL_FILE_SELECTORS:
        try:
            element = page.query_selector(selector)
            if element:
                href = element.get_attribute("href") or ""
                if href and any(
                    href.lower().endswith(ext)
                    for ext in config.VALID_IMAGE_EXTENSIONS
                ):
                    download_link = element
                    Logger.info(f"  Found download link via selector: {selector}")
                    break
        except Exception:
            continue

    if download_link is None:
        # Try a broader search: any link pointing to upload.wikimedia.org
        try:
            links = page.query_selector_all("a[href*='upload.wikimedia.org']")
            if links:
                download_link = links[0]
                Logger.info("  Found download link via upload.wikimedia.org href match")
        except Exception:
            pass

    if download_link is None:
        raise WikimediaManualRequired(
            f"Could not find a download link on the Wikimedia page.\n"
            f"  URL: {source_url}\n"
            f"  Please download the original image manually."
        )

    # Extract href prior to clicking in case click causes page navigation
    href = ""
    try:
        href = download_link.get_attribute("href") or ""
    except Exception:
        pass

    os.makedirs(download_dir, exist_ok=True)
    suggested_name = "wikimedia_download.jpg"
    dest_path = os.path.join(download_dir, suggested_name)

    # -- 3. Click and capture download -----------------------------------------
    Logger.info("  Clicking download link ...")
    try:
        with page.expect_download(timeout=10_000) as dl_info:
            download_link.click()
        download = dl_info.value
        suggested_name = download.suggested_filename or suggested_name
        dest_path = os.path.join(download_dir, suggested_name)
        download.save_as(dest_path)
        Logger.info(f"  Saved download to temp folder: {dest_path}")
    except PWTimeoutError:
        Logger.warn("  expect_download timed out -- fetching image via browser request context")
        target_url = href or page.url
        if target_url:
            if not target_url.startswith("http"):
                from urllib.parse import urljoin
                target_url = urljoin(page.url, target_url)
            resp = page.request.get(target_url)
            if resp.ok:
                actual_name = os.path.basename(target_url.split("?")[0]) or suggested_name
                dest_path = os.path.join(download_dir, actual_name)
                with open(dest_path, "wb") as f:
                    f.write(resp.body())
                Logger.info(f"  Saved download directly via request context to temp folder: {dest_path}")
            else:
                raise WikimediaDownloadError(f"HTTP fetch failed ({resp.status}): {target_url}")
        else:
            raise WikimediaDownloadError("Download link had no valid URL")

    # Settle time
    time.sleep(config.POST_DOWNLOAD_SETTLE_TIME)

    if not os.path.isfile(dest_path) or os.path.getsize(dest_path) == 0:
        raise WikimediaDownloadError(
            f"Downloaded file is missing or empty: {dest_path}"
        )

    return dest_path
