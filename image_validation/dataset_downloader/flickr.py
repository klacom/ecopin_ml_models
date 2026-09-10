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

import requests

import config
from browser import BlockingDetectedError, BrowserManager
from logger import Logger
from rate_limiter import RateLimitError, parse_retry_after


class FlickrDownloadError(Exception):
    pass


class FlickrManualRequired(Exception):
    pass


# CSS selectors for the download button on Flickr photo pages (fallback)
_DOWNLOAD_BUTTON_SELECTORS = [
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


def _download_via_oembed(source_url: str, download_dir: str) -> Optional[str]:
    """
    Query Flickr official oEmbed API to resolve a static CDN image URL.

    The oEmbed 'thumbnail_url' is always a small thumbnail (e.g. _q or _m suffix).
    We upgrade it to the large size (_b.jpg = 1024px) by replacing the size suffix.
    This is a public, unauthenticated, CDN URL -- no API key required.

    Returns local path on success, or None to trigger browser fallback.
    Raises RateLimitError if throttled (429/503).
    """
    if "flickr.com/photos/" not in source_url:
        return None

    oembed_url = f"https://www.flickr.com/services/oembed/?url={source_url}&format=json"
    headers = {"User-Agent": config.FLICKR_USER_AGENT}

    try:
        res = requests.get(oembed_url, headers=headers, timeout=15.0)

        if res.status_code in (429, 503):
            retry_after = parse_retry_after(res.headers.get("Retry-After"))
            raise RateLimitError(
                f"Flickr oEmbed API returned HTTP {res.status_code} (Rate Limited)",
                retry_after=retry_after,
                source=config.SOURCE_FLICKR,
            )

        if not res.ok:
            Logger.warn(f"  Flickr oEmbed API returned HTTP {res.status_code} -- using browser fallback")
            return None

        data = res.json()

        # oEmbed 'thumbnail_url' is a small CDN image like:
        #   https://live.staticflickr.com/SERVERID/PHOTOID_HASH_q.jpg   (75x75)
        # Upgrade to _b suffix (longest edge = 1024px, always public)
        thumb_url = data.get("thumbnail_url", "")
        if not thumb_url or "staticflickr.com" not in thumb_url:
            Logger.warn("  Flickr oEmbed did not return a staticflickr.com thumbnail -- using browser fallback")
            return None

        import re
        # Replace size suffix: _q, _m, _s, _t, _n, _w, _z, _c, _l, _h, _k, _o -> _b
        large_url = re.sub(r"_[qmstznwchlko]\.jpg$", "_b.jpg", thumb_url)
        if large_url == thumb_url:
            # No recognisable suffix to upgrade; strip any suffix and try _b
            large_url = re.sub(r"_[^._]+\.jpg$", "_b.jpg", thumb_url)

        Logger.info(f"  Resolved large CDN URL via Flickr oEmbed API: {large_url}")

        # Stream direct image download from static CDN
        img_res = requests.get(large_url, headers=headers, stream=True, timeout=30.0)

        if img_res.status_code in (429, 503):
            retry_after = parse_retry_after(img_res.headers.get("Retry-After"))
            raise RateLimitError(
                f"Flickr CDN returned HTTP {img_res.status_code} (Rate Limited)",
                retry_after=retry_after,
                source=config.SOURCE_FLICKR,
            )

        if img_res.status_code == 403:
            # Photo is access-restricted (private/friend-only); cannot download without login
            Logger.warn("  Flickr CDN returned 403 -- photo may be private or restricted. Using browser fallback.")
            return None

        if not img_res.ok:
            Logger.warn(f"  Flickr CDN returned HTTP {img_res.status_code} for large URL -- using browser fallback")
            return None

        os.makedirs(download_dir, exist_ok=True)
        filename = os.path.basename(large_url.split("?")[0]) or "flickr_download.jpg"
        dest_path = os.path.join(download_dir, filename)

        with open(dest_path, "wb") as f:
            for chunk in img_res.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

        if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
            Logger.info(f"  Downloaded via Flickr oEmbed+CDN: {dest_path}")
            return dest_path

    except RateLimitError:
        raise
    except requests.exceptions.ConnectionError as ex:
        Logger.warn(f"  Flickr oEmbed connection error ({ex}) -- using browser fallback")
    except Exception as ex:
        Logger.warn(f"  Flickr oEmbed API resolution encountered error ({ex}) -- using browser fallback")

    return None


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
        RateLimitError         -- server throttling / 429 response
    """
    Logger.info("Source: Flickr")
    Logger.info(f"URL: {source_url}")

    # -- 1. Prefer Flickr oEmbed API (fast, lightweight, compliant) -----------
    api_path = _download_via_oembed(source_url, download_dir)
    if api_path:
        return api_path

    # -- 2. Browser Automation Fallback ---------------------------------------
    page: Page = bm.page
    bm.navigate(source_url, wait_until="domcontentloaded")
    time.sleep(3)

    page_text = ""
    try:
        page_text = (page.inner_text("body") or "")[:3000].lower()
    except Exception:
        pass

    if "429" in page_text or "too many requests" in page_text:
        raise RateLimitError(
            "Flickr page returned 429 / Too Many Requests",
            retry_after=60.0,
            source=config.SOURCE_FLICKR,
        )

    if "sign in" in page_text and "download" not in page_text:
        raise FlickrManualRequired(
            "Flickr is showing a login wall. Please sign in manually and then confirm."
        )

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

    try:
        download_btn.click(timeout=10_000)
        time.sleep(1.5)
    except PWTimeoutError:
        raise FlickrDownloadError("Timed out clicking the download button.")

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
        try:
            links = page.query_selector_all("a[href*='live.staticflickr.com']")
            if links:
                size_link = links[-1]
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

    time.sleep(config.POST_DOWNLOAD_SETTLE_TIME)

    if not os.path.isfile(dest_path) or os.path.getsize(dest_path) == 0:
        raise FlickrDownloadError(
            f"Downloaded file is missing or empty: {dest_path}"
        )

    return dest_path
