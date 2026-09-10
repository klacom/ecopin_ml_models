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

import urllib.parse
import requests

import config
from browser import BlockingDetectedError, BrowserManager
from logger import Logger
from rate_limiter import RateLimitError, parse_retry_after


class WikimediaDownloadError(Exception):
    pass


class WikimediaManualRequired(Exception):
    pass


# Selectors to find the original-file download link on a Wikimedia File: page (fallback).
_ORIGINAL_FILE_SELECTORS = [
    "a:has-text('Original file')",
    "#file a",
    "div.fullImageLink a",
    ".fullMedia a",
    "a.internal",
    "a[href*='/commons/']",
    "a[href*='upload.wikimedia.org']",
]


def _safe_temp_filename(image_url: str, fallback: str = "wikimedia_download.jpg") -> str:
    """
    Derive a short, Windows-safe temporary filename from a Wikimedia image URL.

    We ONLY use the file extension from the URL -- never the Wikimedia source
    filename, which may be percent-encoded, contain Unicode characters, or exceed
    Windows' MAX_PATH limit.

    The returned name is always short (e.g. 'wikimedia_download.jpg') and safe
    to use as a local temp path on Windows.
    """
    try:
        # Decode the URL first to handle percent-encoding, then take extension only
        decoded_path = urllib.parse.unquote(image_url.split("?")[0])
        ext = Path(decoded_path).suffix.lower()
        if ext in config.VALID_IMAGE_EXTENSIONS:
            return f"wikimedia_download{ext}"
    except Exception:
        pass
    return fallback


def _download_via_action_api(source_url: str, download_dir: str) -> Optional[str]:
    """
    Query Wikimedia Action API with a compliant User-Agent to resolve direct image URL.

    Returns local path on success, or None on fallback.
    Raises:
        RateLimitError          -- server returned HTTP 429/503 (retryable)
        WikimediaDownloadError  -- API resolved the URL but saving locally failed (do NOT open browser)
    """
    # Extract file title (e.g. File:Garbage_on_road.jpg)
    if "/wiki/" not in source_url:
        return None

    raw_title = source_url.split("/wiki/")[-1].split("#")[0]
    title = urllib.parse.unquote(raw_title)

    api_url = (
        f"https://commons.wikimedia.org/w/api.php"
        f"?action=query&titles={urllib.parse.quote(title)}"
        f"&prop=imageinfo&iiprop=url&format=json"
    )
    headers = {"User-Agent": config.WIKIMEDIA_USER_AGENT}

    # -- Step 1: resolve the direct CDN image URL via the API ------------------
    image_url: Optional[str] = None
    try:
        res = requests.get(api_url, headers=headers, timeout=15.0)

        if res.status_code in (429, 503):
            retry_after = parse_retry_after(res.headers.get("Retry-After"))
            raise RateLimitError(
                f"Wikimedia API returned HTTP {res.status_code} (Rate Limited)",
                retry_after=retry_after,
                source=config.SOURCE_WIKIMEDIA,
            )

        if not res.ok:
            Logger.warn(f"  Wikimedia API returned HTTP {res.status_code} -- using browser fallback")
            return None

        data = res.json()
        pages = data.get("query", {}).get("pages", {})
        for _page_id, page_data in pages.items():
            imageinfo = page_data.get("imageinfo", [])
            if imageinfo and "url" in imageinfo[0]:
                image_url = imageinfo[0]["url"]
                break

    except RateLimitError:
        raise
    except requests.exceptions.ConnectionError as ex:
        Logger.warn(f"  Wikimedia API connection error ({ex}) -- using browser fallback")
        return None
    except Exception as ex:
        Logger.warn(f"  Wikimedia API query error ({ex}) -- using browser fallback")
        return None

    if not image_url:
        Logger.warn("  Wikimedia API returned no image URL -- using browser fallback")
        return None

    Logger.info(f"  Resolved direct image URL via Wikimedia API: {image_url}")

    # -- Step 2: download the image from the CDN --------------------------------
    # Use a SHORT, SAFE temp filename -- never use the Wikimedia source filename.
    # Wikimedia filenames can be percent-encoded Unicode (e.g. Thai) and hundreds
    # of characters long, which causes [Errno 22] Invalid argument on Windows.
    safe_name = _safe_temp_filename(image_url)
    os.makedirs(download_dir, exist_ok=True)
    dest_path = os.path.join(download_dir, safe_name)

    try:
        img_res = requests.get(image_url, headers=headers, stream=True, timeout=60.0)

        if img_res.status_code in (429, 503):
            retry_after = parse_retry_after(img_res.headers.get("Retry-After"))
            raise RateLimitError(
                f"Wikimedia CDN returned HTTP {img_res.status_code} (Rate Limited)",
                retry_after=retry_after,
                source=config.SOURCE_WIKIMEDIA,
            )

        if not img_res.ok:
            # CDN request failed -- browser might do better (different auth/redirect)
            Logger.warn(f"  Wikimedia CDN returned HTTP {img_res.status_code} -- using browser fallback")
            return None

        with open(dest_path, "wb") as f:
            for chunk in img_res.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

    except RateLimitError:
        raise
    except (OSError, IOError) as ex:
        # Local filesystem error (e.g. path too long, invalid char, disk full).
        # This is NOT a network/API failure -- do NOT fall back to the browser,
        # which would load Wikimedia and risk triggering the block-keyword detector.
        raise WikimediaDownloadError(
            f"Local file save failed for '{safe_name}' in '{download_dir}': {ex}\n"
            f"  API had already resolved the image URL successfully."
        ) from ex
    except requests.exceptions.ConnectionError as ex:
        Logger.warn(f"  Wikimedia CDN connection error ({ex}) -- using browser fallback")
        return None
    except Exception as ex:
        Logger.warn(f"  Wikimedia CDN download error ({ex}) -- using browser fallback")
        return None

    if not os.path.isfile(dest_path) or os.path.getsize(dest_path) == 0:
        raise WikimediaDownloadError(
            f"Wikimedia API download produced an empty file: {dest_path}"
        )

    Logger.info(f"  Downloaded via Wikimedia Action API -> temp: {dest_path}")
    return dest_path


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
        RateLimitError           -- server throttling / 429 response
    """
    Logger.info("Source: Wikimedia Commons")
    Logger.info(f"URL: {source_url}")

    # -- 1. Prefer Wikimedia Action API (fast, lightweight, policy compliant) --
    api_path = _download_via_action_api(source_url, download_dir)
    if api_path:
        return api_path

    # -- 2. Browser Automation Fallback ---------------------------------------
    page: Page = bm.page
    bm.navigate(source_url, wait_until="domcontentloaded")
    time.sleep(2)

    # Check browser page for rate limit / 429
    try:
        body_text = (page.inner_text("body") or "")[:2000].lower()
        if "429" in body_text or "too many requests" in body_text:
            raise RateLimitError(
                "Wikimedia page returned 429 / Too Many Requests",
                retry_after=60.0,
                source=config.SOURCE_WIKIMEDIA,
            )
    except RateLimitError:
        raise  # Always propagate rate-limit signals
    except Exception:
        pass  # Suppress DOM/page-read errors only


    # Find download link
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

    href = ""
    try:
        href = download_link.get_attribute("href") or ""
    except Exception:
        pass

    os.makedirs(download_dir, exist_ok=True)
    # Always use a short, safe temp filename -- never the raw Wikimedia URL basename
    suggested_name = "wikimedia_download.jpg"
    dest_path = os.path.join(download_dir, suggested_name)

    Logger.info("  Clicking download link ...")
    try:
        with page.expect_download(timeout=10_000) as dl_info:
            download_link.click()
        download = dl_info.value
        # Playwright's suggested_filename can also be the long Unicode Wikimedia name;
        # use it only to extract the extension, then keep our safe stem.
        pw_suggested = download.suggested_filename or ""
        ext = Path(pw_suggested).suffix.lower()
        if ext in config.VALID_IMAGE_EXTENSIONS:
            suggested_name = f"wikimedia_download{ext}"
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
            if resp.status in (429, 503):
                raise RateLimitError(
                    f"Wikimedia returned HTTP {resp.status} (Rate Limited)",
                    retry_after=60.0,
                    source=config.SOURCE_WIKIMEDIA,
                )
            if resp.ok:
                # Use safe temp name (extension from URL only)
                safe_ext = Path(urllib.parse.unquote(target_url.split("?")[0])).suffix.lower()
                if safe_ext in config.VALID_IMAGE_EXTENSIONS:
                    suggested_name = f"wikimedia_download{safe_ext}"
                dest_path = os.path.join(download_dir, suggested_name)
                with open(dest_path, "wb") as f:
                    f.write(resp.body())
                Logger.info(f"  Saved download via request context to temp folder: {dest_path}")
            else:
                raise WikimediaDownloadError(f"HTTP fetch failed ({resp.status}): {target_url}")
        else:
            raise WikimediaDownloadError("Download link had no valid URL")

    time.sleep(config.POST_DOWNLOAD_SETTLE_TIME)

    if not os.path.isfile(dest_path) or os.path.getsize(dest_path) == 0:
        raise WikimediaDownloadError(
            f"Downloaded file is missing or empty: {dest_path}"
        )

    return dest_path
