"""
downloader.py -- Orchestrates one image download end-to-end.

Coordinates: navigate -> download -> validate -> rename -> move -> log.
Handles retries, manual intervention prompts, and all error cases.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import config
import file_manager
import progress as prog
import validator
from browser import BlockingDetectedError, BrowserManager
from flickr import FlickrDownloadError, FlickrManualRequired, download_from_flickr
from logger import Logger
from spreadsheet import ImageRow
from wikimedia import WikimediaDownloadError, WikimediaManualRequired, download_from_wikimedia


def _find_recent_manual_download(download_dirs: list[str]) -> Optional[str]:
    """Find the most recently modified valid image file in candidate download directories."""
    candidates = []
    now = time.time()
    for d in download_dirs:
        if not os.path.isdir(d):
            continue
        for entry in os.listdir(d):
            full_path = os.path.join(d, entry)
            if not os.path.isfile(full_path):
                continue
            ext = Path(full_path).suffix.lower()
            if ext in config.VALID_IMAGE_EXTENSIONS:
                mtime = os.path.getmtime(full_path)
                # Look for files downloaded/modified in the last 20 minutes (1200 seconds)
                if (now - mtime) <= 1200:
                    candidates.append((mtime, full_path))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None


def _manual_intervention(
    row:     ImageRow,
    reason:  str,
    logger:  Logger,
    tracker: prog.ProgressTracker,
) -> str:
    """
    Pause the automation and ask the user to manually complete the download.

    Returns one of: 'done', 'skip', 'abort'
    """
    Logger.warn("\n" + "=" * 60)
    Logger.warn("  MANUAL INTERVENTION REQUIRED")
    Logger.warn("=" * 60)
    Logger.warn(f"  Row       : {row.excel_row}")
    Logger.warn(f"  Image ID  : {row.image_id}")
    Logger.warn(f"  Source    : {row.source_name}")
    Logger.warn(f"  URL       : {row.source_url}")
    Logger.warn(f"  Reason    : {reason}")
    Logger.warn(f"  Filename  : {row.filename}")
    Logger.warn(f"  Dest dir  : {row.destination_folder}")
    Logger.warn("")
    Logger.warn("  Options:")
    Logger.warn("    [d] I manually downloaded the file (or placed it in Downloads/destination) -> mark DONE")
    Logger.warn("    [s] Skip this image for now")
    Logger.warn("    [a] Abort the entire run")
    Logger.warn("=" * 60)

    tracker.upsert(
        image_id    = row.image_id,
        excel_row   = row.excel_row,
        source_name = row.source_name,
        source_url  = row.source_url,
        filename    = row.filename,
        destination = row.destination_path,
        status      = prog.STATUS_MANUAL_INTERVENTION,
        error       = reason,
    )
    logger.log(
        excel_row   = row.excel_row,
        image_id    = row.image_id,
        source      = row.source_name,
        filename    = row.filename,
        status      = prog.STATUS_MANUAL_INTERVENTION,
        destination = row.destination_folder,
        error       = reason,
    )

    while True:
        try:
            choice = input("  Your choice [d/s/a]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "abort"

        if choice == "d":
            # 1. Check if file is already placed at expected destination path
            target_path = row.destination_path
            if not os.path.isfile(target_path):
                # 2. Check temp download dir & system Downloads directory
                downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
                found = _find_recent_manual_download([config.BROWSER_DOWNLOAD_DIR, downloads_folder])
                if found:
                    Logger.info(f"  Found manually downloaded file: {found}")
                    try:
                        target_path = file_manager.move_and_rename(
                            src_path        = found,
                            destination_dir = row.destination_folder,
                            target_filename = row.filename,
                        )
                    except Exception as ex:
                        Logger.error(f"  Error moving manual download: {ex}")

            if os.path.isfile(target_path):
                ok, msg, _ = validator.validate(target_path)
                if ok:
                    Logger.success(f"  Manually downloaded file verified: {msg}")
                    tracker.upsert(
                        image_id    = row.image_id,
                        excel_row   = row.excel_row,
                        source_name = row.source_name,
                        source_url  = row.source_url,
                        filename    = row.filename,
                        destination = target_path,
                        status      = prog.STATUS_DOWNLOADED,
                    )
                    logger.log(
                        excel_row   = row.excel_row,
                        image_id    = row.image_id,
                        source      = row.source_name,
                        filename    = row.filename,
                        status      = prog.STATUS_DOWNLOADED,
                        destination = target_path,
                    )
                    return "done"
                else:
                    Logger.error(f"  File found but validation FAILED: {msg}")
                    Logger.warn("  Try again -- place a valid image file first.")
            else:
                Logger.error(f"  File not found at expected path: {row.destination_path} or in Downloads folder.")
                Logger.warn("  Please download the file in browser and then press [d] again.")

        elif choice == "s":
            tracker.upsert(
                image_id    = row.image_id,
                excel_row   = row.excel_row,
                source_name = row.source_name,
                source_url  = row.source_url,
                filename    = row.filename,
                destination = row.destination_path,
                status      = prog.STATUS_MANUAL_INTERVENTION,
                error       = f"Skipped by user. Reason: {reason}",
            )
            return "skip"

        elif choice == "a":
            return "abort"
        else:
            Logger.warn("  Please enter d, s, or a.")


def process_one(
    row:     ImageRow,
    bm:      BrowserManager,
    logger:  Logger,
    tracker: prog.ProgressTracker,
    index:   int,
    total:   int,
) -> str:
    """
    Process a single pending image row.

    Returns the final status string.
    """
    Logger.step(index, total, row.image_id, row.source_name)

    # -- Already done? ---------------------------------------------------------
    if tracker.is_done(row.image_id):
        Logger.info(f"  Already completed in progress file -- skipping.")
        return prog.STATUS_ALREADY_EXISTS

    # -- Duplicate / existing file check ---------------------------------------
    exists_valid, exist_msg = file_manager.check_existing_file(row.destination_path)
    if exists_valid:
        Logger.success(f"Already on disk: {exist_msg}")
        tracker.upsert(
            image_id    = row.image_id,
            excel_row   = row.excel_row,
            source_name = row.source_name,
            source_url  = row.source_url,
            filename    = row.filename,
            destination = row.destination_path,
            status      = prog.STATUS_ALREADY_EXISTS,
        )
        logger.log(
            excel_row   = row.excel_row,
            image_id    = row.image_id,
            source      = row.source_name,
            filename    = row.filename,
            status      = prog.STATUS_ALREADY_EXISTS,
            destination = row.destination_path,
        )
        return prog.STATUS_ALREADY_EXISTS

    # -- Validate URL ----------------------------------------------------------
    if not row.source_url or not row.source_url.startswith("http"):
        Logger.error(f"Invalid/missing URL: {row.source_url!r}")
        tracker.upsert(
            image_id    = row.image_id,
            excel_row   = row.excel_row,
            source_name = row.source_name,
            source_url  = row.source_url,
            filename    = row.filename,
            destination = row.destination_path,
            status      = prog.STATUS_INVALID_URL,
            error       = f"URL is empty or malformed: {row.source_url!r}",
        )
        logger.log(
            excel_row   = row.excel_row,
            image_id    = row.image_id,
            source      = row.source_name,
            filename    = row.filename,
            status      = prog.STATUS_INVALID_URL,
            error       = f"URL: {row.source_url!r}",
        )
        return prog.STATUS_INVALID_URL

    # -- Retry loop ------------------------------------------------------------
    attempt = 0
    last_error = ""

    while attempt <= config.MAX_RETRIES:
        if attempt > 0:
            Logger.warn(f"  Retry {attempt}/{config.MAX_RETRIES} …")
            time.sleep(config.DELAY_BETWEEN_IMAGES * 2)

        tracker.upsert(
            image_id    = row.image_id,
            excel_row   = row.excel_row,
            source_name = row.source_name,
            source_url  = row.source_url,
            filename    = row.filename,
            destination = row.destination_path,
            status      = prog.STATUS_DOWNLOADING,
            retry_count = attempt,
        )

        # Clear temp download dir before each attempt
        file_manager.clean_download_dir(config.BROWSER_DOWNLOAD_DIR)

        try:
            # -- Dispatch to source-specific handler (saves to temp download dir) --
            if row.source_name == config.SOURCE_WIKIMEDIA:
                downloaded_path = download_from_wikimedia(
                    bm, row.source_url, config.BROWSER_DOWNLOAD_DIR
                )
            elif row.source_name == config.SOURCE_FLICKR:
                downloaded_path = download_from_flickr(
                    bm, row.source_url, config.BROWSER_DOWNLOAD_DIR
                )
            else:
                last_error = f"Unknown source: {row.source_name}"
                break

            Logger.info(f"  Download complete: {downloaded_path}")

            # -- Validate temp image -------------------------------------------
            Logger.info("  Validating image ...")
            ok, val_msg, correct_ext = validator.validate(downloaded_path)

            if not ok:
                Logger.error(f"  Validation FAILED: {val_msg}")
                if os.path.isfile(downloaded_path):
                    os.remove(downloaded_path)
                last_error = f"Validation failed: {val_msg}"
                attempt += 1
                continue

            Logger.success(f"Validation passed: {val_msg}")

            # -- Determine target filename (fix extension if needed) ------------
            target_filename = row.filename
            declared_ext    = Path(target_filename).suffix.lower()
            if correct_ext and correct_ext != declared_ext:
                Logger.warn(
                    f"  Extension mismatch: declared {declared_ext}, "
                    f"actual {correct_ext} -- renaming."
                )
                target_filename = Path(target_filename).stem + correct_ext

            final_target = os.path.join(row.destination_folder, target_filename)

            # -- Collision check -----------------------------------------------
            if os.path.exists(final_target):
                Logger.error(
                    f"  CONFLICT: Destination already exists: {final_target}\n"
                    f"  Will NOT overwrite. Flagging as conflict."
                )
                if os.path.isfile(downloaded_path):
                    os.remove(downloaded_path)
                error_msg = f"Destination already exists (conflict): {final_target}"
                tracker.upsert(
                    image_id    = row.image_id,
                    excel_row   = row.excel_row,
                    source_name = row.source_name,
                    source_url  = row.source_url,
                    filename    = target_filename,
                    destination = final_target,
                    status      = prog.STATUS_ALREADY_EXISTS,
                    error       = error_msg,
                )
                logger.log(
                    excel_row   = row.excel_row,
                    image_id    = row.image_id,
                    source      = row.source_name,
                    filename    = target_filename,
                    status      = "conflict",
                    destination = final_target,
                    error       = error_msg,
                )
                return "conflict"

            # -- Move + rename from temp download dir to dataset folder --------
            Logger.info(f"  Moving to dataset directory: {final_target}")
            final_path = file_manager.move_and_rename(
                src_path        = downloaded_path,
                destination_dir = row.destination_folder,
                target_filename = target_filename,
            )

            # -- Success -------------------------------------------------------
            Logger.success(f"SUCCESS -> {final_path}")
            tracker.upsert(
                image_id    = row.image_id,
                excel_row   = row.excel_row,
                source_name = row.source_name,
                source_url  = row.source_url,
                filename    = target_filename,
                destination = final_path,
                status      = prog.STATUS_DOWNLOADED,
                retry_count = attempt,
            )
            logger.log(
                excel_row   = row.excel_row,
                image_id    = row.image_id,
                source      = row.source_name,
                filename    = target_filename,
                status      = prog.STATUS_DOWNLOADED,
                destination = final_path,
            )
            return prog.STATUS_DOWNLOADED

        except BlockingDetectedError as exc:
            Logger.error(f"  BLOCKED / RATE-LIMITED: {exc}")
            tracker.upsert(
                image_id    = row.image_id,
                excel_row   = row.excel_row,
                source_name = row.source_name,
                source_url  = row.source_url,
                filename    = row.filename,
                destination = row.destination_path,
                status      = prog.STATUS_BLOCKED,
                error       = str(exc),
            )
            logger.log(
                excel_row   = row.excel_row,
                image_id    = row.image_id,
                source      = row.source_name,
                filename    = row.filename,
                status      = prog.STATUS_BLOCKED,
                error       = str(exc)[:200],
            )
            raise  # Propagate to main -- abort the entire run

        except (WikimediaManualRequired, FlickrManualRequired) as exc:
            result = _manual_intervention(row, str(exc), logger, tracker)
            if result == "done":
                return prog.STATUS_DOWNLOADED
            elif result == "skip":
                return prog.STATUS_MANUAL_INTERVENTION
            else:  # abort
                raise KeyboardInterrupt("User chose to abort.")

        except (WikimediaDownloadError, FlickrDownloadError, file_manager.FileManagerError) as exc:
            last_error = str(exc)
            Logger.error(f"  Error (attempt {attempt}): {last_error}")
            attempt += 1

        except Exception as exc:
            last_error = f"Unexpected error: {exc}"
            Logger.error(f"  Unexpected error: {exc}")
            attempt += 1

    # -- All retries exhausted -------------------------------------------------
    Logger.error(f"  FAILED after {config.MAX_RETRIES + 1} attempts: {last_error}")
    tracker.upsert(
        image_id    = row.image_id,
        excel_row   = row.excel_row,
        source_name = row.source_name,
        source_url  = row.source_url,
        filename    = row.filename,
        destination = row.destination_path,
        status      = prog.STATUS_FAILED,
        error       = last_error,
        retry_count = attempt,
    )
    logger.log(
        excel_row   = row.excel_row,
        image_id    = row.image_id,
        source      = row.source_name,
        filename    = row.filename,
        status      = prog.STATUS_FAILED,
        error       = last_error[:200],
    )
    return prog.STATUS_FAILED
