"""
main.py -- CLI entry point for the EcoPin dataset image downloader.

Usage:
    python main.py                      # Normal run (downloads pending images)
    python main.py --dry-run            # Plan without downloading
    python main.py --diagnostic         # Show row status for every row
    python main.py --test               # Test mode: process first 3 pending rows only
    python main.py --test-row 102       # Process a single specific row (by Excel row number)
    python main.py --summary            # Show progress file summary and exit
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Add this directory to the path so modules resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

import config
import progress as prog
from browser import BlockingDetectedError, BrowserManager
from downloader import process_one
from logger import Logger
from rate_limiter import RateLimitExceededError, calculate_pacing_delay
from spreadsheet import ImageRow, RowStatus, load_rows, summarise


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EcoPin dataset image downloader (browser automation)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read spreadsheet and print planned operations WITHOUT downloading",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Print per-row highlight/status analysis and exit",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process only the first 3 pending rows (smoke test)",
    )
    parser.add_argument(
        "--test-row",
        type=int,
        metavar="ROW",
        help="Process a single specific Excel row number (e.g. 102)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show the progress file summary and exit",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (overrides config.HEADLESS)",
    )
    parser.add_argument(
        "--helium",
        action="store_true",
        help="Use custom Helium browser binary (overrides config.USE_HELIUM)",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Dry-run printer
# -----------------------------------------------------------------------------

def print_dry_run(pending_rows: list[ImageRow], tracker: prog.ProgressTracker) -> None:
    Logger.section("DRY RUN - Planned operations (no downloads will be performed)")
    print(f"  {'Row':<6} {'Image ID':<14} {'Source':<20} {'Filename':<18} "
          f"{'Dest':<12} {'Status'}")
    print("  " + "-" * 100)

    for row in pending_rows:
        already_done = tracker.is_done(row.image_id)
        on_disk      = row.on_disk

        if already_done:
            planned_status = "ALREADY DONE (progress file)"
        elif on_disk:
            planned_status = "ALREADY EXISTS (on disk)"
        else:
            planned_status = "PENDING DOWNLOAD"

        dest_short = "/".join(row.destination_path.replace("\\", "/").split("/")[-3:])
        print(
            f"  {row.excel_row:<6} {row.image_id:<14} {row.source_name:<20} "
            f"{row.filename:<18} .../{dest_short:<40}  {planned_status}"
        )

    print()
    pending_count  = sum(1 for r in pending_rows if not tracker.is_done(r.image_id) and not r.on_disk)
    existing_count = sum(1 for r in pending_rows if tracker.is_done(r.image_id) or r.on_disk)
    print(f"  Would download : {pending_count}")
    print(f"  Already done   : {existing_count}")
    print(f"  Total pending  : {len(pending_rows)}")


# -----------------------------------------------------------------------------
# Main run loop
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.headless:
        config.HEADLESS = True

    if args.helium:
        config.USE_HELIUM = True

    # -- Summary mode ----------------------------------------------------------
    if args.summary:
        tracker = prog.ProgressTracker()
        tracker.print_summary()
        return

    # -- Load spreadsheet ------------------------------------------------------
    print()
    Logger.section("Loading dataset spreadsheet …")

    all_rows = load_rows(diagnostic=args.diagnostic)

    if args.diagnostic:
        summarise(all_rows)
        return

    summarise(all_rows)

    # -- Filter to rows that need attention ------------------------------------
    tracker = prog.ProgressTracker()

    if args.test_row:
        # Single-row test mode
        target_rows = [r for r in all_rows if r.excel_row == args.test_row]
        if not target_rows:
            Logger.error(f"No row found with Excel row number {args.test_row}")
            sys.exit(1)
        pending_rows = target_rows
        Logger.section(f"TEST MODE: processing single row {args.test_row}")
    else:
        # Normal: only PENDING rows
        pending_rows = [r for r in all_rows if r.status == RowStatus.PENDING]

        if args.test:
            pending_rows = pending_rows[:3]
            Logger.section(f"TEST MODE: processing first 3 pending rows")

    # -- Dry-run mode ----------------------------------------------------------
    if args.dry_run or config.DRY_RUN:
        print_dry_run(pending_rows, tracker)
        return

    # -- Sanity: skip / partial rows -------------------------------------------
    skip_rows = [r for r in all_rows if r.status == RowStatus.SKIP_PARTIAL]
    if skip_rows:
        Logger.warn(
            f"\n  Note: {len(skip_rows)} rows have an unusual partial highlight pattern "
            f"and will be SKIPPED (flagged as manual_intervention_required)."
        )
        for r in skip_rows:
            Logger.warn(f"    Row {r.excel_row}: {r.image_id} -- yellow cols={r.yellow_cols}")

    Logger.section(
        f"Starting download run: {len(pending_rows)} pending rows"
    )
    print(f"  Progress file : {config.PROGRESS_FILE}")
    print(f"  Log file      : {config.LOG_FILE}")
    print(f"  Download dir  : {config.BROWSER_DOWNLOAD_DIR}")
    print(f"  Dataset root  : {config.DATASET_RAW_DIR}")
    print()

    logger  = Logger()
    blocked = False
    aborted = False

    # Mark skip_partial rows in progress
    for r in skip_rows:
        if not tracker.is_done(r.image_id):
            tracker.upsert(
                image_id    = r.image_id,
                excel_row   = r.excel_row,
                source_name = r.source_name,
                source_url  = r.source_url,
                filename    = r.filename,
                destination = r.destination_path,
                status      = prog.STATUS_MANUAL_INTERVENTION,
                error       = f"Unusual highlight pattern: yellow_cols={r.yellow_cols}",
            )
            logger.log(
                excel_row   = r.excel_row,
                image_id    = r.image_id,
                source      = r.source_name,
                filename    = r.filename,
                status      = prog.STATUS_MANUAL_INTERVENTION,
                error       = f"Partial highlight: yellow_cols={r.yellow_cols}",
            )

    # -- Browser session -------------------------------------------------------
    try:
        with BrowserManager() as bm:
            total = len(pending_rows)
            for idx, row in enumerate(pending_rows, start=1):
                try:
                    status = process_one(row, bm, logger, tracker, idx, total)
                    Logger.info(f"  Status: {status}")

                    # Inter-image pacing delay (randomized, skip for already-existing)
                    if status not in (prog.STATUS_ALREADY_EXISTS, prog.STATUS_SKIP):
                        pace_delay = calculate_pacing_delay()
                        Logger.info(
                            f"  Pacing: waiting {pace_delay:.1f}s before next image …"
                        )
                        time.sleep(pace_delay)

                except BlockingDetectedError:
                    Logger.error(
                        "\n  [STOP] BLOCKING DETECTED -- stopping the run.\n"
                        "  Wait a while and then resume with: python main.py\n"
                        "  (Already-completed images will be skipped automatically.)"
                    )
                    blocked = True
                    break

                except RateLimitExceededError as exc:
                    Logger.warn(
                        f"\n  [RATE LIMIT PAUSE] {exc}\n"
                        "  The downloader is pausing to respect server throttling.\n"
                        "  Resume later with: python main.py\n"
                        "  (Rate-limited images will be retried automatically.)"
                    )
                    blocked = True
                    break

                except KeyboardInterrupt:
                    Logger.warn("\n  Run aborted by user.")
                    aborted = True
                    break

    except KeyboardInterrupt:
        Logger.warn("\n  Interrupted during browser startup.")

    # -- Final summary ---------------------------------------------------------
    logger.close()
    Logger.section("Run complete")
    tracker.print_summary()

    if blocked:
        Logger.warn("  The run was stopped due to rate-limiting/blocking.")
        Logger.warn("  Resume later with: python main.py")
    elif aborted:
        Logger.warn("  The run was aborted by the user.")
        Logger.warn("  Resume with: python main.py")
    else:
        Logger.success("All pending rows have been processed.")

    print(f"\n  Full log: {os.path.abspath(config.LOG_FILE)}")
    print(f"  Progress: {os.path.abspath(config.PROGRESS_FILE)}")


if __name__ == "__main__":
    main()
