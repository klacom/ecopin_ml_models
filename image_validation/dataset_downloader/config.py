"""
config.py -- All configurable constants for the EcoPin dataset downloader.

Edit this file to adjust paths, timeouts, and behavior before running.
"""

import os

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

# Absolute path to the Excel dataset spreadsheet
EXCEL_FILE = r"c:\dev\ecopin_ml_models\image_validation\ecopin_data_set.xlsx"

# Name of the worksheet that contains the dataset rows
DATASET_SHEET = "Dataset"

# Root directory of the raw image dataset
# New images will be placed into subdirectories of this folder
DATASET_RAW_DIR = r"c:\dev\datasets\ecopin_dataset\raw"

# Temporary download directory -- the browser will save files here first.
# Must be an absolute path that already exists (or will be created).
BROWSER_DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "ecopin_temp")

# Progress tracking file (relative to the script's working directory)
PROGRESS_FILE = "download_progress.json"

# Log file (CSV, relative to the script's working directory)
LOG_FILE = "download_log.csv"

# -----------------------------------------------------------------------------
# TIMING
# -----------------------------------------------------------------------------

# Seconds to wait between processing each image.
DELAY_BETWEEN_IMAGES: float = 4.0

# Minimum and maximum delay in seconds between processing each image (pacing)
MIN_DELAY_BETWEEN_IMAGES: float = 3.0
MAX_DELAY_BETWEEN_IMAGES: float = 7.0

# Jitter percentage (0.2 = ±20% variation added to delays)
JITTER_FACTOR: float = 0.2

# Initial backoff delay in seconds when throttled (e.g. HTTP 429)
RATE_LIMIT_INITIAL_BACKOFF: float = 15.0

# Multiplier for exponential backoff
RATE_LIMIT_BACKOFF_FACTOR: float = 3.0

# Maximum single backoff wait in seconds before pausing / failing gracefully
RATE_LIMIT_MAX_BACKOFF: float = 300.0

# Maximum consecutive rate limit occurrences allowed per source before pausing
MAX_CONSECUTIVE_RATE_LIMITS: int = 3

# Descriptive, honest User-Agent for Wikimedia API compliance
WIKIMEDIA_USER_AGENT: str = "EcoPinDatasetDownloader/1.0 (https://github.com/ecopin/dataset_downloader; contact@ecopin.ai) python-requests"

# User-Agent for Flickr API requests
FLICKR_USER_AGENT: str = "EcoPinDatasetDownloader/1.0 (contact@ecopin.ai)"

# Seconds to wait for a page to fully load before interacting with it.
PAGE_LOAD_TIMEOUT: float = 30.0

# Maximum seconds to wait for a file download to finish.
DOWNLOAD_TIMEOUT: float = 120.0

# How many times to retry a failed download before marking it as 'failed'.
MAX_RETRIES: int = 2

# Extra seconds to wait after a download completes before renaming/moving.
POST_DOWNLOAD_SETTLE_TIME: float = 1.5

# -----------------------------------------------------------------------------
# BROWSER
# -----------------------------------------------------------------------------

# Set to False to run the browser in visible mode (required for manual intervention).
HEADLESS: bool = False

# Browser to use: "chromium", "firefox", or "webkit"
BROWSER_TYPE: str = "chromium"

# Set to True to use custom Helium browser executable instead of Playwright bundled Chromium
USE_HELIUM: bool = False

# Path to the Helium browser executable on Windows
HELIUM_EXECUTABLE_PATH: str = r"C:\Users\murasakino\AppData\Local\imput\Helium\Application\chrome.exe"

# Dedicated automation user profile directory (prevents locking or corrupting your main daily profile)
AUTOMATION_USER_DATA_DIR: str = os.path.join(os.path.dirname(__file__), ".browser_profile")

# -----------------------------------------------------------------------------
# MODES
# -----------------------------------------------------------------------------

# When True: read spreadsheet, compute plan, print it -- but download NOTHING.
DRY_RUN: bool = False

# -----------------------------------------------------------------------------
# SPREADSHEET COLUMN INDICES  (1-based, matching openpyxl)
# -----------------------------------------------------------------------------

COL_IMAGE_ID            = 1   # A -- unique image identifier e.g. WST_001
COL_FILENAME            = 2   # B -- target filename e.g. WST_001.jpg
COL_PRIMARY_LABEL       = 3   # C -- waste / flooding / pollution / non_environmental
COL_SUBCATEGORY         = 4   # D
COL_DIFFICULTY          = 5   # E
COL_SOURCE_NAME         = 6   # F -- "Wikimedia Commons" or "Flickr"
COL_SOURCE_URL          = 7   # G -- the page URL
COL_ORIGINAL_AUTHOR     = 8   # H
COL_LICENSE             = 9   # I
COL_LICENSE_URL         = 10  # J
COL_ATTRIBUTION_REQUIRE = 11  # K
COL_DOWNLOAD_DATE       = 12  # L
COL_SOURCE_ID           = 13  # M
COL_RESOLUTION          = 14  # N
COL_QUALITY_FLAG        = 15  # O
COL_LOCATION            = 16  # P
COL_EVENT_GROUP         = 17  # Q
COL_NOTES               = 18  # R
COL_LICENSE_VERIFIED    = 19  # S
COL_APPROVED_TRAINING   = 20  # T

# Total number of data columns
TOTAL_COLS = 20

# -----------------------------------------------------------------------------
# HIGHLIGHT COLOR RULES  (ARGB format used by openpyxl)
# -----------------------------------------------------------------------------

# Green on column A  ->  row is actively marked PENDING for download
PENDING_GREEN_COLOR = "FF00FF00"

# Yellow on cells  ->  data is present / recently entered
YELLOW_COLOR = "FFFFF2CC"

# Light green on subcategory cells  ->  informational, not a status flag
SUBCATEGORY_GREEN_COLOR = "FFD9EAD3"

# Minimum number of yellow-highlighted columns to consider a row as "data-entered"
MIN_YELLOW_COLS_FOR_PENDING = 10

# -----------------------------------------------------------------------------
# LABEL -> FOLDER MAPPING
# -----------------------------------------------------------------------------

# Maps the value in column C (primary_label) to a subdirectory name under DATASET_RAW_DIR.
LABEL_FOLDER_MAP: dict = {
    "waste":              "waste",
    "flooding":           "flooding",
    "pollution":          "pollution",
    "non_environmental":  "non_environmental",
}

# -----------------------------------------------------------------------------
# SOURCE DETECTION
# -----------------------------------------------------------------------------

# Values in column F that identify each source
SOURCE_WIKIMEDIA = "Wikimedia Commons"
SOURCE_FLICKR    = "Flickr"

# -----------------------------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------------------------

# Minimum acceptable file size in bytes (smaller = suspect)
MIN_IMAGE_BYTES = 5_000

# Extensions that are always valid images
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tiff", ".tif", ".bmp"}

# -----------------------------------------------------------------------------
# BLOCKING DETECTION KEYWORDS
# -----------------------------------------------------------------------------
# If any of these strings appear in a page's title or body, stop and alert.
BLOCK_KEYWORDS = [
    "rate limit",
    "too many requests",
    "429",
    "access denied",
    "403 forbidden",
    "unusual traffic",
    "captcha",
    "robot",
    "blocked",
    "temporarily unavailable",
    "service unavailable",
    "503",
]
