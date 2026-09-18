import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Server Settings
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8001"))

# Supabase Credentials
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("WARNING: Supabase credentials not found in environment variables.")

# Time Horizons (in days)
TIME_HORIZONS = {
    "daily": 1,
    "weekly": 7,
    "monthly": 30
}

# DBSCAN Clustering Parameters
# eps: neighborhood radius — reports within this many meters are considered neighbors
DBSCAN_EPS_METERS = float(os.environ.get("DBSCAN_EPS_METERS", "150"))
# min_samples: minimum reports in a neighborhood to form a core point
DBSCAN_MIN_SAMPLES = int(os.environ.get("DBSCAN_MIN_SAMPLES", "3"))

# Risk Classification Thresholds (by report count)
RISK_HIGH_THRESHOLD = int(os.environ.get("RISK_HIGH_THRESHOLD", "10"))
RISK_MEDIUM_THRESHOLD = int(os.environ.get("RISK_MEDIUM_THRESHOLD", "5"))

# Bounding Box (Philippines default if none provided)
# [min_lon, min_lat, max_lon, max_lat]
DEFAULT_BBOX = [116.9, 4.6, 126.6, 21.4]
