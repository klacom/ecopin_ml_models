"""
Configuration for Hotspot Forecasting Module
"""

import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Service Configuration
HOST = os.environ.get("SPATIAL_FORECAST_HOST", "127.0.0.1")
PORT = int(os.environ.get("SPATIAL_FORECAST_PORT", "8001"))

# Supabase Configuration
SUPABASE_URL = os.environ.get("SPATIAL_FORECAST_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SPATIAL_FORECAST_SUPABASE_KEY")

# Time Horizon Configurations (in days)
TIME_HORIZONS = {
    "daily": 1,
    "weekly": 7,
    "monthly": 30
}

# Forecasting Configuration
FORECAST_LOOKBACK_WEEKS = 4  # Analyze patterns from past 4 weeks for forecasting
FORECAST_MIN_CONSISTENCY = 2  # Minimum weeks with hotspot status to forecast persistence

# Hotspot Thresholds
RISK_SCORE_HIGH_THRESHOLD = 0.7
RISK_SCORE_MEDIUM_THRESHOLD = 0.4
GI_STATISTIC_THRESHOLD = 1.96  # For 95% confidence level
P_VALUE_THRESHOLD = 0.05

# Spatial Analysis Settings
KDE_BANDWIDTH = 0.01  # Degrees (~1km at equator)
GRID_RESOLUTION = 0.005  # Degrees (~500m at equator)

# Report-Based Spatial Binning Settings
GRID_CELL_SIZE_METERS = 25  # Size of grid cells for spatial binning (25m for building/lot-level detection)
MIN_REPORTS_PER_HOTSPOT = 3  # Minimum reports to consider a region a hotspot

# Density-based Hotspot Thresholds
MIN_REPORTS_FOR_HIGH_RISK = 3  # High risk: 3+ reports in a grid cell
MIN_REPORTS_FOR_MEDIUM_RISK = 2  # Medium risk: 2 reports in a grid cell

# Database Query Settings
MAX_REPORTS_PER_QUERY = 10000
QUERY_TIMEOUT_SECONDS = 30

# Cache Settings
ENABLE_CACHE = True
CACHE_TTL_SECONDS = 3600  # 1 hour
