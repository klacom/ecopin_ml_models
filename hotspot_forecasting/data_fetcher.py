"""
Data Fetcher for Hotspot Forecasting
Fetches report data from Supabase for spatial analysis
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import logging
from shapely import wkb

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from supabase import create_client, Client
except ImportError:
    print("Warning: supabase-py not installed. Install with: pip install supabase")
    raise

from config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    TIME_HORIZONS,
    MAX_REPORTS_PER_QUERY,
    QUERY_TIMEOUT_SECONDS,
    FORECAST_LOOKBACK_WEEKS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches report data from Supabase for spatial analysis"""
    
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
        
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("DataFetcher initialized with Supabase")
    
    def fetch_reports_by_time_horizon(
        self,
        time_horizon: str = "weekly",
        bounding_box: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Fetch reports within a time horizon for hotspot analysis
        
        Args:
            time_horizon: 'daily', 'weekly', or 'monthly'
            bounding_box: Optional dict with 'min_lat', 'max_lat', 'min_lon', 'max_lon'
        
        Returns:
            List of report dictionaries with location, timestamp, and metadata
        """
        if time_horizon not in TIME_HORIZONS:
            raise ValueError(f"Invalid time_horizon: {time_horizon}. Must be one of {list(TIME_HORIZONS.keys())}")
        
        days = TIME_HORIZONS[time_horizon]
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Fetching reports for {time_horizon} horizon ({days} days) from {start_date} to {end_date}")
        
        # Build query - fetch all reports, not cluster-filtered
        query = (
            self.client
            .table('reports')
            .select('id, location, created_at, issue_type, status')
            .gte('created_at', start_date.isoformat())
            .lte('created_at', end_date.isoformat())
            .order('created_at', desc=True)
            .limit(MAX_REPORTS_PER_QUERY)
        )
        
        # Apply bounding box filter if provided
        if bounding_box:
            # PostGIS ST_MakeEnvelope for bounding box
            bbox_filter = (
                f"ST_Within(location, "
                f"ST_MakeEnvelope({bounding_box['min_lon']}, {bounding_box['min_lat']}, "
                f"{bounding_box['max_lon']}, {bounding_box['max_lat']}, 4326))"
            )
            query = query.filter('location', 'cs', bbox_filter)  # Custom filter
        
        try:
            response = query.execute()
            reports = response.data
            
            # Parse location from PostGIS binary format (EWKB)
            for report in reports:
                if report.get('location'):
                    try:
                        # PostGIS returns location as EWKB hex string
                        loc_hex = report['location']
                        if isinstance(loc_hex, str):
                            # Parse EWKB using shapely
                            point = wkb.loads(bytes.fromhex(loc_hex))
                            report['longitude'] = point.x
                            report['latitude'] = point.y
                        elif isinstance(loc_hex, bytes):
                            # Parse EWKB from bytes
                            point = wkb.loads(loc_hex)
                            report['longitude'] = point.x
                            report['latitude'] = point.y
                    except Exception as e:
                        logger.warning(f"Failed to parse location for report {report.get('id')}: {e}")
                else:
                    logger.warning(f"Report {report.get('id')} has no location data")
            
            logger.info(f"Fetched {len(reports)} reports")
            return reports
            
        except Exception as e:
            logger.error(f"Error fetching reports: {e}")
            return []
    
    def fetch_historical_reports_for_forecast(
        self,
        weeks_back: int = 4
    ) -> Dict[str, List[Dict]]:
        """
        Fetch reports from multiple weeks for forecasting analysis
        
        Args:
            weeks_back: Number of weeks to look back (default 4)
        
        Returns:
            Dict mapping week identifier to list of reports
        """
        try:
            historical_data = {}
            end_date = datetime.now(timezone.utc)
            
            for week in range(weeks_back):
                week_start = end_date - timedelta(weeks=week + 1)
                week_end = end_date - timedelta(weeks=week)
                
                week_key = f"week_{week}"
                
                logger.info(f"Fetching reports for {week_key}: {week_start.isoformat()} to {week_end.isoformat()}")
                
                query = (
                    self.client
                    .table('reports')
                    .select('id, location, created_at, issue_type, status')
                    .gte('created_at', week_start.isoformat())
                    .lte('created_at', week_end.isoformat())
                    .order('created_at', desc=True)
                    .limit(MAX_REPORTS_PER_QUERY)
                )
                
                response = query.execute()
                
                if response.data:
                    reports = response.data
                    
                    # Parse location data
                    for report in reports:
                        if report.get('location'):
                            try:
                                loc_hex = report['location']
                                if isinstance(loc_hex, str):
                                    point = wkb.loads(bytes.fromhex(loc_hex))
                                    report['longitude'] = point.x
                                    report['latitude'] = point.y
                                elif isinstance(loc_hex, bytes):
                                    point = wkb.loads(loc_hex)
                                    report['longitude'] = point.x
                                    report['latitude'] = point.y
                            except Exception as e:
                                logger.warning(f"Failed to parse location for report {report.get('id')}: {e}")
                    
                    historical_data[week_key] = reports
                    logger.info(f"Fetched {len(reports)} reports for {week_key}")
                else:
                    historical_data[week_key] = []
                    logger.info(f"No reports found for {week_key}")
            
            return historical_data
        except Exception as e:
            logger.error(f"Error fetching historical reports: {e}")
            raise


# Singleton instance
_fetcher_instance = None

def get_data_fetcher() -> DataFetcher:
    """Get or create singleton DataFetcher instance"""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = DataFetcher()
    return _fetcher_instance
