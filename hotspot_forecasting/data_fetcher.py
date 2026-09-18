from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, TIME_HORIZONS
from datetime import datetime, timedelta
import struct

def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise ValueError("Supabase credentials not configured.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def parse_wkb_location(hex_str):
    """Parse a PostGIS WKB hex string and return (lon, lat)."""
    if not hex_str or len(hex_str) < 50:
        return None, None
    try:
        lon_hex = hex_str[18:34]
        lat_hex = hex_str[34:50]
        lon = struct.unpack('<d', bytes.fromhex(lon_hex))[0]
        lat = struct.unpack('<d', bytes.fromhex(lat_hex))[0]
        return lon, lat
    except Exception:
        return None, None

def fetch_reports(time_horizon: str, bbox=None):
    """
    Fetch verified/completed reports from Supabase within the time horizon.
    Only validated reports are used to ensure cluster quality.
    """
    supabase = get_supabase_client()

    days_back = TIME_HORIZONS.get(time_horizon, 7)
    start_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

    # Only fetch verified or completed reports — exclude noise from unvalidated submissions
    response = (
        supabase.table('reports')
        .select('id, location, created_at, issue_type, status')
        .gte('created_at', start_date)
        .in_('status', ['verified', 'completed', 'resolved'])
        .execute()
    )
    reports = response.data

    # Fall back to all reports if no verified ones found (e.g., dev environment)
    if not reports:
        print(f"[DataFetcher] No verified reports found, falling back to all reports")
        response = (
            supabase.table('reports')
            .select('id, location, created_at, issue_type, status')
            .gte('created_at', start_date)
            .execute()
        )
        reports = response.data

    print(f"[DataFetcher] Fetched {len(reports)} reports for {time_horizon} horizon")

    # Parse WKB location to latitude/longitude
    parsed_reports = []
    for r in reports:
        if not r.get('location'):
            continue

        lon, lat = parse_wkb_location(r['location'])
        if lon is None or lat is None:
            print(f"[DataFetcher] Failed to parse location for report {r.get('id')}")
            continue

        # Sanity check: valid coordinates
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            continue

        # Optional bounding box filter
        if bbox:
            if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                continue

        r['longitude'] = lon
        r['latitude'] = lat
        parsed_reports.append(r)

    print(f"[DataFetcher] Parsed {len(parsed_reports)} reports with valid coordinates")
    return parsed_reports
