"""Optional read-only smoke check against a migrated Supabase database."""
from data_fetcher import fetch_zone_history
from zone_forecaster import build_projection, utc_datetime


if __name__ == '__main__':
    now = utc_datetime()
    zones, history = fetch_zone_history('h3', now)
    for horizon in ('7d', '30d', '90d'):
        result = build_projection(zones, history, horizon, now)
        print(horizon, result['forecast_date_start'], result['forecast_date_end'],
              'zones:', result['total_regions'], 'high risk:', result['hotspot_count'])
