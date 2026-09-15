# Hotspot Forecasting Service

Spatial forecasting for environmental report hotspots using Kernel Density Estimation (KDE) and Getis-Ord Gi* statistics.

## Installation

### Requirements
- Python 3.8+
- Supabase credentials

### Install Dependencies
```bash
pip install scipy numpy supabase geojson
```

## Configuration

Set the following environment variables:

```bash
# Service Configuration
SPATIAL_FORECAST_HOST=127.0.0.1
SPATIAL_FORECAST_PORT=8001

# Supabase Configuration
SPATIAL_FORECAST_SUPABASE_URL=your_supabase_url
SPATIAL_FORECAST_SUPABASE_KEY=your_supabase_service_role_key
```

## Running the Service

```bash
cd d:\ecopin_image_validation\hotspot_forecasting
python inference_service.py
```

The service will start on http://127.0.0.1:8001

## API Endpoints

### GET /health
Health check endpoint

### POST /forecast
Generate hotspot predictions

Request body:
```json
{
  "time_horizon": "daily|weekly|monthly",
  "bounding_box": {
    "min_lat": 14.5,
    "max_lat": 14.7,
    "min_lon": 120.9,
    "max_lon": 121.1
  }
}
```

Response:
```json
{
  "time_horizon": "weekly",
  "prediction_date": "2026-09-15T12:00:00Z",
  "total_clusters": 10,
  "hotspot_count": 3,
  "total_reports": 150,
  "cluster_analyses": {
    "cluster_id": {
      "risk_score": 0.85,
      "risk_level": "high",
      "is_hotspot": true,
      "report_count": 25,
      "gi_star": 2.5,
      "p_value": 0.01
    }
  },
  "geojson": {
    "type": "FeatureCollection",
    "features": [...]
  }
}
```

## Architecture

- **spatial_analysis.py**: KDE and Getis-Ord Gi* calculations
- **data_fetcher.py**: Supabase data fetching
- **config.py**: Configuration settings
- **inference_service.py**: HTTP server

## Integration with Node.js Backend

The Node.js backend calls this service via the `SPATIAL_FORECAST_SERVICE_URL` environment variable (default: http://127.0.0.1:8001).
