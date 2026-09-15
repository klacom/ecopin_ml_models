"""
HTTP Inference Service for Hotspot Forecasting
Provides REST API endpoints for spatial hotspot predictions
Report-based approach - analyzes reports directly without cluster dependency
"""

import os
import sys
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import HOST, PORT, TIME_HORIZONS
from data_fetcher import get_data_fetcher
from spatial_analysis import get_spatial_analyzer

# Diagnostic helpers
def _sep(char="─", width=64):
    sys.stderr.write(char * width + "\n")
    sys.stderr.flush()

def _diag(msg):
    sys.stderr.write(f"[hotspot] {msg}\n")
    sys.stderr.flush()

# Startup
_sep("═")
sys.stderr.write("[hotspot] EcoPin Hotspot Forecasting Service — Starting Up\n")
_sep("═")
sys.stderr.write(f"[hotspot] Host           : {HOST}\n")
sys.stderr.write(f"[hotspot] Port           : {PORT}\n")
sys.stderr.write(f"[hotspot] Time horizons  : {list(TIME_HORIZONS.keys())}\n")
sys.stderr.write("[hotspot] Note: Services initialized on-demand in request handlers\n")
_sep()

# Initialize services in request handlers (ThreadingHTTPServer context issue workaround)
_analyzer_lock = threading.Lock()

_sep("═")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[hotspot] %s - %s\n" % (self.address_string(), fmt % args))
        sys.stderr.flush()
    
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    
    def _read_body(self):
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            return None
        try:
            length = int(length_header)
        except ValueError:
            return None
        if length <= 0:
            return b""
        return self.rfile.read(length)
    
    def do_GET(self):
        sys.stderr.write(f"[hotspot] do_GET called with path: {self.path}\n")
        sys.stderr.flush()
        if self.path == "/health":
            # Initialize services for health check
            try:
                sys.stderr.write("[hotspot] Health check: Starting service initialization...\n")
                sys.stderr.flush()
                data_fetcher = get_data_fetcher()
                sys.stderr.write(f"[hotspot] Health check: data_fetcher = {data_fetcher}\n")
                sys.stderr.flush()
                spatial_analyzer = get_spatial_analyzer()
                sys.stderr.write(f"[hotspot] Health check: spatial_analyzer = {spatial_analyzer}\n")
                sys.stderr.flush()
                self._send_json(200, {
                    "status": "ok",
                    "service": "hotspot_forecasting",
                    "time_horizons": list(TIME_HORIZONS.keys()),
                    "data_fetcher_ready": data_fetcher is not None,
                    "spatial_analyzer_ready": spatial_analyzer is not None
                })
            except Exception as exc:
                import traceback
                sys.stderr.write(f"[hotspot] Health check: Exception: {exc}\n")
                sys.stderr.write(f"[hotspot] Health check: Traceback: {traceback.format_exc()}\n")
                sys.stderr.flush()
                self._send_json(500, {"error": f"Service initialization failed: {exc}", "traceback": traceback.format_exc()})
        else:
            self._send_json(404, {"error": "Not found"})
    
    def do_POST(self):
        sys.stderr.write(f"[hotspot] do_POST called with path: {self.path}\n")
        if self.path == "/forecast":
            _sep()
            _diag(f"POST /forecast from {self.address_string()}")
            
            # Initialize services in handler
            data_fetcher = None
            spatial_analyzer = None
            try:
                data_fetcher = get_data_fetcher()
                spatial_analyzer = get_spatial_analyzer()
                _diag("  Services initialized")
            except Exception as exc:
                _diag(f"  Result: ✗ ERROR - Service initialization failed: {exc}")
                import traceback as tb
                _diag(f"  Traceback: {tb.format_exc()}")
                self._send_json(500, {"error": f"Service initialization failed: {exc}", "traceback": tb.format_exc()})
                _sep()
                return
            
            body = self._read_body()
            if not body:
                _diag("  Body read: ✗ Empty body")
                self._send_json(400, {"error": "Empty request body"})
                _sep()
                return
            
            try:
                params = json.loads(body.decode("utf-8"))
                _diag(f"  Parameters: {json.dumps(params, indent=2)}")
            except json.JSONDecodeError as exc:
                _diag(f"  Body read: ✗ JSON decode error - {exc}")
                self._send_json(400, {"error": "Invalid JSON"})
                _sep()
                return
            
            time_horizon = params.get("time_horizon", "weekly")
            bounding_box = params.get("bounding_box")
            
            if time_horizon not in TIME_HORIZONS:
                _diag(f"  Result: ✗ Invalid time_horizon: {time_horizon}")
                self._send_json(400, {
                    "error": f"Invalid time_horizon. Must be one of {list(TIME_HORIZONS.keys())}"
                })
                _sep()
                return
            
            try:
                with _analyzer_lock:
                    _diag("  Stage: Fetching reports")
                    reports = data_fetcher.fetch_reports_by_time_horizon(
                        time_horizon=time_horizon,
                        bounding_box=bounding_box
                    )
                    _diag(f"  Stage: Fetched {len(reports)} reports")
                    
                    _diag("  Stage: Fetching historical data for forecasting")
                    historical_reports = data_fetcher.fetch_historical_reports_for_forecast(weeks_back=4)
                    _diag(f"  Stage: Fetched historical data for {len(historical_reports)} weeks")
                    
                    _diag("  Stage: Forecasting hotspots based on persistence patterns")
                    forecast_analyses = spatial_analyzer.forecast_hotspots_persistence(historical_reports)
                    _diag(f"  Stage: Forecasted {len(forecast_analyses)} persistent hotspots")
                    
                    _diag("  Stage: Analyzing current reports for comparison")
                    current_hotspots = spatial_analyzer.analyze_report_hotspots(reports)
                    _diag(f"  Stage: Analyzed {len(current_hotspots)} current hotspot regions")
                    
                    _diag("  Stage: Generating heatmap GeoJSON")
                    heatmap_geojson = spatial_analyzer.generate_heatmap_geojson(reports)
                    _diag("  Stage: Heatmap GeoJSON generated")
                    
                    _diag("  Stage: Generating forecast GeoJSON")
                    geojson = spatial_analyzer.generate_hotspot_geojson(current_hotspots)
                    _diag("  Stage: Forecast GeoJSON generated")
                    
                    # Count forecasted hotspots
                    forecast_count = len(forecast_analyses)
                    current_count = sum(1 for a in current_hotspots.values() if a.get('is_hotspot', False))
                    
                    # Merge forecast analyses into region_analyses for compatibility
                    merged_analyses = current_hotspots.copy()
                    for region_id, forecast in forecast_analyses.items():
                        if region_id not in merged_analyses:
                            merged_analyses[region_id] = {
                                'region_id': region_id,
                                'risk_score': forecast['confidence'],
                                'risk_level': forecast['predicted_risk_level'],
                                'is_hotspot': forecast['is_forecasted'],
                                'report_count': forecast['predicted_report_count'],
                                'gi_star': 0.0,
                                'p_value': 1.0,
                                'is_significant': forecast['is_forecasted'],
                                'region_center_lat': forecast['region_center_lat'],
                                'region_center_lng': forecast['region_center_lng'],
                                'region_radius_meters': 25,
                                'is_forecasted': True,
                                'confidence': forecast['confidence'],
                                'persistence_weeks': forecast['persistence_weeks']
                            }
                        else:
                            # Add forecast info to existing analysis
                            merged_analyses[region_id]['is_forecasted'] = True
                            merged_analyses[region_id]['confidence'] = forecast['confidence']
                            merged_analyses[region_id]['persistence_weeks'] = forecast['persistence_weeks']
                    
                    result = {
                        "time_horizon": time_horizon,
                        "prediction_date": datetime.now(timezone.utc).isoformat(),
                        "total_regions": len(merged_analyses),
                        "hotspot_count": forecast_count,  # Forecasted hotspots
                        "current_hotspot_count": current_count,  # Current hotspots
                        "total_reports": len(reports),
                        "region_analyses": merged_analyses,
                        "forecast_analyses": forecast_analyses,
                        "geojson": geojson,
                        "heatmap_geojson": heatmap_geojson,
                        "forecast_type": "persistence"
                    }
                    
                    _diag(f"  Result: 200 OK - {forecast_count} forecasted hotspots, {current_count} current hotspots")
                    _sep()
                    self._send_json(200, result)
                    
            except Exception as exc:
                _diag(f"  Result: ✗ ERROR - {exc}")
                _diag(traceback.format_exc())
                _sep()
                self._send_json(500, {"error": f"Forecast failed: {exc}"})
        
        else:
            self._send_json(404, {"error": "Not found"})


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    sys.stderr.write(f"[hotspot] Listening on http://{HOST}:{PORT}\n")
    sys.stderr.write(f"[hotspot]   GET  /health\n")
    sys.stderr.write(f"[hotspot]   POST /forecast\n")
    _sep("═")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\n[hotspot] Shutting down.\n")
        server.server_close()


if __name__ == "__main__":
    main()
