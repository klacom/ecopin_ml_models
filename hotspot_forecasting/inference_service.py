import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from config import HOST, PORT, DEFAULT_BBOX, DBSCAN_EPS_METERS, DBSCAN_MIN_SAMPLES
from data_fetcher import fetch_reports
from spatial_analysis import run_dbscan_hotspots


class HotspotHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/forecast':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                payload = json.loads(post_data.decode('utf-8'))
                time_horizon = payload.get('time_horizon', 'weekly')
                bbox = payload.get('bounding_box', DEFAULT_BBOX)

                print(f"[Server] Forecast request: horizon={time_horizon}")
                reports = fetch_reports(time_horizon, bbox)

                print(f"[Server] Running DBSCAN on {len(reports)} reports...")
                result = run_dbscan_hotspots(reports, bbox)
                result['time_horizon'] = time_horizon

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))

            except Exception as e:
                import traceback
                print(f"[Server] Error: {e}")
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "algorithm": "DBSCAN",
                "params": {
                    "eps_meters": DBSCAN_EPS_METERS,
                    "min_samples": DBSCAN_MIN_SAMPLES,
                }
            }).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default HTTP server logs to reduce noise; we use print() above
        pass


def run(server_class=ThreadingHTTPServer, handler_class=HotspotHandler):
    server_address = (HOST, PORT)
    httpd = server_class(server_address, handler_class)
    print(f"[Server] DBSCAN Hotspot Service running on {HOST}:{PORT}")
    print(f"[Server] eps={DBSCAN_EPS_METERS}m, min_samples={DBSCAN_MIN_SAMPLES}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("[Server] Stopped.")


if __name__ == '__main__':
    run()
