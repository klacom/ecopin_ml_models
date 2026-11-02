import json
import logging
import hmac
from threading import BoundedSemaphore
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from config import HOST, PORT, SPATIAL_ANALYTICS_TOKEN
from data_fetcher import fetch_zone_history, fetch_pattern_history
from historical_patterns import build_pattern_analysis

PATTERN_SLOT = BoundedSemaphore(1)
from zone_forecaster import HORIZONS, MODEL_VERSION, build_projection, utc_datetime


class HotspotHandler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        path = urlsplit(self.path).path
        if path == '/patterns/analyze':
            return self.analyze_patterns()
        if path not in ('/forecast/project', '/forecast', '/feedback'):
            return self.send_json(404, {'error': 'Not found'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 1024 * 1024:
                raise ValueError('Request must contain a JSON object under 1 MB')
            payload = json.loads(self.rfile.read(length).decode('utf-8'))
            if not isinstance(payload, dict):
                raise ValueError('Request must be a JSON object')
            if path == '/feedback':
                try:
                    from training_loop import handle_feedback
                    handle_feedback(payload)
                    return self.send_json(200, {'status': 'updated'})
                except Exception:
                    logging.exception('Hotspot feedback failed')
                    return self.send_json(500, {'error': 'Unable to process feedback'})
            default_horizon = '7d' if path == '/forecast' else '30d'
            horizon = payload.get('horizon', payload.get('time_horizon', default_horizon))
            if not isinstance(horizon, str):
                raise ValueError('horizon must be 7d, 30d, or 90d')
            if path == '/forecast':
                horizon = {'daily': '7d', 'weekly': '7d', 'monthly': '30d'}.get(horizon, horizon)
            if horizon not in HORIZONS:
                raise ValueError('horizon must be 7d, 30d, or 90d')
            zone_type = payload.get('zone_type', 'h3')
            if zone_type not in ('h3', 'barangay'):
                raise ValueError('zone_type must be h3 or barangay')
        except (ValueError, UnicodeDecodeError) as error:
            return self.send_json(400, {'error': str(error)})
        try:
            now = utc_datetime()
            zones, history = fetch_zone_history(zone_type, now)
            return self.send_json(200, build_projection(zones, history, horizon, now))
        except Exception:
            logging.exception('Hotspot projection failed')
            return self.send_json(500, {'error': 'Unable to generate hotspot forecast'})

    def analyze_patterns(self):
        if not SPATIAL_ANALYTICS_TOKEN:
            return self.send_json(503, {'error': 'Historical analysis requires server setup'})
        authorization = self.headers.get('Authorization', '')
        if not hmac.compare_digest(authorization.encode(), ('Bearer ' + SPATIAL_ANALYTICS_TOKEN).encode()):
            return self.send_json(403, {'error': 'Forbidden'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 1024:
                raise ValueError()
            if json.loads(self.rfile.read(length).decode('utf-8')) != {}:
                raise ValueError()
        except (ValueError, UnicodeDecodeError):
            return self.send_json(400, {'error': 'Send an empty JSON object'})
        if not PATTERN_SLOT.acquire(blocking=False):
            return self.send_json(503, {'error': 'Historical analysis is busy; retry shortly'})
        try:
            result = build_pattern_analysis(fetch_pattern_history())
            return self.send_json(200, result)
        except Exception as error:
            # Never include database details, report data or credentials in responses.
            logging.error('Historical analysis failed (%s)', type(error).__name__)
            message = ('History exceeds the processing limit; no points were sampled'
                       if getattr(error, 'code', None) == '54000'
                       else 'Historical analysis unavailable; verify database and service setup')
            return self.send_json(503, {'error': message})
        finally:
            PATTERN_SLOT.release()

    def do_GET(self):
        if urlsplit(self.path).path != '/health':
            return self.send_json(404, {'error': 'Not found'})
        self.send_json(200, {'status': 'ok', 'algorithm': 'Holt damped weekly trend',
                             'model_version': MODEL_VERSION, 'history_weeks': 52,
                             'horizons': list(HORIZONS), 'risk_score_type': 'bounded risk index'})

    def log_message(self, format, *args):
        pass


def run(server_class=ThreadingHTTPServer, handler_class=HotspotHandler):
    server = server_class((HOST, PORT), handler_class)
    print(f'[Server] Forward hotspot forecast service running on {HOST}:{PORT}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    run()
