import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from inference_service import HotspotHandler
from test_zone_forecaster import ZONE


class ForecastHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), HotspotHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, payload, path='/forecast/project', method='POST'):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port)
        connection.request(method, path, json.dumps(payload) if payload is not None else None,
                           {'Content-Type': 'application/json'})
        response = connection.getresponse()
        result = (response.status, json.loads(response.read()))
        connection.close()
        return result

    @patch('inference_service.fetch_zone_history', return_value=([ZONE], []))
    def test_forward_contract_and_legacy_alias(self, fetch):
        status, result = self.request({'horizon': '90d', 'zone_type': 'h3'})
        self.assertEqual(status, 200)
        self.assertEqual(result['horizon_days'], 90)
        status, result = self.request({'time_horizon': 'weekly'}, '/forecast')
        self.assertEqual(status, 200)
        self.assertEqual(result['horizon'], '7d')
        status, result = self.request({}, '/forecast')
        self.assertEqual(status, 200)
        self.assertEqual(result['horizon'], '7d')

    @patch('inference_service.fetch_zone_history', return_value=([ZONE], []))
    def test_invalid_input_is_bad_request(self, fetch):
        for payload in ({'horizon': 'yesterday'}, {'zone_type': 'unknown'}, [], {'horizon': []}):
            self.assertEqual(self.request(payload)[0], 400)

    @patch('inference_service.fetch_zone_history', side_effect=RuntimeError('secret database information'))
    def test_internal_error_does_not_leak_database_details(self, fetch):
        status, result = self.request({'horizon': '30d'})
        self.assertEqual(status, 500)
        self.assertNotIn('secret', json.dumps(result))

    def test_health_identifies_forecast_model(self):
        status, result = self.request(None, '/health', 'GET')
        self.assertEqual(status, 200)
        self.assertEqual(result['model_version'], 'zone-timeseries-v1')

    @patch('training_loop.handle_feedback')
    def test_feedback_compatibility_and_safe_failure(self, feedback):
        self.assertEqual(self.request({'task_id': 'task'}, '/feedback'), (200, {'status': 'updated'}))
        feedback.side_effect = RuntimeError('private information')
        self.assertEqual(self.request({'task_id': 'task'}, '/feedback')[0], 500)


if __name__ == '__main__':
    unittest.main()
