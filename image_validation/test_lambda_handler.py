import base64
import importlib
import json
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch


class LambdaContractTests(unittest.TestCase):
    def setUp(self):
        self.model = types.SimpleNamespace(_model_lock=threading.Lock(), run_inference=Mock(
            return_value={'predicted_class': 'VALID', 'decision': 'APPROVED', 'confidence': 0.9}))
        with patch.dict(sys.modules, {'inference_service_exp32': self.model}):
            self.handler = importlib.reload(importlib.import_module('lambda_handler'))

    def event(self, body='', method='POST', **extra):
        return {'requestContext': {'http': {'method': method}}, 'body': body, **extra}

    def test_existing_api_gateway_json_contract(self):
        image = b'fixture-image'
        result = self.handler.handler(self.event(json.dumps({'image': base64.b64encode(image).decode()})), None)
        self.assertEqual(result['statusCode'], 200)
        self.assertEqual(json.loads(result['body'])['decision'], 'APPROVED')
        self.model.run_inference.assert_called_once_with(image)

    def test_gateway_encoded_body_and_health(self):
        body = base64.b64encode(json.dumps({'image': 'YQ=='}).encode()).decode()
        self.assertEqual(self.handler.handler(self.event(body, isBase64Encoded=True), None)['statusCode'], 200)
        self.assertEqual(self.handler.handler(self.event(method='GET'), None)['statusCode'], 200)

    def test_invalid_inputs_never_reach_model(self):
        for body in ['[]', '{}', '{', '{"image":"!invalid!"}', '{"image":""}']:
            self.assertEqual(self.handler.handler(self.event(body), None)['statusCode'], 400)
        self.assertEqual(self.handler.handler(self.event('x' * (6 * 1024 * 1024 + 1)), None)['statusCode'], 413)
        self.model.run_inference.assert_not_called()

    def test_inference_failure_does_not_leak_internal_details(self):
        self.model.run_inference.side_effect = RuntimeError('private-checkpoint-path')
        result = self.handler.handler(self.event('{"image":"YQ=="}'), None)
        self.assertEqual(result['statusCode'], 500)
        self.assertNotIn('private-checkpoint-path', result['body'])
