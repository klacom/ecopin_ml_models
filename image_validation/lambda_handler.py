"""Keep the existing API Gateway JSON contract while using the Exp 3.2 model."""
import base64
import binascii
import json

import inference_service_exp32 as model


def reply(status, body):
    return {'statusCode': status, 'headers': {'content-type': 'application/json'},
            'body': json.dumps(body, allow_nan=False)}


def handler(event, context):
    method = event.get('requestContext', {}).get('http', {}).get('method', event.get('httpMethod'))
    if method == 'GET':
        return reply(200, {'status': 'ok', 'experiment': '3.2', 'model': 'RegMLP'})
    if method != 'POST':
        return reply(405, {'error': 'Method not allowed'})
    try:
        body = event.get('body') or ''
        if len(body) > 6 * 1024 * 1024:
            return reply(413, {'error': 'Image payload too large'})
        if event.get('isBase64Encoded'):
            body = base64.b64decode(body, validate=True)
        payload = json.loads(body)
        encoded = payload.get('image') if isinstance(payload, dict) else None
        if not isinstance(encoded, str) or not encoded:
            raise ValueError('image must contain base64-encoded image bytes')
        image = base64.b64decode(encoded, validate=True)
        if not image:
            raise ValueError('Image is empty')
    except (ValueError, TypeError, binascii.Error):
        return reply(400, {'error': 'Expected a JSON object with a valid base64 image'})
    try:
        with model._model_lock:
            return reply(200, model.run_inference(image))
    except Exception:
        return reply(500, {'error': 'Image inference failed'})
