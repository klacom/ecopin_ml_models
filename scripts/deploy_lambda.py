"""Deploy one ECR image and restore the previous image if the smoke test fails."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path


SERVICES = {
    'forecasting': ('ecopin-temp-hotspot-forecasting-aka-spatial-analysis', 'POST', '/forecast/project', 200),
    'vrp': ('ecopin-vrp-service', 'POST', '/solve_vrp', 400),
    'image-validation': ('clip-validator', 'GET', '/clip-validator', 200),
}


def aws(*arguments):
    result = subprocess.run(['aws', *arguments, '--region', 'ap-southeast-1', '--output', 'json'],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('service', choices=SERVICES)
    parser.add_argument('image')
    args = parser.parse_args()
    function, method, path, status = SERVICES[args.service]
    previous = aws('lambda', 'get-function', '--function-name', function)
    revision = previous['Configuration']['RevisionId']
    old_image = previous['Code']['ResolvedImageUri']
    event = {'version': '2.0', 'routeKey': f'{method} {path}', 'rawPath': path,
             'rawQueryString': '', 'headers': {'content-type': 'application/json'},
             'requestContext': {'http': {'method': method, 'path': path,
                'sourceIp': '127.0.0.1', 'protocol': 'HTTP/1.1'}},
             'body': json.dumps({'horizon': '7d', 'zone_type': 'h3'}) if args.service == 'forecasting'
                 else '{}' if method == 'POST' else '', 'isBase64Encoded': False}
    changed = False
    try:
        aws('lambda', 'update-function-code', '--function-name', function,
            '--image-uri', args.image, '--revision-id', revision)
        changed = True
        aws('lambda', 'wait', 'function-updated-v2', '--function-name', function)
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / 'request.json'
            output = Path(directory) / 'response.json'
            request.write_text(json.dumps(event), encoding='utf-8')
            metadata = aws('lambda', 'invoke', '--function-name', function,
                '--payload', 'fileb://' + str(request), '--cli-read-timeout', '180', str(output))
            result = json.loads(output.read_text())
            if metadata.get('FunctionError') or result.get('statusCode') != status:
                raise RuntimeError(f'{function} failed smoke test: {result.get("statusCode")}')
            body = json.loads(result['body'])
            if args.service == 'forecasting' and body.get('model_version') != 'zone-timeseries-v1':
                raise RuntimeError('Unexpected forecasting model version')
            if args.service == 'forecasting' and not body.get('region_analyses'):
                raise RuntimeError('Forecast database smoke test returned no stable zones')
            if args.service == 'image-validation' and body.get('experiment') != '3.2':
                raise RuntimeError('Unexpected image validation experiment')
        print(f'Healthy Lambda deployment: {function}')
    except Exception:
        if changed:
            current = aws('lambda', 'get-function-configuration', '--function-name', function)
            aws('lambda', 'update-function-code', '--function-name', function,
                '--image-uri', old_image, '--revision-id', current['RevisionId'])
            aws('lambda', 'wait', 'function-updated-v2', '--function-name', function)
            print(f'Restored previous image: {function}')
        raise


if __name__ == '__main__':
    main()
