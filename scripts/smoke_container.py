"""Exercise real image startup and inference before pushing anything to ECR."""
import argparse
import base64
import json
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('service', choices=['forecasting', 'vrp', 'image-validation'])
    args = parser.parse_args()
    native = args.service == 'image-validation'
    port = {'forecasting': 8002, 'vrp': 8003, 'image-validation': 8080}[args.service]
    identifier = subprocess.check_output(['docker', 'run', '-d', '--read-only', '--tmpfs', '/tmp',
        '-p', f'127.0.0.1:9080:{port}', 'service:test'], text=True).strip()
    try:
        url = 'http://127.0.0.1:9080'
        if native:
            url += '/2015-03-31/functions/function/invocations'
            body = json.dumps({'requestContext': {'http': {'method': 'GET'}}}).encode()
            request = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
        elif args.service == 'vrp':
            request = urllib.request.Request(url + '/solve_vrp', data=b'{}', headers={'Content-Type': 'application/json'})
        else:
            request = urllib.request.Request(url + '/health')
        for attempt in range(90):
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    result = json.load(response)
                    if native:
                        assert result['statusCode'] == 200, result
                        assert json.loads(result['body'])['experiment'] == '3.2'
                    else:
                        assert result['model_version'] == 'zone-timeseries-v1'
                    break
            except urllib.error.HTTPError as error:
                if args.service == 'vrp' and error.code == 400:
                    break
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(2)
        else:
            raise RuntimeError('Container did not become healthy')
        if native:
            # A valid PNG exercises CLIP, the classifier and response serialization.
            png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='
            event = {'requestContext': {'http': {'method': 'POST'}}, 'body': json.dumps({'image': png})}
            request = urllib.request.Request(url, data=json.dumps(event).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
            assert result['statusCode'] == 200, result
            prediction = json.loads(result['body'])
            assert prediction['decision'] in ['APPROVED', 'REJECTED', 'MANUAL_REVIEW']
            assert 0 <= prediction['confidence'] <= 1
        print(f'Container smoke test passed: {args.service}')
    except Exception:
        subprocess.run(['docker', 'logs', '--tail', '60', identifier], check=False)
        raise
    finally:
        subprocess.run(['docker', 'rm', '-f', identifier], check=False, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
