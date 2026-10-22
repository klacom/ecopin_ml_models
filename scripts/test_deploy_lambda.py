import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('deploy_lambda', Path(__file__).with_name('deploy_lambda.py'))
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class DeploymentTests(unittest.TestCase):
    def execute(self, fail=False):
        calls = []

        def aws(*arguments):
            calls.append(arguments)
            if arguments[1] == 'get-function':
                return {'Configuration': {'RevisionId': 'original'}, 'Code': {'ResolvedImageUri': 'old-image'}}
            if arguments[1] == 'get-function-configuration':
                return {'RevisionId': 'current'}
            if arguments[1] == 'invoke':
                Path(arguments[-1]).write_text(json.dumps({'statusCode': 500 if fail else 200,
                    'body': json.dumps({'experiment': '3.2'})}))
            return {}

        with patch.object(deploy, 'aws', side_effect=aws), patch.object(sys, 'argv', ['deploy', 'image-validation', 'new-image']):
            if fail:
                with self.assertRaises(RuntimeError):
                    deploy.main()
            else:
                deploy.main()
        return calls

    def test_success_updates_once_and_checks_inference_service_version(self):
        calls = self.execute()
        updates = [call for call in calls if call[1] == 'update-function-code']
        self.assertEqual(len(updates), 1)
        self.assertIn('original', updates[0])

    def test_failed_smoke_restores_previous_image_with_current_revision(self):
        calls = self.execute(fail=True)
        updates = [call for call in calls if call[1] == 'update-function-code']
        self.assertEqual(len(updates), 2)
        self.assertIn('old-image', updates[1])
        self.assertIn('current', updates[1])


if __name__ == '__main__':
    unittest.main()
