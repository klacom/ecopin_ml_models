"""Exercise inference response logic without loading model weights or downloading CLIP."""
import ast
import io
from pathlib import Path
import time
import unittest
from unittest.mock import Mock

from PIL import Image
import torch


class SeverityInferenceTests(unittest.TestCase):
    def inference(self, filename, probability):
        # Isolate request logic from the services' model-loading startup side effects.
        tree = ast.parse(Path(__file__).with_name(filename).read_text(encoding='utf-8'))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_inference']
        embedding = torch.ones(1, 512)
        namespace = dict(
            io=io, time=time, Image=Image, torch=torch, device='cpu',
            THRESHOLD=.5, REVIEW_LO=.35, REVIEW_HI=.65,
            _log=Mock(), clip_preprocess=Mock(return_value=torch.ones(3, 224, 224)),
            clip_model=Mock(encode_image=Mock(return_value=embedding)),
            classifier=Mock(return_value=torch.logit(torch.tensor([probability]))),
            _decision=lambda p: ('VALID' if p >= .5 else 'INVALID',
                                 'MANUAL_REVIEW' if .35 <= p <= .65 else 'APPROVED' if p >= .5 else 'REJECTED',
                                 .35 <= p <= .65),
            _classify_category=Mock(return_value='waste'),
            _classify_scale=Mock(return_value=('large', .8, {'small': .1, 'medium': .1, 'large': .8})),
            _classify_obstruction=Mock(return_value=('complete', .9, {'none': .05, 'partial': .05, 'complete': .9})),
        )
        exec(compile(ast.Module(body=functions, type_ignores=[]), filename, 'exec'), namespace)
        image = io.BytesIO()
        Image.new('RGB', (8, 8)).save(image, format='PNG')
        return namespace['run_inference'](image.getvalue()), namespace

    def test_approved_and_review_images_keep_validity_and_category_without_image_severity(self):
        for filename in ['inference_service.py', 'inference_service_exp32.py']:
            for probability in [.95, .5]:
                with self.subTest(service=filename, probability=probability):
                    result, namespace = self.inference(filename, probability)
                    self.assertEqual(result['decision'], 'APPROVED' if probability == .95 else 'MANUAL_REVIEW')
                    self.assertEqual(result['predicted_category'], 'waste')
                    self.assertAlmostEqual(result['confidence'], probability)
                    for key in ['predicted_scale', 'scale_confidence', 'scale_probabilities',
                                'predicted_obstruction', 'obstruction_confidence', 'obstruction_probabilities']:
                        self.assertIsNone(result[key])
                    namespace['_classify_scale'].assert_not_called()
                    namespace['_classify_obstruction'].assert_not_called()

    def test_rejected_images_have_no_severity_predictions(self):
        for filename in ['inference_service.py', 'inference_service_exp32.py']:
            result, namespace = self.inference(filename, .1)
            self.assertIn('predicted_scale', result)
            self.assertIsNone(result['predicted_scale'])
            self.assertIsNone(result['predicted_obstruction'])
            namespace['_classify_scale'].assert_not_called()
            namespace['_classify_obstruction'].assert_not_called()

    def test_shared_estimator_can_select_every_scale_and_obstruction_level(self):
        from severity_estimator import SeverityEstimator, WASTE_SCALES, OBSTRUCTION_LEVELS
        model = Mock(encode_text=Mock(return_value=torch.eye(3)))
        estimator = SeverityEstimator(model, lambda prompts: torch.ones(3, 3), 'cpu')
        for index in range(3):
            embedding = torch.eye(3)[index:index + 1]
            for classify, labels in [(estimator.classify_scale, WASTE_SCALES),
                                     (estimator.classify_obstruction, OBSTRUCTION_LEVELS)]:
                label, confidence, probabilities = classify(embedding)
                self.assertEqual(label, labels[index])
                self.assertGreater(confidence, .99)
                self.assertAlmostEqual(sum(probabilities.values()), 1)


if __name__ == '__main__':
    unittest.main()
