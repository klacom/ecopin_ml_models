import csv
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from joint_severity_review import JointStore, FIELDS


class JointStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'images').mkdir()
        (self.root / 'images/photo.jpg').write_bytes(b'image')
        self.path = self.root / 'joint_review.csv'
        self.row = dict(image_id='TEST_001', image_path='images/photo.jpg',
                        decision_status='pending_joint_review', decision_reason='', disagreement_fields='scale')
        for f in FIELDS:
            self.row['final_' + f] = ''
            self.row['reviewer1_' + f] = self.row['reviewer2_' + f] = 'original'
        with self.path.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.row))
            writer.writeheader(); writer.writerow(self.row)
        self.original = self.path.read_bytes()
        self.store = JointStore(self.root, self.root)
        self.answers = dict(validity='valid', issue_category='waste', scale='medium',
                            obstruction_target='road', obstruction='partial')

    def test_agreed_round_trip_preserves_source_and_backup(self):
        self.store.save(0, self.answers, 'Visible road remains partly open.', 'agreed')
        reopened = JointStore(self.root, self.root)
        self.assertEqual(reopened.rows[0]['final_scale'], 'medium')
        self.assertEqual(reopened.rows[0]['decision_status'], 'agreed')
        self.assertEqual(reopened.rows[0]['reviewer1_scale'], 'original')
        self.assertEqual((self.root / 'joint_review.before-app.csv').read_bytes(), self.original)

    def test_requirements_and_consistency(self):
        for answers, reason in [({}, 'why'), (self.answers, ''),
                                (self.answers | {'obstruction_target': 'unknown'}, 'why'),
                                (self.answers | {'validity': 'invalid'}, 'why'),
                                (self.answers | {'scale': 'huge'}, 'why')]:
            with self.assertRaises(ValueError):
                self.store.save(0, answers, reason, 'agreed')
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_partial_draft_and_unresolved(self):
        self.store.save(0, {'scale': 'small'}, '', 'draft')
        self.assertEqual(self.store.rows[0]['decision_status'], 'draft')
        self.store.save(0, {}, 'Access target is not identifiable; no shared conclusion.', 'unresolved')
        self.assertEqual(JointStore(self.root, self.root).rows[0]['decision_status'], 'unresolved')

    def test_external_change_is_not_overwritten(self):
        self.path.write_bytes(self.original + b'\n')
        with self.assertRaises(RuntimeError):
            self.store.save(0, self.answers, 'why', 'agreed')

    def test_failed_replace_keeps_answers_and_cleans_temp(self):
        with patch('joint_severity_review.os.replace', side_effect=PermissionError('Excel lock')):
            with self.assertRaises(PermissionError):
                self.store.save(0, self.answers, 'why', 'agreed')
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(self.store.rows[0]['decision_status'], 'pending_joint_review')
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_reject_unsafe_image(self):
        self.path.write_text(self.path.read_text(encoding='utf-8-sig').replace('images/photo.jpg', '../outside.jpg'), encoding='utf-8-sig')
        with self.assertRaises(ValueError):
            JointStore(self.root, self.root)


if __name__ == '__main__':
    unittest.main()
