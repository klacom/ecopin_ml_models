import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from review_severity_pilot import FIELDS, ReviewStore


class ReviewStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'images').mkdir()
        (self.root / 'images/one.jpg').write_bytes(b'image')
        self.output = self.root / 'reviewer1.csv'
        self.row = dict.fromkeys(FIELDS, '') | {
            'image_id': 'SEV30_001', 'image_path': 'images/one.jpg',
            'criteria_version': 'category-v2-draft'}
        self.write(self.output, self.row)
        self.write(self.root / 'reviewer2.csv', self.row)
        self.original = self.output.read_bytes()
        self.store = ReviewStore(self.root, 'reviewer1')
        self.answers = dict(reviewer_id='Reviewer A', validity='valid', issue_category='waste',
                            scale='small', obstruction_target='road', obstruction='none',
                            visible_evidence='A few items beside a visibly open road.', notes='')

    def write(self, output, row):
        with output.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader(); writer.writerow(row)

    def test_completed_answers_resume_and_preserve_other_reviewer_and_identifiers(self):
        other = (self.root / 'reviewer2.csv').read_bytes()
        self.store.save(0, self.answers, complete=True)
        saved = ReviewStore(self.root, 'reviewer1').rows[0]
        self.assertEqual(saved['review_status'], 'complete')
        self.assertEqual(saved['scale_basis'], 'waste accumulation')
        self.assertEqual(saved['image_id'], self.row['image_id'])
        self.assertTrue(saved['reviewed_at'])
        self.assertEqual((self.root / 'reviewer2.csv').read_bytes(), other)
        self.assertEqual(self.output.with_suffix('.before-app.csv').read_bytes(), self.original)

    def test_drafts_can_be_incomplete_and_completed_edits_become_drafts(self):
        self.store.save(0, self.answers, complete=True)
        self.store.save(0, dict(self.answers, visible_evidence=''))
        self.assertEqual(self.store.rows[0]['review_status'], 'draft')
        self.assertEqual(self.store.rows[0]['reviewed_at'], '')

    def test_completion_requires_every_answer_and_visible_evidence(self):
        for field in ['reviewer_id', 'validity', 'issue_category', 'scale',
                      'obstruction_target', 'obstruction', 'visible_evidence']:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.store.save(0, dict(self.answers, **{field: ''}), complete=True)
        self.assertEqual(self.output.read_bytes(), self.original)

    def test_unknown_target_does_not_allow_a_conclusive_obstruction(self):
        with self.assertRaises(ValueError):
            self.store.save(0, dict(self.answers, obstruction_target='unknown'), complete=True)
        self.store.save(0, dict(self.answers, obstruction_target='unknown', obstruction='uncertain'), complete=True)

    def test_external_changes_are_never_overwritten(self):
        self.output.write_bytes(b'external changes')
        with self.assertRaises(RuntimeError):
            self.store.save(0, self.answers)
        self.assertEqual(self.output.read_bytes(), b'external changes')

    def test_failed_replace_preserves_saved_file_and_in_memory_answers(self):
        with patch('review_severity_pilot.os.replace', side_effect=PermissionError('CSV open in Excel')):
            with self.assertRaises(PermissionError):
                self.store.save(0, self.answers)
        self.assertEqual(self.output.read_bytes(), self.original)
        self.assertEqual(self.store.rows[0]['validity'], '')

    def test_image_paths_and_reviewer_selection_are_restricted(self):
        with self.assertRaises(ValueError): ReviewStore(self.root, '../reviewer1')
        self.write(self.output, dict(self.row, image_path='../outside.jpg'))
        with self.assertRaises(ValueError): ReviewStore(self.root, 'reviewer1')


if __name__ == '__main__':
    unittest.main()
