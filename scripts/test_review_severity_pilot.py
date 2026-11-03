import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from review_severity_pilot import CHOICES, FIELDS, FIELDS_V3, ReviewStore, create_blank_sheet
from severity_reviewer_exe import run as run_executable


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


class V3ReviewStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'images').mkdir()
        (self.root / 'images/one.jpg').write_bytes(b'image')
        self.output = self.root / 'reviewer1.csv'
        self.row = dict.fromkeys(FIELDS_V3, '') | {
            'image_id': 'CAL_001', 'image_path': 'images/one.jpg', 'criteria_version': 'category-v3-draft'}
        self.write(self.row)
        self.store = ReviewStore(self.root, 'reviewer1')
        self.answers = dict(reviewer_id='Reviewer A', validity='valid', issue_category='waste',
                            setting='street', scale='small', extends_beyond_frame='no',
                            obstruction_target='road', obstruction='none',
                            visible_evidence='A few items beside a visibly open road.', notes='')

    def write(self, *rows):
        with self.output.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS_V3)
            writer.writeheader(); writer.writerows(rows)

    def test_v3_sheet_saves_new_fields_and_keeps_its_version(self):
        self.store.save(0, self.answers, complete=True)
        saved = ReviewStore(self.root, 'reviewer1').rows[0]
        self.assertEqual(saved['criteria_version'], 'category-v3-draft')
        self.assertEqual((saved['setting'], saved['extends_beyond_frame']), ('street', 'no'))
        self.assertEqual(saved['scale_basis'], 'waste accumulation')

    def test_category_specific_questions_are_required_only_for_their_category(self):
        pollution = dict(self.answers, issue_category='pollution', depth_reference='knee')
        with self.assertRaises(ValueError):
            self.store.save(0, pollution, complete=True)
        self.store.save(0, dict(pollution, pollution_subtype='burning_smoke'), complete=True)
        self.assertEqual(self.store.rows[0]['depth_reference'], '')
        flooding = dict(self.answers, issue_category='flooding', pollution_subtype='burning_smoke')
        with self.assertRaises(ValueError):
            self.store.save(0, flooding, complete=True)
        self.store.save(0, dict(flooding, depth_reference='ankle'), complete=True)
        self.assertEqual(self.store.rows[0]['pollution_subtype'], '')

    def test_every_v3_question_is_required_for_completion(self):
        for field in ['setting', 'extends_beyond_frame']:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.store.save(0, dict(self.answers, **{field: ''}), complete=True)

    def test_uncertain_category_requires_uncertain_extent(self):
        with self.assertRaises(ValueError):
            self.store.save(0, dict(self.answers, issue_category='uncertain'), complete=True)
        self.store.save(0, dict(self.answers, issue_category='uncertain', scale='uncertain'), complete=True)

    def test_v3_uses_drain_or_waterway_and_v2_keeps_drain(self):
        with self.assertRaises(ValueError):
            self.store.save(0, dict(self.answers, obstruction_target='drain'))
        self.store.save(0, dict(self.answers, obstruction_target='drain_or_waterway'), complete=True)
        self.assertEqual([v for _, v in CHOICES['obstruction_target']][2], 'drain')

    def test_blank_v3_sheet_lists_photos_and_never_overwrites(self):
        (self.root / 'images/two.png').write_bytes(b'image')
        (self.root / 'images/readme.txt').write_text('not a photo')
        create_blank_sheet(self.root, 'reviewer2')
        rows = ReviewStore(self.root, 'reviewer2').rows
        self.assertEqual([r['image_id'] for r in rows], ['one', 'two'])
        self.assertEqual({r['criteria_version'] for r in rows}, {'category-v3-draft'})
        with self.assertRaises(FileExistsError): create_blank_sheet(self.root, 'reviewer2')

    def test_rows_must_match_the_sheet_version(self):
        self.write(self.row, dict(self.row, image_id='CAL_002', criteria_version='category-v2-draft'))
        with self.assertRaises(ValueError): ReviewStore(self.root, 'reviewer1')


class ExecutableEntryTests(unittest.TestCase):
    def test_frozen_app_uses_its_exe_folder_and_reviewer2_without_arguments(self):
        executable = str(Path('Buddy review folder/EcoPin Review.exe').resolve())
        with patch('severity_reviewer_exe.sys.executable', executable), \
             patch('severity_reviewer_exe.sys.frozen', True, create=True), \
             patch('severity_reviewer_exe.sys.argv', [executable]), \
             patch('severity_reviewer_exe.run_review') as launch:
            run_executable()
        launch.assert_called_once_with(['--root', str(Path(executable).parent), '--reviewer', 'reviewer2'])


if __name__ == '__main__':
    unittest.main()
