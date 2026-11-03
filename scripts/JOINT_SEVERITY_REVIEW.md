# Joint severity review

Offline Windows program for the 28 disagreements in the November 2026 pilot.
It displays each original photo, both reviewers' unchanged labels and evidence,
and five final questions. Discuss the case together, enter a shared reason, then
choose **Agreed & next** or **Unresolved & next**. Uncertain answers remain allowed.
Scroll the right-hand scrollbar to reach the reason field on smaller displays.

Edits autosave as drafts after 600 ms. Reopening resumes the first pending/draft
case. Previous/Next allow revisiting decisions; changing a saved decision makes
it a draft until explicitly confirmed again. A reason is required for agreed and
unresolved decisions; only agreed decisions require all five consistent labels.

The app atomically updates `joint_review.csv`, retaining original reviewer
columns and row order. It creates `joint_review.before-app.csv` on the first save,
detects external CSV edits, and prevents two Windows instances editing the same
sheet. Close Excel before editing. The two fully matching photos remain in the
separate `comparison.csv`; this app handles only the 28 disputed photos.

## Local executable

`severity_pilot_20261102_balanced/comparison_20261103/EcoPin Joint Review.exe`

Keep the executable beside `joint_review.csv`. Images and criteria are resolved
from the sibling `../review_pack` directory, so keep that directory in place.
This executable needs neither Python nor internet, but is unsigned and targets
Windows x64. Moving only the executable to another machine is insufficient.
Return the saved `joint_review.csv` after the discussion. Original independent
submission snapshots, images, model data, and production services are untouched.

## Source and checks

Run `python scripts/joint_severity_review.py --root <comparison-directory>
--images-root <review-pack-directory>` from the ML repository (one command).
Run `python -m unittest discover -s scripts -p test_joint_severity_review.py`.

Build using the existing isolated reviewer environment (Python 3.14,
PyInstaller 6.22.3, Pillow 12.3.0):

```
python -m PyInstaller --onefile --windowed --noupx --name "EcoPin Joint Review" scripts/joint_severity_review.py
```

Six storage tests cover validation, drafts/unresolved decisions, reopening,
backup/source preservation, external edit conflicts, failed atomic writes and
unsafe image paths. The frozen app's startup and layout were checked on a
disposable CSV copy. Real human consensus labels were not entered during testing.
