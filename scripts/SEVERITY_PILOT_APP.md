# Category-v2 pilot reviewer app

`review_severity_pilot.py` is a local Tkinter app for the 14-column pilot answer sheets. It is separate from the older nine-column validity labeler. Python 3, Tkinter and Pillow are required; no network or database service is used.

```powershell
python scripts/review_severity_pilot.py --root C:/dev/ecopin/severity_pilot_20261102_balanced/review_pack --reviewer reviewer1
python -m unittest discover -s scripts -p test_review_severity_pilot.py
```

The packaged Windows launchers select the appropriate reviewer. Both reviewers see the same images and blank choices, with no source-folder labels or model predictions. The app autosaves drafts, validates explicit completion, resumes at the first incomplete photo, and derives scale_basis from the chosen issue_category. Unknown access targets require uncertain obstruction; invalid/uncertain validity requires uncertain extent and obstruction.

The original CSV is backed up before the first save. Updates use an atomic replace; external edits and locked files fail without discarding the previous data. Windows holds a separate per-reviewer lock for the life of the app. A changed completed row becomes a draft until explicitly completed again. GUI smoke testing uses separate copied fixtures, never production reviewer answers.

This collects qualitative judgments for rubric calibration. It does not score model accuracy, assign numeric incident severity, train models, or alter the source images.
