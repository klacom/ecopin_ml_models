# EcoPin ML — Git Cleanup Report

> **Generated:** 2026-09-12
> **Branch:** `TRAIN-retrain-image-validation-1k-data`
> **Repository root:** `c:\dev\ecopin_ml_models`

---

## 1. What the 138+ changed files actually are

GitHub Desktop was reporting a large number of changes because it **enumerates every individual file inside untracked directories**, expanding them all at once. The actual `git status` output (with the updated `.gitignore`) is:

```
M  .gitignore
M  image_validation/ecopin_data_set.xlsx
?? final_1000_dataset/
?? image_validation/.env.example
?? image_validation/ecopin_data_set_backup.xlsx
?? image_validation/ecopin_data_set_modified.xlsx
?? image_validation/requirements.txt
```

Previously (before the `.gitignore` update), `archive_original_350_dataset/` and `baseline_original_350/` were also untracked directories, each containing dozens of files — which is what drove GitHub Desktop's count to 138+.

---

## 2. Source files that SHOULD be committed

These are source or documentation files with long-term value:

| File / Directory | Category | Status |
|---|---|---|
| `.gitignore` | Project config | **Modified — commit this** |
| `MANUAL_TESTING.md` | Documentation | **New — commit this** |
| `GIT_CLEANUP_REPORT.md` | Documentation | **New — commit this** |
| `image_validation/inference_service.py` | Inference service source | Already tracked |
| `image_validation/requirements.txt` | Dependency spec | **New — commit this** |
| `image_validation/.env.example` | Env variable template | **New — commit this** |
| `image_validation/training/train_experiment8.py` | Historical training script | Already tracked |
| `image_validation/training/` (all other scripts) | Training scripts | Already tracked |
| `image_validation/data_config/` | Dataset configuration | Already tracked |
| `image_validation/dataset_downloader/` | Downloader scripts | Already tracked |
| `image_validation/artifacts/` (JSONs, CSVs, MDs) | Experiment metrics | Already tracked |
| `image_validation/my-skills/` | Reference documents | Already tracked |
| `final_1000_dataset/README.md` | New experiment overview | **Untracked — commit this** |
| `final_1000_dataset/experiments/*/train_experiment0*.py` | Final experiment scripts | **Untracked — commit these** |
| `final_1000_dataset/experiments/*/config.json` | Experiment configurations | **Untracked — commit these** |
| `final_1000_dataset/experiments/*/README.md` | Experiment documentation | **Untracked — commit these** |
| `final_1000_dataset/experiments/*/training_history.csv` | Training curves | **Untracked — commit these** |
| `final_1000_dataset/experiments/*/classification_report.json` | Per-class metrics | **Untracked — commit these** |
| `final_1000_dataset/experiments/*/confusion_matrix.json` | Confusion matrices | **Untracked — commit these** |
| `final_1000_dataset/experiments/*/per_epoch_metrics.json` | Per-epoch metrics | **Untracked — commit these** |
| `final_1000_dataset/experiments/*/val_predictions.json` | Validation predictions | **Untracked — commit these** |
| `final_1000_dataset/experiments/*/test_predictions.json` | Test predictions | **Untracked — commit these** |

---

## 3. Generated / large files that should NOT be committed

| File / Directory | Reason | Handled by |
|---|---|---|
| `archive_original_350_dataset/` | Duplicates files already tracked in `image_validation/`; purely archival | `.gitignore` — **now ignored** |
| `baseline_original_350/best_model.pt` | Large binary; duplicate of `image_validation/checkpoints/exp8/best.pt` which is already committed | `.gitignore` — **now ignored** |
| `baseline_original_350/historical_baseline_model.pt` | Another copy of the same checkpoint | `.gitignore` — **now ignored** |
| `final_1000_dataset/**/*.pt` | Model checkpoint binaries (~15–16 MB each); 4 files total across Exp01/Exp02 | `.gitignore` — **now ignored** |
| `image_validation/.env` | Secrets / local configuration | `.gitignore` (was already in `.env.*` pattern; now explicitly added) |
| `image_validation/ecopin_data_set_backup.xlsx` | Backup file, not a meaningful source artifact | Review before committing — recommend ignoring |
| `image_validation/ecopin_data_set_modified.xlsx` | Intermediate working copy | Review before committing — recommend ignoring |
| `__pycache__/` | Python bytecode | `.gitignore` — already ignored |
| `.venv/` | Virtual environment | `.gitignore` — already ignored |

---

## 4. What the new `.gitignore` patterns exclude

The following patterns were **added** to `.gitignore` by this cleanup:

```gitignore
# Entire archive directory (duplicates already-tracked files)
archive_original_350_dataset/

# Historical baseline directory (checkpoint already committed elsewhere)
baseline_original_350/

# Model checkpoint binaries in final 1,000-image experiments (~15–16 MB each)
final_1000_dataset/**/*.pt
final_1000_dataset/**/*.pth

# Inference service local environment file
image_validation/.env
```

The existing patterns for `__pycache__/`, `.venv/`, `.env.*`, and `checkpoints/*/` remain unchanged.

---

## 5. Important files that remain trackable

These files are inside `final_1000_dataset/` and are **NOT** ignored by the new patterns:

- `final_1000_dataset/README.md`
- `final_1000_dataset/experiments/experiment_01_fresh_pretrained/train_experiment01.py`
- `final_1000_dataset/experiments/experiment_01_fresh_pretrained/README.md`
- `final_1000_dataset/experiments/experiment_01_fresh_pretrained/config.json`
- `final_1000_dataset/experiments/experiment_01_fresh_pretrained/training_history.csv`
- `final_1000_dataset/experiments/experiment_01_fresh_pretrained/classification_report.json`
- `final_1000_dataset/experiments/experiment_01_fresh_pretrained/confusion_matrix.json`
- `final_1000_dataset/experiments/experiment_01_fresh_pretrained/per_epoch_metrics.json`
- `final_1000_dataset/experiments/experiment_01_fresh_pretrained/val_predictions.json`
- `final_1000_dataset/experiments/experiment_01_fresh_pretrained/test_predictions.json`
- *(and the same 10 files for `experiment_02_baseline_finetune/`)*

---

## 6. Large files that should NOT be committed

| File | Size | Recommendation |
|---|---|---|
| `final_1000_dataset/experiments/experiment_01_fresh_pretrained/best_model.pt` | ~15.6 MB | Do NOT commit — use `MODEL_CHECKPOINT_PATH` |
| `final_1000_dataset/experiments/experiment_01_fresh_pretrained/final_model.pt` | ~15.6 MB | Do NOT commit |
| `final_1000_dataset/experiments/experiment_02_baseline_finetune/best_model.pt` | ~15.6 MB | Do NOT commit — see §7 |
| `final_1000_dataset/experiments/experiment_02_baseline_finetune/final_model.pt` | ~15.6 MB | Do NOT commit |
| `baseline_original_350/best_model.pt` | ~15.6 MB | Do NOT commit (duplicate) |
| `baseline_original_350/historical_baseline_model.pt` | ~15.6 MB | Do NOT commit (duplicate) |

---

## 7. Should `best_model.pt` (Exp 02) be committed, Git-LFS tracked, or distributed separately?

### Current situation

- **Git LFS is already configured** on this repository.
- The `ecopin_data_set.xlsx` file is LFS-tracked (confirmed by `git lfs status`).
- The three existing `.pt` files in `image_validation/checkpoints/` are committed as **regular Git objects** (not LFS). Each is ~15.6 MB.
- GitHub allows regular Git objects up to 100 MB, so these are within limits — but they permanently increase the repository clone size.

### Recommendation: Distribute the Exp 02 checkpoint separately

> **Do NOT commit `best_model.pt` as a regular Git object.**
> **Do NOT add new `.pt` files without LFS.**

**Reasons:**
1. The repo already has three `.pt` files at ~47 MB total as regular objects. Adding four more would push the repo to ~110 MB of binary weight history.
2. LFS is available but not configured for `.pt` files yet. Migrating existing `.pt` files to LFS would require rewriting history (destructive and disruptive).
3. The inference service already supports `MODEL_CHECKPOINT_PATH` as an env var, making the checkpoint location fully flexible without needing Git.

**Recommended distribution strategies:**

| Strategy | How |
|---|---|
| **Shared network drive / USB** | Copy `best_model.pt` to a shared folder; teammate sets `MODEL_CHECKPOINT_PATH` |
| **Cloudinary / S3 / Supabase Storage** | Upload the checkpoint; teammates download once |
| **Google Drive link (simplest for a small team)** | Share a download link; teammate sets `MODEL_CHECKPOINT_PATH` |
| **Git LFS (future `.pt` files)** | Run `git lfs track "*.pt"` before committing any new checkpoints. Do NOT do this retroactively without team agreement. |

**Teammate setup with out-of-band checkpoint:**
```powershell
# After obtaining best_model.pt from your team:
$env:MODEL_CHECKPOINT_PATH = "C:\path\to\best_model.pt"
python inference_service.py
```

---

## 8. What to review in GitHub Desktop before committing

After the `.gitignore` update, GitHub Desktop should show a much smaller changeset. Review each item:

| GitHub Desktop entry | Action |
|---|---|
| `.gitignore` (modified) | **Stage and commit** |
| `MANUAL_TESTING.md` (new) | **Stage and commit** |
| `GIT_CLEANUP_REPORT.md` (new) | **Stage and commit** |
| `image_validation/requirements.txt` (new) | **Stage and commit** |
| `image_validation/.env.example` (new) | **Stage and commit** |
| `image_validation/ecopin_data_set.xlsx` (LFS, modified) | **Stage and commit** (it's the updated dataset) |
| `final_1000_dataset/` (untracked, partially) | **Stage selectively** — add source files only (see §2); the `.pt` files will be auto-excluded by `.gitignore` |
| `image_validation/ecopin_data_set_backup.xlsx` | **Do not stage** — this is a transient backup |
| `image_validation/ecopin_data_set_modified.xlsx` | **Do not stage** unless this is the intended final version |

---

## 9. Already-tracked generated files that require separate Git action

The following files are **already committed** to the repository and `.gitignore` cannot untrack them. They remain in Git history.

| File | Size | Issue |
|---|---|---|
| `image_validation/checkpoints/efficientnet_b0_best.pt` | ~15.6 MB | Binary checkpoint committed as regular Git object |
| `image_validation/checkpoints/efficientnet_b0_exp2_best.pt` | ~15.6 MB | Binary checkpoint committed as regular Git object |
| `image_validation/checkpoints/exp8/best.pt` | ~15.6 MB | Binary checkpoint committed as regular Git object (this is the active default model) |
| `image_validation/scraped_metadata.json` | ~506 KB | Large JSON committed as regular Git object |

> **These files are permanently part of the Git history.** They cannot be removed without rewriting history (e.g., `git filter-repo`), which is a destructive operation requiring coordination with all teammates.

**Recommendation:** Leave these as-is unless the team agrees to run a history rewrite. For any future model checkpoints, configure Git LFS first with `git lfs track "*.pt"`.

---

## 10. Recommended next steps

### Immediate (required)

- [ ] **Commit the new source files** listed in §8:
  ```powershell
  git add .gitignore MANUAL_TESTING.md GIT_CLEANUP_REPORT.md
  git add image_validation/requirements.txt image_validation/.env.example
  git add image_validation/ecopin_data_set.xlsx
  git commit -m "docs: add manual testing guide, inference requirements, and git cleanup"
  ```

- [ ] **Stage final_1000_dataset source files** (scripts, configs, reports — NOT `.pt` files):
  ```powershell
  git add final_1000_dataset/
  # Verify .pt files are excluded
  git status final_1000_dataset/
  git commit -m "feat: add final 1000-image dataset experiment scripts and results (Exp01, Exp02)"
  ```

### Recommended

- [ ] **Share `best_model.pt` (Exp 02) out-of-band** with your teammate using one of the strategies in §7.
- [ ] **Document the model distribution method** in `MANUAL_TESTING.md` §3 once decided.
- [ ] **Decide on `ecopin_data_set_backup.xlsx` and `ecopin_data_set_modified.xlsx`** — delete or ignore them if they are working copies.

### Optional (future hygiene)

- [ ] **Configure Git LFS for `.pt` files** before adding any new model checkpoints:
  ```powershell
  git lfs track "*.pt"
  git add .gitattributes
  git commit -m "chore: track .pt model checkpoints via Git LFS"
  ```
  Note: This does NOT retroactively move already-committed `.pt` files to LFS.
- [ ] **Consider `git filter-repo`** to clean the three existing committed `.pt` files from history if repo size becomes a problem (requires team coordination).

---

*Report generated automatically. Verify all paths and file sizes before committing.*
