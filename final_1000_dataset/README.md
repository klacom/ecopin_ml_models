# EcoPin ML Experiments — Final 1,000-Image Dataset

## Dataset Specifications
- **Original Dataset:** ~350 images (Historical Experiments 1–16)
- **Final Dataset:** Exactly 1,000 images
- **Class Distribution:** Perfectly balanced across 4 classes (250 images / 25.0% each):
  - `flooding`: 250 images
  - `waste`: 250 images
  - `pollution`: 250 images
  - `non_environmental`: 250 images
- **Stratified Split (80/10/10):**
  - **Train:** 800 images (200 per class)
  - **Validation:** 100 images (25 per class)
  - **Test:** 100 images (25 per class)
- **Dataset Path:** `c:/dev/datasets/ecopin_dataset/split/`
- **Leakage Replacements:** 96 replacement records verified and downloaded below marker.
- **Group Leakage Constraints (Preserved within single splits):**
  - Group 1 (`FLD_074.jpg`, `FLD_098.jpg`) -> `val`
  - Group 2 (`FLD_083.jpg`, `FLD_087.jpg`) -> `train`
  - Group 3 (`NEG_178.jpg`, `NEG_179.jpg`) -> `val`
  - Group 4 (`POL_154.jpg`, `POL_180.jpg`) -> `train`
  - Group 5 (`POL_217.jpg`, `POL_227.jpg`) -> `train`

---

## Planned Experiments
### Experiment 01 (`experiment_01_fresh_pretrained/`)
- **Objective:** Train EfficientNet-B0 from scratch using default ImageNet-1k pretrained weights on the finalized 1,000-image dataset (800 train / 100 val / 100 test).
- **Status:** NOT YET TRAINED

### Experiment 02 (`experiment_02_baseline_finetune/`)
- **Objective:** Fine-tune EfficientNet-B0 initialized from the historical best baseline model (`baseline_original_350/best_model.pt`, trained on the original 350-image set) on the finalized 1,000-image dataset.
- **Status:** NOT YET TRAINED

---

## Historical Baseline Reference
- **Archived Experiments:** `c:/dev/ecopin_ml_models/archive_original_350_dataset/`
- **Preserved Baseline Model:** `c:/dev/ecopin_ml_models/baseline_original_350/` (Experiment 8: 69.81% Val Acc / 0.6743 Macro F1)
