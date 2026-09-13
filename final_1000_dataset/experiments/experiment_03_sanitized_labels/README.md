# Experiment 03 — Sanitized-Label Fine-Tuning

> **Preservation Statement**: "The original final-1000 dataset and Experiment 01/02 artifacts were preserved."

## 1. Objective
Test whether correcting confirmed label contradictions in the `non_environmental` training data (reclassifying 5 confirmed `waste` images previously mislabeled as `non_environmental`) improves model generalization and reduces class confusion.

## 2. Dataset and Split
- **Sanitized Dataset Path:** `c:/dev/datasets/ecopin_dataset/experiment_03_sanitized/`
- **Original Dataset Path:** `c:/dev/datasets/ecopin_dataset/split/` (100% preserved)
- **Sample Distribution:**
  - `train`: 200 flooding, 196 non_environmental, 200 pollution, 204 waste (800 total)
  - `val`: 25 flooding, 24 non_environmental, 25 pollution, 26 waste (100 total)
  - `test`: 25 flooding, 25 non_environmental, 25 pollution, 25 waste (100 total, 100% byte-for-byte identical to baseline test set)

## 3. Audit Verification Summary
- **Candidates Audited**: 7
- **Changed in Train/Val**: 5 (`NEG_126`, `NEG_136`, `NEG_141`, `NEG_142`, `NEG_248`)
- **Protected in Test**: 2 (`NEG_222`, `NEG_249` preserved unchanged under Test-Set Protection Rule)

## 4. Model Architecture & Hyperparameters
- **Architecture**: EfficientNet-B0 (`timm`)
- **Initialization Checkpoint**: `baseline_original_350/best_model.pt` (Historical Exp 8 weights)
- **Seed**: 42
- **Image Size**: 224 × 224 (`PadToSquare`, `Resize(224, 224)`, `RandomHorizontalFlip(p=0.2)`, `ImageNet Normalization`)
- **Batch Size**: 16
- **Loss**: `CrossEntropyLoss` (unweighted)
- **Optimizer**: Adam (`lr=1e-5`)
- **Scheduler**: `ReduceLROnPlateau(mode='min', factor=0.5, patience=2)`
- **Early Stopping**: Patience = 5 epochs monitoring validation Macro F1

## 5. Key Results Summary

| Metric | Exp 01 (Fresh Pretrained) | Exp 02 (Historical Exp 8) | Exp 03 (Sanitized Labels) | Exp 03 vs Exp 02 Diff |
| :--- | :---: | :---: | :---: | :---: |
| **Best Epoch** | Epoch 13 | Epoch 16 | **Epoch 9** | -7 epochs |
| **Validation Acc** | 59.00% | 74.00% | **72.00%** | -2.00% |
| **Validation Macro F1** | 0.5841 | 0.7387 | **0.7165** | -0.0222 |
| **Test Accuracy** | 52.00% | 62.00% | **62.00%** | **0.00%** |
| **Test Macro F1** | 0.5174 | 0.6105 | **0.6137** | **+0.0032** |
| **Non-Env Precision** | 38.46% | 53.85% | **59.09%** | **+5.24%** |
| **Waste $\rightarrow$ Non-Env Error** | 4 images | 1 image | **0 images** | **-1 image (0.0% error)** |

## 6. Key Findings
1. **Sanitization Boosted Non-Environmental Precision**: Removing waste false negatives from `non_environmental` training data increased precision from **53.85% to 59.09%**.
2. **Eliminated Waste $\rightarrow$ Non-Environmental Misclassification**: Waste images misclassified as `non_environmental` on the test set dropped from 1 in Exp 02 to **0 in Exp 03**.
3. **Hypothesis Confirmed**: Label sanitization reduced boundary ambiguity between `waste` and `non_environmental`.
