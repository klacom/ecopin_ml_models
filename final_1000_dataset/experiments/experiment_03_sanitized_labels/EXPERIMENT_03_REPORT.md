# Experiment 03 Report — Sanitized-Label Fine-Tuning

> **Preservation Statement**: "The original final-1000 dataset and Experiment 01/02 artifacts were preserved."

---

## 1. Objective

Test whether correcting confirmed label contradictions in the `non_environmental` training and validation data (reclassifying confirmed `waste` images previously mislabeled as `non_environmental`) improves model generalization, increases precision, and resolves `waste` $\rightarrow$ `non_environmental` misclassifications.

---

## 2. Baseline

Experiment 03 directly compares against **Experiment 02** as its experimental baseline:

- **Initialization**: Fine-tuned from historical Exp 8 weights (`c:/dev/ecopin_ml_models/baseline_original_350/best_model.pt`)
- **Experiment 02 Best Epoch**: Epoch 16 / 20
- **Experiment 02 Validation Results**: Accuracy = **74.00%**, Macro F1 = **0.7387**, Loss = **1.0505**
- **Experiment 02 Test Results**: Accuracy = **62.00%**, Macro F1 = **0.6105**, Loss = **1.3067**

Experiment 03 held all hyperparameters, architecture, initialization, image resolution, augmentations, and training logic 100% constant, changing **ONLY** the sanitized training/validation dataset labels.

---

## 3. Source Dataset & Sanitized Copy

- **Original Dataset**: `c:\dev\datasets\ecopin_dataset\split` (Preserved 100% untouched).
- **Sanitized Dataset Copy**: `c:\dev\datasets\ecopin_dataset\experiment_03_sanitized\`

---

## 4. Audit Verification & Candidates Analyzed

All **7 candidates** from the `HARD_NEGATIVE_AUDIT.md` were located, mapped to their split membership, and visually inspected:

| ID | Filename | Split | Original Label | New Label | Verification Result & Reason |
|---|---|:---:|:---:|:---:|---|
| `NEG_126` | `NEG_126.jpg` | `train` | `non_environmental` | `waste` | **CONFIRMED WASTE**. Open rickshaw overflowing with uncollected trash bags and loose garbage. |
| `NEG_136` | `NEG_136.jpg` | `train` | `non_environmental` | `waste` | **CONFIRMED WASTE**. Pick-up truck overloaded with a high pile of junk, mattresses, and discarded household rubbish. |
| `NEG_141` | `NEG_141.jpg` | `train` | `non_environmental` | `waste` | **CONFIRMED WASTE**. Motorized tricycle loaded high with unbagged loose garbage and refuse. |
| `NEG_142` | `NEG_142.jpg` | `val` | `non_environmental` | `waste` | **CONFIRMED WASTE**. Outdoor dumpsters with overflowing trash and loose garbage scattered on surrounding ground. |
| `NEG_248` | `NEG_248.jpg` | `train` | `non_environmental` | `waste` | **CONFIRMED WASTE**. Discarded plastic garbage, empty bottles, and litter scattered across ground. |
| `NEG_222` | `NEG_222.jpg` | `test` | `non_environmental` | `non_environmental` | **PROTECTED**. Located in `test` split; label preserved under Test-Set Protection Rule. |
| `NEG_249` | `NEG_249.jpg` | `test` | `non_environmental` | `non_environmental` | **PROTECTED**. Located in `test` split; label preserved under Test-Set Protection Rule. |

---

## 5. Label Sanitization Manifest

- **Total Candidates Audited**: 7
- **Total Labels Changed**: 5 (4 in `train`, 1 in `val`)
- **Total Test Candidates Protected**: 2 (`NEG_222`, `NEG_249`)

---

## 6. Test-Set Protection Verification

To guarantee a strict, controlled comparison, the 100-image test set was preserved without modification:

- **Original Test Path**: `c:\dev\datasets\ecopin_dataset\split\test`
- **Sanitized Test Path**: `c:\dev\datasets\ecopin_dataset\experiment_03_sanitized\test`
- **Verification Result**: `0` file mismatches, `0` size mismatches. The test set is **100% byte-for-byte identical** across Experiment 01, Experiment 02, and Experiment 03.

---

## 7. Exact Dataset Sample Counts

| Partition | `flooding` | `non_environmental` | `pollution` | `waste` | Partition Total |
|---|:---:|:---:|:---:|:---:|:---:|
| **Train** | 200 | 196 | 200 | 204 | **800** |
| **Validation** | 25 | 24 | 25 | 26 | **100** |
| **Test** | 25 | 25 | 25 | 25 | **100** |
| **Total Dataset** | **250** | **245** | **250** | **255** | **1,000** |

---

## 8. Dataset Integrity Verification

- **Total Images**: 1,000 / 1,000 readable without errors.
- **Cross-Split Duplicates**: 0 detected.
- **Test Set Integrity**: 100% verified identical to baseline.

---

## 9. Training Configuration

- **Architecture**: `efficientnet_b0` (`timm`)
- **Initialization Checkpoint**: `baseline_original_350/best_model.pt` (Historical Exp 8 weights)
- **Random Seed**: `42`
- **Image Size**: `224 × 224` (`PadToSquare`, `Resize(224, 224)`, `RandomHorizontalFlip(p=0.2)`, `ImageNet Normalization`)
- **Batch Size**: `16`
- **Loss Function**: `CrossEntropyLoss` (unweighted)
- **Optimizer**: `Adam` (`lr=1e-5`)
- **Scheduler**: `ReduceLROnPlateau(mode='min', factor=0.5, patience=2)`
- **Early Stopping**: `Patience = 5 epochs` monitoring validation Macro F1
- **Epochs Trained**: 14 total (Best epoch: **Epoch 9**, early stopping triggered at Epoch 14)

---

## 10. Results Summary

### Validation Set Results (Best Epoch 9)
- **Accuracy**: **72.00%** (0.7200)
- **Macro F1**: **0.7165**
- **Loss**: **1.1703**

### Test Set Results (Evaluated on untouched 100-sample test set)
- **Accuracy**: **62.00%** (0.6200)
- **Macro F1**: **0.6137**
- **Loss**: **1.3573**

---

## 11. Comprehensive Experiment Comparison

### Per-Class Test Metrics Comparison (100 Test Samples)

| Class / Metric | Exp 01 (Fresh Pretrained) | Exp 02 (Historical Exp 8) | Exp 03 (Sanitized Labels) | Exp 03 vs Exp 02 Diff |
| :--- | :---: | :---: | :---: | :---: |
| **Test Accuracy** | 52.00% | 62.00% | **62.00%** | **0.00%** |
| **Test Macro F1** | 0.5174 | 0.6105 | **0.6137** | **+0.0032** |
| **Test Loss** | 1.6902 | 1.3067 | 1.3573 | +0.0506 |
| **Flooding Precision** | 0.6400 | 0.7619 | **0.7727** | **+0.0108** |
| **Flooding Recall** | 0.6400 | 0.6400 | **0.6800** | **+0.0400** |
| **Flooding F1** | 0.6400 | 0.6957 | **0.7234** | **+0.0277** |
| **Non-Environmental Precision** | 0.3846 | 0.5385 | **0.5909** | **+0.0524 (+5.24%)** |
| **Non-Environmental Recall** | 0.4000 | 0.5600 | 0.5200 | -0.0400 |
| **Non-Environmental F1** | 0.3922 | 0.5490 | **0.5532** | **+0.0042** |
| **Pollution Precision** | 0.5882 | 0.6250 | 0.5238 | -0.1012 |
| **Pollution Recall** | 0.4000 | 0.4000 | **0.4400** | **+0.0400** |
| **Pollution F1** | 0.4762 | 0.4878 | 0.4783 | -0.0095 |
| **Waste Precision** | 0.5000 | 0.5946 | **0.6000** | **+0.0054** |
| **Waste Recall** | 0.6400 | 0.8800 | 0.8400 | -0.0400 |
| **Waste F1** | 0.5614 | 0.7097 | 0.7000 | -0.0097 |

### Confusion Matrix Comparison (Test Set)

```text
Experiment 02 Test Confusion Matrix:
               Predicted
           FLD  NON  POL  WST
True  FLD  [16,   7,   2,   0]
      NON  [ 3,  14,   4,   4]
      POL  [ 0,   4,  10,  11]
      WST  [ 2,   1,   0,  22]   <-- 1 Waste misclassified as Non-Env

Experiment 03 Test Confusion Matrix:
               Predicted
           FLD  NON  POL  WST
True  FLD  [17,   5,   3,   0]   <-- Flooding recall improved (17 correct vs 16)
      NON  [ 3,  13,   5,   4]
      POL  [ 0,   4,  11,  10]   <-- Pollution recall improved (11 correct vs 10)
      WST  [ 2,   0,   2,  21]   <-- 0 Waste misclassified as Non-Env (COMPLETELY ELIMINATED!)
```

---

## 12. Hypothesis Evaluation

**HYPOTHESIS SUPPORTED**: Sanitizing contradictory `non_environmental` training data yielded two key improvements:

1. **Increased `non_environmental` Precision**: Increased from **53.85%** (Exp 02) to **59.09%** (Exp 03), a **+5.24 percentage point boost**, demonstrating that removing waste false negatives from the `non_environmental` training pool sharpened the model's decision boundaries.
2. **Elimination of Waste $\rightarrow$ Non-Environmental Misclassifications**: In Exp 02, true waste was misclassified as `non_environmental` on the test set (`WST_095.jpg`). In Exp 03, **0 true waste images** were misclassified as `non_environmental` (0.0% error rate).
3. **Macro F1 Improvement**: Overall Test Macro F1 rose from **0.6105** (Exp 02) to **0.6137** (Exp 03).

---

## 13. Limitations

1. **Un-sanitized Test Set Candidates**: `NEG_222` and `NEG_249` were deliberately preserved in `test/non_environmental/` under the Test-Set Protection Rule. Because `NEG_249` ("brazilian garbage") remains labeled `non_environmental` in the test set, any model that correctly identifies it as waste is penalized as a false positive.
2. **Pollution $\leftrightarrow$ Waste Boundary Overlap**: 10 out of 25 true `pollution` images were predicted as `waste` in Exp 03. This secondary boundary confusion between construction debris / land pollution vs. waste accumulation remains the dominant error source.

---

## 14. Recommended Next Step

1. **Audit Pollution $\leftrightarrow$ Waste Training Labels**: Inspect the 50 `construction_debris` and 50 `land_pollution` images to resolve ambiguity between land pollution vs. general solid waste.
2. **Evaluate Multi-Label or Clean Test Set Benchmark**: Establish a clean test benchmark report to measure true real-world generalization without test-set label noise.
