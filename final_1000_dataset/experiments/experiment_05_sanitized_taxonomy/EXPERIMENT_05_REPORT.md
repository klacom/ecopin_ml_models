# EXPERIMENT 05 — Pollution to Waste Taxonomy Sanitization + Retraining

## 1. Objective

Evaluate whether correcting the severe `pollution` -> `waste` taxonomy overlap identified in Experiment 04 improves EfficientNet-B0 performance while preserving the original 100-image test benchmark byte-for-byte.

## 2. Baseline Context

- Experiment 01 test accuracy / macro F1: **52.00% / 0.5174**
- Experiment 02 test accuracy / macro F1: **62.00% / 0.6105**
- Experiment 03 test accuracy / macro F1: **62.00% / 0.6137**
- Experiment 03 dominant boundary: `pollution -> waste = 10` errors and `waste -> pollution = 2` errors on the protected 100-image test set.

## 3. Source of Truth and Inputs

- Audit report: `experiment_04_pollution_waste_audit.md`
- Candidate inventory: `target_audit_list.json`
- Experiment 02 source-of-truth training script: `c:\dev\ecopin_ml_models\final_1000_dataset\experiments\experiment_02_baseline_finetune\train_experiment02.py`
- Source dataset: `c:\dev\datasets\ecopin_dataset\split`
- Sanitized dataset copy: `c:\dev\datasets\ecopin_dataset\experiment_05_sanitized_taxonomy`
- Initialization checkpoint: `c:\dev\ecopin_ml_models\baseline_original_350\best_model.pt`

## 4. Audit Outcome Summary

- Total audited candidates: **101**
- `RECLASSIFY_TO_WASTE`: **91**
- `KEEP`: **8**
- `AMBIGUOUS`: **2**

## 5. Candidate Handling Policy

- Train/val `RECLASSIFY_TO_WASTE` candidates were physically moved from `pollution` to `waste`.
- Test `RECLASSIFY_TO_WASTE` candidates were explicitly protected and left unchanged to preserve direct benchmark comparability with Experiments 01-03.
- `KEEP` and `AMBIGUOUS` candidates remained labeled as `pollution`.
- No oversampling, undersampling, class weights, or synthetic balancing were introduced.

## 6. Candidate Handling Breakdown by Split

- Reviewed candidates in `train`: **79**
- Reviewed candidates in `val`: **10**
- Reviewed candidates in `test`: **12**
- Applied relabels in `train`: **69**
- Applied relabels in `val`: **10**
- Protected relabel candidates in `test`: **12**
- Unchanged `KEEP` + `AMBIGUOUS` candidates: **10**

## 7. Applied Label Changes

- Total applied relabels: **79**
- `land_pollution` relabeled to `waste`: **43**
- `construction_debris` relabeled to `waste`: **36**

## 8. Protected Test Candidates

Protected filenames (12): POL_113.jpg, POL_121.jpg, POL_126.jpg, POL_135.jpg, POL_143.jpg, POL_145.jpg, POL_202.jpg, POL_209.jpg, POL_214.jpg, POL_222.jpg, POL_223.jpg, POL_246.jpg

These files remain labeled as `pollution` in the benchmark even though Experiment 04 marked them `RECLASSIFY_TO_WASTE`.

## 9. Dataset Construction Procedure

1. Copied `c:\dev\datasets\ecopin_dataset\split` to `c:\dev\datasets\ecopin_dataset\experiment_05_sanitized_taxonomy`.
2. Parsed Experiment 04 audit decisions for all 101 targeted `pollution` candidates.
3. Moved only approved `train` and `val` files from `pollution` to `waste`.
4. Left the `test` split untouched and verified it against the source split with SHA256 hashes.

## 10. Dataset Class Distributions

| Split | flooding | non_environmental | pollution | waste | Total |
|---|---:|---:|---:|---:|---:|
| train | 200 | 200 | 131 | 269 | 800 |
| val | 25 | 25 | 15 | 35 | 100 |
| test | 25 | 25 | 25 | 25 | 100 |

## 11. Dataset Integrity Verification

- Count check passed: **True**
- Readable image failures: **0**
- Missing files vs source by image ID: **0**
- Orphan files vs source by image ID: **0**
- Duplicate IDs in source/sanitized: **0 / 0**
- Split membership changed: **0**
- Image-content hash mismatches by image ID: **0**
- Unexpected label changes: **0**
- Missing intended relabels: **0**
- Protected test label changes: **0**
- Duplicate filenames across splits: **0**
- Duplicate hashes across splits: **0**
- Test-set hash mismatches: **0**
- Overall integrity verdict: **True**

## 12. Exp02 Methodology Reproduction

Compared against the actual Experiment 02 training script and config:

- Source of truth script: `c:\dev\ecopin_ml_models\final_1000_dataset\experiments\experiment_02_baseline_finetune\train_experiment02.py`
- Source of truth config: `c:\dev\ecopin_ml_models\final_1000_dataset\experiments\experiment_02_baseline_finetune\config.json`
- Reproduced exactly: seed `42`, EfficientNet-B0, historical Exp8 initialization checkpoint, image size `224`, batch size `16`, PadToSquare + Resize + horizontal flip + ImageNet normalization, unweighted CrossEntropyLoss, Adam, learning rate `1e-05`, ReduceLROnPlateau, early stopping patience `5` on validation Macro F1, max epochs `20`, and best-checkpoint selection by validation Macro F1.
- Intentional differences: dataset root changed to the sanitized copy, and Exp05 adds manifest/integrity/report generation around the unchanged training loop.

## 13. Training Configuration

- Seed: `42`
- Architecture: `efficientnet_b0`
- Initialization: Historical Exp 8 checkpoint
- Optimizer: `Adam`
- Learning rate: `1e-05`
- Scheduler: `ReduceLROnPlateau(mode='min', factor=0.5, patience=2)`
- Early stopping: `patience = 5` on validation Macro F1
- Max epochs: `20`
- Loss: `CrossEntropyLoss (unweighted)`

## 14. Initialization Checkpoint Validation

- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 5050 Laptop GPU`
- Historical checkpoint path loaded successfully before fine-tuning.

## 15. Pre-Fine-Tuning Metrics

- Validation accuracy / macro F1 / loss: **61.00% / 0.5908 / 1.5708**
- Test accuracy / macro F1 / loss: **59.00% / 0.5705 / 1.8647**

## 16. Training Summary

- Best epoch: **7**
- Total epochs trained: **12**
- Validation early-stopping target: **Macro F1**

## 17. Best Validation Results

- Accuracy: **71.00%**
- Macro Precision: **0.6906**
- Macro Recall: **0.7014**
- Macro F1: **0.6945**
- Weighted F1: **0.7092**
- Loss: **1.0116**

Validation per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| flooding | 0.7857 | 0.8800 | 0.8302 | 25 |
| non_environmental | 0.6400 | 0.6400 | 0.6400 | 25 |
| pollution | 0.5625 | 0.6000 | 0.5806 | 15 |
| waste | 0.7742 | 0.6857 | 0.7273 | 35 |
| macro avg | 0.6906 | 0.7014 | 0.6945 | 100 |

## 18. Final Test Results

- Accuracy: **64.00%**
- Macro Precision: **0.6385**
- Macro Recall: **0.6400**
- Macro F1: **0.6203**
- Weighted F1: **0.6203**
- Loss: **1.5433**

Test per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| flooding | 0.7273 | 0.6400 | 0.6809 | 25 |
| non_environmental | 0.6400 | 0.6400 | 0.6400 | 25 |
| pollution | 0.5714 | 0.3200 | 0.4103 | 25 |
| waste | 0.6154 | 0.9600 | 0.7500 | 25 |
| macro avg | 0.6385 | 0.6400 | 0.6203 | 100 |

## 19. Test Confusion Matrix

| True \ Pred | flooding | non_environmental | pollution | waste |
|---|---:|---:|---:|---:|
| flooding | 16 | 6 | 2 | 1 |
| non_environmental | 3 | 16 | 4 | 2 |
| pollution | 2 | 3 | 8 | 12 |
| waste | 1 | 0 | 0 | 24 |

## 20. Pollution-Waste Boundary Analysis

- Experiment 01: `pollution -> waste = 7`, `waste -> pollution = 2`
- Experiment 02: `pollution -> waste = 11`, `waste -> pollution = 0`
- Experiment 03: `pollution -> waste = 10`, `waste -> pollution = 2`
- Experiment 05: `pollution -> waste = 12`, `waste -> pollution = 0`

Protected-candidate behavior:
- Exp 01 predicted `waste` on protected test candidates: **6/12**
- Exp 02 predicted `waste` on protected test candidates: **7/12**
- Exp 03 predicted `waste` on protected test candidates: **7/12**
- Exp 05 predicted `waste` on protected test candidates: **8/12**

Class-specific test metrics:
- Pollution precision / recall / F1: **0.5714 / 0.3200 / 0.4103**
- Waste precision / recall / F1: **0.6154 / 0.9600 / 0.7500**

## 21. Comparative Results Across Experiments

| Experiment | Test Acc | Test Macro F1 | Pollution P | Pollution R | Pollution F1 | Waste P | Waste R | Waste F1 | pollution->waste | waste->pollution | Boundary Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Experiment 01 | 0.5200 | 0.5174 | 0.5882 | 0.4000 | 0.4762 | 0.5000 | 0.6400 | 0.5614 | 7 | 2 | 9 |
| Experiment 02 | 0.6200 | 0.6105 | 0.6250 | 0.4000 | 0.4878 | 0.5946 | 0.8800 | 0.7097 | 11 | 0 | 11 |
| Experiment 03 | 0.6200 | 0.6137 | 0.5238 | 0.4400 | 0.4783 | 0.6000 | 0.8400 | 0.7000 | 10 | 2 | 12 |
| Experiment 05 | 0.6400 | 0.6203 | 0.5714 | 0.3200 | 0.4103 | 0.6154 | 0.9600 | 0.7500 | 12 | 0 | 12 |

## 22. Limitations and Threats to Validity

- The primary test set intentionally preserves 12 audited label-noise candidates, so benchmark metrics still include known taxonomy mismatch.
- Experiment 05 isolates taxonomy sanitization only; it does not test class balancing, augmentation changes, or architecture changes.
- Two audited images remained `AMBIGUOUS`, so some residual boundary uncertainty is expected inside `pollution`.

## 23. Conclusion

Experiment 05 improved overall test Macro F1 relative to Experiment 03 (**0.6203 vs 0.6137**) and raised test accuracy to **64.00%**. However, the measured benchmark boundary itself did **not** improve on the protected test set: `pollution -> waste` increased from **10** in Experiment 03 to **12** in Experiment 05, while `waste -> pollution` dropped from **2** to **0**. This result is consistent with the benchmark caveat that 12 protected test images remain labeled `pollution` even though the audit concluded they belong to `waste`.

Hypothesis status: **Partially supported: overall benchmark performance improved slightly, but the protected pollution/waste boundary did not improve.**
