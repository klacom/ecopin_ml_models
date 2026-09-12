# Experiment 02 — Historical Exp 8 Fine-Tune on Final 1,000-Image Dataset

## 1. Experiment Objective
Determine whether fine-tuning the historical best EcoPin model (Experiment 8, trained on the original ~350-image dataset) yields superior generalization on the finalized 1,000-image dataset compared to the fresh ImageNet-pretrained model from Experiment 01.

## 2. Dataset and Split
- **Total Images:** 1,000 images
- **Dataset Path:** `c:/dev/datasets/ecopin_dataset/split/`
- **Partition:** 800 train / 100 validation / 100 test (80/10/10 stratified split)
- **Class Distribution:** Exactly 250 images per class (200 train / 25 val / 25 test per class)
  - `flooding`: 250 images
  - `non_environmental`: 250 images
  - `pollution`: 250 images
  - `waste`: 250 images

## 3. Model Architecture
- **Architecture:** EfficientNet-B0 (`timm`)
- **Number of Classes:** 4 (`flooding`, `non_environmental`, `pollution`, `waste`)

## 4. Historical Checkpoint Used
- **Source Checkpoint:** `c:/dev/ecopin_ml_models/baseline_original_350/best_model.pt` (Experiment 8)
- **Original Training:** Trained on the 350-image dataset (245 train / 53 val / 52 test)
- **Historical Benchmark:** 69.81% Val Acc, 0.6743 Macro F1 on the old 53-image validation split.

## 5. Training Configuration
- **Random Seed:** 42
- **Image Size:** 224 × 224 (aspect-preserving square padding)
- **Batch Size:** 16
- **Loss Function:** `CrossEntropyLoss` (unweighted; uniform class representation)
- **Optimizer:** Adam
- **Initial Learning Rate:** 1.0e-05 (conservative fine-tuning)
- **Scheduler:** `ReduceLROnPlateau(mode='min', factor=0.5, patience=2)`
- **Early Stopping:** Patience = 5 epochs monitoring validation Macro F1
- **Augmentation:** `PadToSquare`, `Resize(224, 224)`, `RandomHorizontalFlip(p=0.2)`, `ImageNet Normalization`
- **Total Epochs Trained:** 20

## 6. Fine-Tuning Strategy
Direct end-to-end conservative fine-tuning initialized from the historical Exp 8 weights at a low learning rate (`1e-5`). Since the classification head was already trained for the 4 EcoPin classes, unfreezing the full network immediately at `1e-5` allows seamless adaptation across feature extraction and classification layers without catastrophic forgetting.

## 7. Best Epoch
- **Best Epoch:** Epoch 16 (out of 20 epochs)

## 8. Best Validation Metrics
- **Validation Accuracy:** 74.00% (0.7400)
- **Validation Macro F1:** 0.7387
- **Validation Loss:** 1.0505

## 9. Final Test Metrics (Evaluated Once on `split/test`)
- **Test Accuracy:** 62.00% (0.6200)
- **Test Macro F1:** 0.6105
- **Test Loss:** 1.3067

## 10. Per-Class Results

### Validation Set (100 samples)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **flooding** | 0.7857 | 0.8800 | 0.8302 | 25.0 |
| **non_environmental** | 0.6538 | 0.6800 | 0.6667 | 25.0 |
| **pollution** | 0.7500 | 0.7200 | 0.7347 | 25.0 |
| **waste** | 0.7727 | 0.6800 | 0.7234 | 25.0 |
| **Macro Avg** | 0.7406 | 0.7400 | **0.7387** | 100.0 |

### Test Set (100 samples)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **flooding** | 0.7619 | 0.6400 | 0.6957 | 25.0 |
| **non_environmental** | 0.5385 | 0.5600 | 0.5490 | 25.0 |
| **pollution** | 0.6250 | 0.4000 | 0.4878 | 25.0 |
| **waste** | 0.5946 | 0.8800 | 0.7097 | 25.0 |
| **Macro Avg** | 0.6300 | 0.6200 | **0.6105** | 100.0 |

## 11. Confusion Matrix

### Validation Confusion Matrix
```text
               Predicted
           FLD  NON  POL  WST
True  FLD  [22,   3,   0,   0]
      NON  [ 3,  17,   3,   2]
      POL  [ 1,   3,  18,   3]
      WST  [ 2,   3,   3,  17]
```

### Test Confusion Matrix
```text
               Predicted
           FLD  NON  POL  WST
True  FLD  [16,   7,   2,   0]
      NON  [ 3,  14,   4,   4]
      POL  [ 0,   4,  10,  11]
      WST  [ 2,   1,   0,  22]
```

## 12. Comparison with Experiment 01

| Metric (Test Set - 100 images) | Experiment 01 (Fresh Pretrained) | Experiment 02 (Historical Exp 8 Fine-Tune) | Difference (Exp 02 - Exp 01) |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 52.00% | 62.00% | +10.00% |
| **Test Macro F1** | 0.5174 | 0.6105 | +0.0931 |
| **Test Loss** | 1.6902 | 1.3067 | -0.3835 |

## 13. Comparison with Historical Exp 8
- **Historical Exp 8 (Validation Benchmark):** 69.81% Val Acc / 0.6743 Macro F1 on 53 validation images from original ~350-image dataset.
- **Experiment 02 (Validation):** 74.00% Val Acc / 0.7387 Macro F1 on 100 validation images from finalized 1,000-image dataset.
- *Context Distinction:* Historical Exp 8 had only 53 validation samples with known leakage before the 96 replacement protocol. Experiment 02 evaluates against a clean, leakage-free 100-sample validation set and 100-sample test set.

## 14. Final Conclusion
Experiment 02 demonstrated superior performance over Experiment 01 on the finalized 100-image test set, achieving **62.00% test accuracy** and **0.6105 test Macro F1**.
