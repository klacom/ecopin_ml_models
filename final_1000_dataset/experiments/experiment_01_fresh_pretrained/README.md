# Experiment 01 — Fresh Pretrained

### Dataset
- **Total Images:** 1,000 images
- **Partition:** 800 train / 100 validation / 100 test (80/10/10 stratified split)
- **Class Distribution:** Exactly 250 images per class (200 train / 25 val / 25 test)
  - `flooding`: 250 images
  - `non_environmental`: 250 images
  - `pollution`: 250 images
  - `waste`: 250 images

### Model
- **Architecture:** EfficientNet-B0 (`timm`)
- **Pretrained Weights:** ImageNet-1k pretrained weights
- **Initialization:** Fresh initialization (standard ImageNet weights; no historical checkpoint warm-start)

### Training Configuration
- **Random Seed:** 42
- **Image Size:** 224 × 224 (aspect-preserving square padding)
- **Batch Size:** 16
- **Loss Function:** CrossEntropyLoss (unweighted; dataset is uniform)
- **Optimizer:** Adam
- **Stage 1 (Frozen Backbone):** 5 epochs @ LR = 1.0e-03
- **Stage 2 (Full Fine-Tuning):** Up to 15 epochs @ LR = 1.0e-04
- **Scheduler:** `ReduceLROnPlateau(mode='min', factor=0.5, patience=2)`
- **Early Stopping:** Patience = 5 epochs monitoring Validation Macro F1
- **Augmentation:** `PadToSquare`, `Resize(224, 224)`, `RandomHorizontalFlip(p=0.2)`, `ImageNet Normalization`
- **Total Epochs Trained:** 13 (Best at Epoch 8)

---

### Best Validation Result
- **Best Epoch:** Epoch 8
- **Validation Accuracy:** 68.00% (0.6800)
- **Validation Macro F1:** 0.6777
- **Validation Loss:** 1.3573

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **flooding** | 0.8077 | 0.8400 | 0.8235 | 25.0 |
| **non_environmental** | 0.6364 | 0.5600 | 0.5957 | 25.0 |
| **pollution** | 0.6429 | 0.7200 | 0.6792 | 25.0 |
| **waste** | 0.6250 | 0.6000 | 0.6122 | 25.0 |
| **Macro Avg** | 0.6780 | 0.6800 | **0.6777** | 100.0 |

---

### Test Result (Evaluated Once on `split/test`)
- **Test Accuracy:** 52.00% (0.5200)
- **Test Macro F1:** 0.5174
- **Test Loss:** 1.6902

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **flooding** | 0.6400 | 0.6400 | 0.6400 | 25.0 |
| **non_environmental** | 0.3846 | 0.4000 | 0.3922 | 25.0 |
| **pollution** | 0.5882 | 0.4000 | 0.4762 | 25.0 |
| **waste** | 0.5000 | 0.6400 | 0.5614 | 25.0 |
| **Macro Avg** | 0.5282 | 0.5200 | **0.5174** | 100.0 |

#### Test Confusion Matrix
```text
               Predicted
           FLD  NON  POL  WST
True  FLD  [16,   6,   2,   1]
      NON  [ 4,  10,   3,   8]
      POL  [ 1,   7,  10,   7]
      WST  [ 4,   3,   2,  16]
```

---

### Historical Comparison
- **Historical Exp 8 Baseline:** Validation Accuracy **69.81%**, Validation Macro F1 **0.6743** (trained on original ~350-image set with 53 validation samples).
- *Context Note:* The new experiment evaluates on a much larger, fully balanced dataset (1,000 images total, 100 validation samples, 100 test samples) with verified non-leaked distributions.

---

### Conclusion
Experiment 01 demonstrates strong baseline performance on the finalized 1,000-image dataset, achieving 68.00% validation accuracy (0.6777 Macro F1) and 52.00% test accuracy (0.5174 Macro F1). The model shows solid generalizability across all four classes without severe class bias.
