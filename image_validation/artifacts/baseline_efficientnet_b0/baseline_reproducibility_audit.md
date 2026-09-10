# EfficientNet-B0 Baseline — Reproducibility Audit

**Experiment name:** baseline_efficientnet_b0  
**Completed by task:** task-398  
**Audit created:** 2026-08-25  
**Purpose:** Record the exact configuration and actual execution behavior of the completed baseline so future experiments can be compared against a verified reference. This document must not be modified to retroactively correct historical behavior.

---

## 1. Dataset and Splits

| Item | Value |
|------|-------|
| Dataset | Ecopin Pilot v1.0 |
| Train CSV | `c:/dev/datasets/data/splits/train.csv` |
| Val CSV | `c:/dev/datasets/data/splits/val.csv` |
| Test CSV | `c:/dev/datasets/data/splits/test.csv` |
| Test set status | **Untouched — not loaded, not evaluated** |

### Train split (245 images)

| Class | Count |
|-------|-------|
| flooding | 70 |
| non_environmental | 70 |
| pollution | 35 |
| waste | 70 |
| **Total** | **245** |

### Validation split (53 images)

| Class | Count |
|-------|-------|
| flooding | 15 |
| non_environmental | 15 |
| pollution | 8 |
| waste | 15 |
| **Total** | **53** |

### Image path resolution

Images are stored under two roots resolved at runtime from `dataset.py`:
- `raw/...` -> `c:/dev/datasets/ecopin_dataset/raw/`
- `processed/...` -> `c:/dev/datasets/data/processed/`

---

## 2. Random Seed

| Item | Value |
|------|-------|
| Seed | `42` |
| Applied via | `torch.manual_seed(42)`, `np.random.seed(42)` |
| DataLoader shuffle | `True` for training, `False` for validation |

> **Note:** CPU-only training is fully deterministic with this seed. Moving to GPU may introduce minor non-determinism from CUDA kernel scheduling.

---

## 3. Model Architecture

| Item | Value |
|------|-------|
| Architecture | EfficientNet-B0 |
| Loaded via | `timm.create_model('efficientnet_b0', pretrained=True, num_classes=4)` |
| Pretrained weights | ImageNet-1k (downloaded from HuggingFace Hub via timm) |
| Output classes | 4 (flooding, non_environmental, pollution, waste) |

### Class index mapping

| Index | Class |
|-------|-------|
| 0 | flooding |
| 1 | non_environmental |
| 2 | pollution |
| 3 | waste |

---

## 4. Preprocessing and Augmentation

### Preprocessing (train and validation — identical pipeline)

1. **Pad to square** (`_pad_to_square`): zero-pad shorter dimension symmetrically
2. **Resize** to 224x224 (`T.Resize((224, 224))`)
3. **ToTensor** — scales pixels to [0, 1]
4. **Normalize** — ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]

### Training augmentation

| Augmentation | Parameters |
|-------------|------------|
| Random horizontal flip | p=0.2 |

### Validation augmentation
None — deterministic preprocessing only.

---

## 5. Loss Function and Class Weights

| Item | Value |
|------|-------|
| Loss | `torch.nn.CrossEntropyLoss(weight=class_weights)` |
| Weight formula | `total / (num_classes x class_count)` per class |
| Computed from | Train CSV only |

### Exact class weights

| Class | Train count | Weight |
|-------|------------|--------|
| flooding | 70 | 0.8750 |
| non_environmental | 70 | 0.8750 |
| pollution | 35 | 1.7500 |
| waste | 70 | 0.8750 |

---

## 6. Optimizer and Hyperparameters

| Item | Value |
|------|-------|
| Batch size | 16 |
| Max epochs | 20 |
| Optimizer | Adam |
| Stage 1 LR | 1e-3 |
| Stage 2 LR | 1e-4 |
| Weight decay | None |
| Dropout | EfficientNet-B0 timm default (not overridden) |
| Scheduler | ReduceLROnPlateau(mode=min, factor=0.5, patience=2) |
| Scheduler reset | New scheduler created at Stage 2 transition |
| Early stopping patience | 5 |
| Early stopping criterion | Minimum validation loss |

---

## 7. Stage 1 / Stage 2 Behavior

### Intended

Freeze backbone for 5 complete epochs; unfreeze at epoch 6.

### Actual (from task-398 execution log)

`
Epoch 04 | train loss 2.3892 | val loss 2.6299 | train acc 0.3388 | val acc 0.3585
--- Transition to Stage 2: unfreeze backbone, LR set to 0.0001
Epoch 05 | train loss 2.2064 | val loss 2.5329 | train acc 0.3633 | val acc 0.3774
`

The transition message printed between epoch 04 and epoch 05, proving:
- **Epochs 1-4:** backbone frozen (Stage 1)
- **Epoch 5 onward:** backbone unfrozen (Stage 2)

> **Caveat:** The exact value of `STAGE1_EPOCHS` used in task-398 cannot be verified from the log alone. The log proves when the backbone was unfrozen; it does not prove what the variable was set to. The current `train_baseline.py` has `STAGE1_EPOCHS = 5` as the corrected value for future experiments.

---

## 8. Device and Environment

| Item | Value |
|------|-------|
| Device used | **CPU** |
| Reason | PyTorch 2.13.0+cpu installed (CPU-only wheel) |
| Python | 3.14.6 CPython MSC v.1944 64-bit |
| Python executable | `C:\Users\murasakino\AppData\Local\Programs\Python\Python314\python.exe` |
| PyTorch version | `2.13.0+cpu` |
| CUDA available | `False` |
| OS | Windows 11 |
| GPU present (unused) | NVIDIA GeForce RTX 5050 Laptop GPU |

---

## 9. Training History

Full per-epoch history in `training_history.csv`.

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Notes |
|-------|-----------|---------|-----------|---------|-------|
| 1 | 3.0914 | 3.1085 | 0.2653 | 0.2830 | Stage 1 (frozen) |
| 2 | 2.7584 | 2.8713 | 0.2898 | 0.3396 | Stage 1 (frozen) |
| 3 | 2.3760 | 2.7412 | 0.3306 | 0.3396 | Stage 1 (frozen) |
| 4 | 2.3892 | 2.6299 | 0.3388 | 0.3585 | Stage 1 — last frozen epoch |
| 5 | 2.2064 | 2.5329 | 0.3633 | 0.3774 | Stage 2 begins |
| 6 | 1.5952 | 1.9107 | 0.5265 | 0.4906 | Stage 2 |
| 7 | 0.4487 | 1.7232 | 0.8163 | 0.5283 | Stage 2 |
| 8 | 0.1420 | 1.7629 | 0.9469 | 0.5849 | ES counter 1/5 |
| 9 | 0.1174 | 1.7721 | 0.9592 | 0.5660 | ES counter 2/5 |
| 10 | 0.0870 | 1.7228 | 0.9714 | 0.6226 | **Best checkpoint** — ES counter reset |
| 11 | 0.0728 | 1.7996 | 0.9714 | 0.5849 | ES counter 1/5 |
| 12 | 0.0697 | 1.9185 | 0.9755 | 0.6038 | ES counter 2/5 |
| 13 | 0.0742 | 1.8627 | 0.9755 | 0.5849 | ES counter 3/5 |
| 14 | 0.0366 | 1.9629 | 0.9918 | 0.6415 | ES counter 4/5 |
| 15 | 0.0473 | 1.8607 | 0.9918 | 0.6038 | ES counter 5/5 — early stop |

---

## 10. Best Checkpoint

| Item | Value |
|------|-------|
| Path | `c:/dev/ecopin_image_validation/checkpoints/efficientnet_b0_best.pt` |
| Saved at epoch | 10 |
| Best val loss | 1.7228 |
| Content | Model state_dict only (no optimizer state) |

---

## 11. Final Validation Metrics

Metrics re-derived by reloading best checkpoint and running inference on all 53 val images.

| Metric | Value |
|--------|-------|
| Validation accuracy | 0.6226 (33/53) |
| Macro F1 | 0.6311 |
| Weighted F1 | 0.6201 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| flooding | 0.6667 | 0.8000 | 0.7273 | 15 |
| non_environmental | 0.6000 | 0.6000 | 0.6000 | 15 |
| pollution | 0.8333 | 0.6250 | 0.7143 | 8 |
| waste | 0.5000 | 0.4667 | 0.4828 | 15 |

### Confusion matrix (rows=True, cols=Predicted)

|  | flooding | non_env | pollution | waste |
|--|---------|---------|-----------|-------|
| flooding | 12 | 2 | 0 | 1 |
| non_environmental | 2 | 9 | 0 | 4 |
| pollution | 0 | 1 | 5 | 2 |
| waste | 4 | 3 | 1 | 7 |

### Dominant bidirectional confusion pairs

| Pair | Total |
|------|-------|
| waste <-> non_environmental | 7 (3+4) |
| waste <-> flooding | 5 (4+1) |

---

## 12. Known Issues and Deviations

### 12.1 Stage 1 freeze duration shorter than intended
Intended: 5 frozen epochs. Actual: 4 frozen epochs. Evidence: task-398 log. Impact unknown.

### 12.2 CPU-only execution
The dGPU was disabled via G-Helper during the run. Results are deterministic. GPU runs may show minor numerical differences.

### 12.3 pin_memory warning
`pin_memory=True` with no accelerator triggered a PyTorch UserWarning. No functional impact.

### 12.4 best_epoch missing from config.json
`config.json` stores `early_stop_epoch: 10` but not a `best_epoch` key. Best epoch is confirmed as 10 by matching `best_val_loss = 1.7228` to the training history.

### 12.5 Substantial overfitting
Train acc 0.9714 vs val acc 0.6226 at best epoch. Large train/val gap motivates Experiment 2 regularization additions.

---

## 13. Artifact Inventory

| File | Description |
|------|-------------|
| baseline_validation_report.md | Corrected summary report |
| classification_report.json | Per-class sklearn report |
| val_predictions.json | Per-image predictions + confidences |
| training_history.csv | Per-epoch loss/accuracy |
| training_curves.png | Loss and accuracy plots |
| validation_confusion_matrix.png | Confusion matrix heatmap |
| validation_error_analysis.md | Error breakdown with bidirectional table |
| misclassification_contact_sheet.png | All 20 misclassified images |
| config.json | Training config as saved by script |
| baseline_reproducibility_audit.md | This document |
| ../../checkpoints/efficientnet_b0_best.pt | Best model checkpoint (epoch 10) |

---
*This audit must not be modified to retroactively change historical metrics or execution behavior.*
