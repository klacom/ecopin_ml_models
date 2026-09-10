# EfficientNet-B0 Baseline – Validation Report

**Dataset:** Ecopin Pilot v1.0 (train/val only — test set untouched)
**Device:** cpu
**Final batch size:** 16
**Total epochs run:** 15
**Best checkpoint epoch:** 10 (of 15 epochs run; early-stopped at patience=5)
**Best val loss (checkpoint):** 1.72280201821957

> **Stage 1 / Stage 2 note:** The task-398 training log shows `--- Transition to Stage 2` printed between epoch 04 and epoch 05, meaning the backbone was frozen for **epochs 1–4 only** in this actual run (`STAGE1_EPOCHS` was effectively 4 at execution time). Epoch 5 onward ran with the backbone unfrozen at lr=1e-4. The current `train_baseline.py` has been subsequently corrected to `STAGE1_EPOCHS = 5`; that corrected value applies to future experiments only, not to this completed baseline.

## Overall Validation Metrics (best checkpoint)

| Metric | Value |
|--------|-------|
| Validation Accuracy | 0.6226 (33/53) |
| Macro F1 | 0.6311 |
| Weighted F1 | 0.6201 |

## Best-Epoch Training Metrics

| Metric | Value |
|--------|-------|
| Train Accuracy | 0.9714 |
| Val Accuracy | 0.6226 |
| Train Loss | 0.0870 |
| Val Loss | 1.7228 |

## Per-Class Results

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| flooding | 0.6667 | 0.8000 | 0.7273 | 15 |
| non_environmental | 0.6000 | 0.6000 | 0.6000 | 15 |
| pollution | 0.8333 | 0.6250 | 0.7143 | 8 |
| waste | 0.5000 | 0.4667 | 0.4828 | 15 |

## Confusion Matrix (Rows = True, Columns = Predicted)

| True \ Pred | flooding | non_environmental | pollution | waste |
|-------------|------|------|------|------|
| flooding | 12 | 2 | 0 | 1 |
| non_environmental | 2 | 9 | 0 | 4 |
| pollution | 0 | 1 | 5 | 2 |
| waste | 4 | 3 | 1 | 7 |

## Most Common Confusions

| True | Predicted | Count |
|------|-----------|-------|
| waste | flooding | 4 |
| non_environmental | waste | 4 |
| waste | non_environmental | 3 |
| pollution | waste | 2 |
| non_environmental | flooding | 2 |
| flooding | non_environmental | 2 |

## Misclassifications (20 / 53)

| Image ID | True Label | Predicted | Confidence |
|----------|-----------|-----------|------------|
| FLD_099 | flooding | waste | 0.840 |
| FLD_033 | flooding | non_environmental | 0.794 |
| FLD_002 | flooding | non_environmental | 0.875 |
| NEG_080 | non_environmental | waste | 0.941 |
| NEG_059 | non_environmental | waste | 0.984 |
| NEG_076 | non_environmental | flooding | 0.847 |
| NEG_033 | non_environmental | flooding | 0.837 |
| NEG_095 | non_environmental | waste | 0.627 |
| NEG_038 | non_environmental | waste | 0.681 |
| POL_024 | pollution | waste | 0.717 |
| POL_011 | pollution | waste | 0.703 |
| POL_019 | pollution | non_environmental | 0.939 |
| WST_049 | waste | pollution | 0.611 |
| WST_099 | waste | flooding | 0.891 |
| WST_058 | waste | non_environmental | 0.802 |
| WST_033 | waste | flooding | 0.968 |
| WST_095 | waste | flooding | 0.950 |
| WST_060 | waste | non_environmental | 0.988 |
| WST_064 | waste | non_environmental | 0.999 |
| WST_030 | waste | flooding | 0.903 |

## Visuals
- Training curves: `artifacts/baseline_efficientnet_b0/training_curves.png`
- Confusion matrix: `artifacts/baseline_efficientnet_b0/validation_confusion_matrix.png`

All artifacts stored in `c:/dev/ecopin_image_validation/artifacts/baseline_efficientnet_b0/`
