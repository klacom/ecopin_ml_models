# EfficientNet-B0 Corrected Baseline â€” Validation Report

**Experiment:** `corrected_baseline_efficientnet_b0`
**Device:** cuda (NVIDIA GeForce RTX 5050 Laptop GPU)
**torch:** 2.13.0+cu132  **CUDA:** 13.2
**Batch size:** 16  **Seed:** 42
**Total epochs run:** 14
**Best checkpoint epoch:** 9 (selected by Macro F1)
**Best Macro F1:** 0.6229
**Best val accuracy:** 0.6226
**Best val loss:** 1.6353

## Changes vs. Corrected Baseline

| Item | Corrected Baseline | Experiment 3 | Type |
|------|--------------------|--------------|------|
| H-flip probability | 0.2 | **0.5** | Augmentation ablation |
| Rotation | None | **RandomRotation(15°)** | Augmentation ablation |
| Checkpoint metric | val_loss | **Macro F1** | Protocol update |

## Unchanged from Corrected Baseline

Augmentation, loss, optimizer, LRs, scheduler, batch size, max epochs, patience,
seed, preprocessing, model architecture, class weights.

## Validation Metrics (best checkpoint â€” epoch 9)

| Metric | Value |
|--------|-------|
| Accuracy | 0.6226 (33/53) |
| Macro F1 | 0.6229 |
| Weighted F1 | 0.6226 |

## Per-Class Results

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| flooding | 0.6111 | 0.7333 | 0.6667 | 15 |
| non_environmental | 0.5333 | 0.5333 | 0.5333 | 15 |
| pollution | 0.6250 | 0.6250 | 0.6250 | 8 |
| waste | 0.7500 | 0.6000 | 0.6667 | 15 |

## Confusion Matrix (rows=True, cols=Predicted)

| True \ Pred | flooding | non_environmental | pollution | waste |
|-------------|------|------|------|------|
| flooding | 11 | 4 | 0 | 0 |
| non_environmental | 3 | 8 | 2 | 2 |
| pollution | 1 | 1 | 5 | 1 |
| waste | 3 | 2 | 1 | 9 |

## Most Common Confusions

| True | Predicted | Count |
|------|-----------|-------|
| flooding | non_environmental | 4 |
| waste | flooding | 3 |
| non_environmental | flooding | 3 |
| waste | non_environmental | 2 |
| non_environmental | waste | 2 |
| non_environmental | pollution | 2 |

## Overfitting Indicator

| | Value |
|--|-------|
| Train accuracy at best epoch | 0.8776 |
| Val accuracy at best epoch | 0.6226 |
| Trainâˆ’Val gap | 0.2549 |

## Misclassifications (20 / 53)

| Image ID | True | Predicted | Confidence |
|----------|------|-----------|------------|
| FLD_099 | flooding | non_environmental | 0.682 |
| FLD_033 | flooding | non_environmental | 0.976 |
| FLD_085 | flooding | non_environmental | 0.999 |
| FLD_002 | flooding | non_environmental | 0.572 |
| NEG_080 | non_environmental | flooding | 0.738 |
| NEG_059 | non_environmental | flooding | 0.623 |
| NEG_099 | non_environmental | waste | 1.000 |
| NEG_076 | non_environmental | pollution | 0.548 |
| NEG_060 | non_environmental | waste | 1.000 |
| NEG_085 | non_environmental | flooding | 0.466 |
| NEG_038 | non_environmental | pollution | 0.803 |
| POL_044 | pollution | flooding | 0.928 |
| POL_024 | pollution | non_environmental | 0.672 |
| POL_023 | pollution | waste | 0.607 |
| WST_049 | waste | pollution | 0.793 |
| WST_058 | waste | non_environmental | 0.915 |
| WST_033 | waste | non_environmental | 0.647 |
| WST_060 | waste | flooding | 0.752 |
| WST_085 | waste | flooding | 0.745 |
| WST_030 | waste | flooding | 0.669 |

## Interpretation Notes

- Validation set is 53 images. A 1-image change ≈ 1.9 pp accuracy swing.
- Best epoch selected by **Macro F1 (maximize)**.
- Test set was NOT evaluated.

All artifacts: `c:/dev/ecopin_image_validation/artifacts/corrected_baseline_efficientnet_b0/`
