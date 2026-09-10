# EfficientNet-B0 Corrected Baseline â€” Validation Report

**Experiment:** `corrected_baseline_efficientnet_b0`
**Device:** cuda (NVIDIA GeForce RTX 5050 Laptop GPU)
**torch:** 2.13.0+cu132  **CUDA:** 13.2
**Batch size:** 16  **Seed:** 42
**Total epochs run:** 14
**Best checkpoint epoch:** 9 (selected by Macro F1)
**Best Macro F1:** 0.5782
**Best val accuracy:** 0.6038
**Best val loss:** 1.6290

## Changes vs. Historical Baseline

| Item | Historical Baseline | Corrected Baseline | Type |
|------|--------------------|--------------------|------|
| Stage 1 freeze duration | 4 epochs (actual) | **5 epochs (corrected)** | Protocol correction |
| Checkpoint-selection metric | val_loss (minimize) | val_loss (minimize) â€” **UNCHANGED** | â€” |
| Per-epoch checkpoint saving | No | **Yes** | Observability |
| Per-epoch metrics JSON | No | **Yes** | Observability |
| Macro F1 in history CSV | No | **Yes** | Observability |
| Device | CPU | **CUDA (RTX 5050)** | Environment |
| pin_memory | True | **False** | Environment fix |
| num_workers | 2 | **0** | Environment fix |

## Unchanged from Historical Baseline

Augmentation, loss, optimizer, LRs, scheduler, batch size, max epochs, patience,
seed, preprocessing, model architecture, class weights.

## Validation Metrics (best checkpoint â€” epoch 9)

| Metric | Value |
|--------|-------|
| Accuracy | 0.6038 (32/53) |
| Macro F1 | 0.5782 |
| Weighted F1 | 0.6080 |

## Per-Class Results

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| flooding | 0.7857 | 0.7333 | 0.7586 | 15 |
| non_environmental | 0.5625 | 0.6000 | 0.5806 | 15 |
| pollution | 0.3333 | 0.3750 | 0.3529 | 8 |
| waste | 0.6429 | 0.6000 | 0.6207 | 15 |

## Confusion Matrix (rows=True, cols=Predicted)

| True \ Pred | flooding | non_environmental | pollution | waste |
|-------------|------|------|------|------|
| flooding | 11 | 4 | 0 | 0 |
| non_environmental | 2 | 9 | 2 | 2 |
| pollution | 1 | 1 | 3 | 3 |
| waste | 0 | 2 | 4 | 9 |

## Most Common Confusions

| True | Predicted | Count |
|------|-----------|-------|
| waste | pollution | 4 |
| flooding | non_environmental | 4 |
| pollution | waste | 3 |
| waste | non_environmental | 2 |
| non_environmental | waste | 2 |
| non_environmental | pollution | 2 |

## Overfitting Indicator

| | Value |
|--|-------|
| Train accuracy at best epoch | 0.9347 |
| Val accuracy at best epoch | 0.6038 |
| Trainâˆ’Val gap | 0.3309 |

## Misclassifications (21 / 53)

| Image ID | True | Predicted | Confidence |
|----------|------|-----------|------------|
| FLD_099 | flooding | non_environmental | 0.830 |
| FLD_033 | flooding | non_environmental | 0.876 |
| FLD_085 | flooding | non_environmental | 0.987 |
| FLD_002 | flooding | non_environmental | 0.743 |
| NEG_059 | non_environmental | flooding | 0.980 |
| NEG_099 | non_environmental | waste | 0.999 |
| NEG_076 | non_environmental | pollution | 0.899 |
| NEG_060 | non_environmental | waste | 0.997 |
| NEG_085 | non_environmental | pollution | 0.450 |
| NEG_030 | non_environmental | flooding | 0.674 |
| POL_044 | pollution | flooding | 0.601 |
| POL_024 | pollution | non_environmental | 0.598 |
| POL_011 | pollution | waste | 0.862 |
| POL_023 | pollution | waste | 0.967 |
| POL_019 | pollution | waste | 0.560 |
| WST_059 | waste | pollution | 0.907 |
| WST_049 | waste | pollution | 0.602 |
| WST_058 | waste | non_environmental | 0.668 |
| WST_060 | waste | pollution | 0.544 |
| WST_064 | waste | non_environmental | 0.900 |
| WST_085 | waste | pollution | 0.847 |

## Interpretation Notes

- Validation set is 53 images. A 1-image change â‰ˆ 1.9 pp accuracy swing.
- Best epoch selected by **val_loss (minimize)** â€” identical to historical baseline protocol.
- Per-epoch checkpoints saved for retrospective analysis: identify minimum val_loss epoch
  (selected checkpoint) and maximum Macro F1 epoch separately from the saved files.
- Test set was NOT evaluated.

All artifacts: `c:/dev/ecopin_image_validation/artifacts/corrected_baseline_efficientnet_b0/`
