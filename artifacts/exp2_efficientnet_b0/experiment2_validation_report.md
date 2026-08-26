# EfficientNet-B0 Experiment 2 â€” Validation Report

**Dataset:** Ecopin Pilot v1.0 (train/val only â€” test set untouched)
**Device:** cuda (NVIDIA GeForce RTX 5050 Laptop GPU)
**torch:** 2.13.0+cu132  **CUDA:** 13.2
**Batch size:** 16  **Seed:** 42
**Total epochs run:** 13
**Best checkpoint epoch:** 8
**Best val loss:** 2.0988

## Key Changes vs Baseline

| Change | Baseline | Experiment 2 |
|--------|----------|--------------|
| Stage 1 frozen epochs | 4 (actual) | **5 (corrected)** |
| Weight decay | None | **1e-4** |
| Label smoothing | None | **0.1** |
| Dropout (drop_rate) | 0.0 (timm default) | **0.2** |
| H-flip prob | 0.2 | **0.5** |
| Rotation | None | **Â±15Â°** |
| ColorJitter | None | **b=0.2, c=0.2, s=0.1** |
| Device | CPU | **CUDA** |

## Comparison Table

| Metric | Baseline | Experiment 2 | Difference |
|--------|----------|--------------|------------|
| Validation Accuracy | 0.6226 | 0.5660 | -0.0566 â†“ |
| Macro F1 | 0.6311 | 0.5415 | -0.0896 â†“ |
| Weighted F1 | 0.6201 | 0.5601 | -0.0600 â†“ |
| Waste F1 | 0.4828 | 0.5185 | ++0.0357 â†‘ |
| Best Val Loss | 1.7228 | 2.0988 | ++0.3760 â†“ |
| Best Epoch | 10 | 8 | -2 |

## Overfitting Comparison

| | Baseline @ best epoch | Experiment 2 @ best epoch |
|--|----------------------|--------------------------|
| Train accuracy | 0.9714 | 0.7510 |
| Val accuracy | 0.6226 | 0.5660 |
| Trainâˆ’Val gap | 0.3488 | 0.1850 |

## Overall Validation Metrics (best checkpoint)

| Metric | Value |
|--------|-------|
| Validation accuracy | 0.5660 (30/53) |
| Macro F1 | 0.5415 |
| Weighted F1 | 0.5601 |

## Per-Class Results

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| flooding | 0.6111 | 0.7333 | 0.6667 | 15 |\n| non_environmental | 0.5625 | 0.6000 | 0.5806 | 15 |\n| pollution | 0.4286 | 0.3750 | 0.4000 | 8 |\n| waste | 0.5833 | 0.4667 | 0.5185 | 15 |\n
## Confusion Matrix (rows=True, cols=Predicted)

| True \ Pred | flooding | non_environmental | pollution | waste |
|-------------|------|------|------|------|
| flooding | 11 | 3 | 0 | 1 |\n| non_environmental | 3 | 9 | 0 | 3 |\n| pollution | 3 | 1 | 3 | 1 |\n| waste | 1 | 3 | 4 | 7 |\n\n## Most Common Confusions\n\n| True | Predicted | Count |\n|------|-----------|-------|\n| waste | pollution | 4 |\n| waste | non_environmental | 3 |\n| pollution | flooding | 3 |\n| non_environmental | waste | 3 |\n| non_environmental | flooding | 3 |\n| flooding | non_environmental | 3 |\n
## Misclassifications (23 / 53)

| Image ID | True | Predicted | Confidence |
|----------|------|-----------|------------|
| FLD_099 | flooding | waste | 0.613 |\n| FLD_033 | flooding | non_environmental | 0.666 |\n| FLD_085 | flooding | non_environmental | 0.997 |\n| FLD_002 | flooding | non_environmental | 0.986 |\n| NEG_080 | non_environmental | waste | 0.689 |\n| NEG_099 | non_environmental | waste | 1.000 |\n| NEG_076 | non_environmental | flooding | 0.713 |\n| NEG_060 | non_environmental | waste | 0.978 |\n| NEG_085 | non_environmental | flooding | 0.915 |\n| NEG_030 | non_environmental | flooding | 0.480 |\n| POL_044 | pollution | non_environmental | 0.502 |\n| POL_024 | pollution | flooding | 0.682 |\n| POL_011 | pollution | flooding | 0.491 |\n| POL_023 | pollution | waste | 0.558 |\n| POL_019 | pollution | flooding | 0.478 |\n| WST_059 | waste | pollution | 0.798 |\n| WST_049 | waste | pollution | 0.881 |\n| WST_058 | waste | pollution | 0.545 |\n| WST_033 | waste | non_environmental | 0.897 |\n| WST_060 | waste | non_environmental | 0.566 |\n| WST_064 | waste | non_environmental | 0.646 |\n| WST_085 | waste | pollution | 0.607 |\n| WST_030 | waste | flooding | 0.962 |\n
## Interpretation Notes

- **Validation set is 53 images.** A 1-image change corresponds to ~1.9 pp accuracy swing.
  Any comparison should be interpreted carefully â€” individual image differences are meaningful.
- Comparison should focus on pattern changes (which classes improved/degraded), not just headline accuracy.
- The overfitting gap (trainâˆ’val accuracy) is the primary indicator of whether regularization helped.

## Visuals
- Training curves: `training_curves.png`
- Confusion matrix: `validation_confusion_matrix.png`

All artifacts in `c:/dev/ecopin_image_validation/artifacts/exp2_efficientnet_b0/`
