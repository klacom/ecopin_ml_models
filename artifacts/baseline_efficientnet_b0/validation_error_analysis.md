# Validation Error Analysis – EfficientNet-B0 Baseline

**Total validation images:** 53  
**Correct predictions:** 33  
**Misclassifications:** 20  
**Accuracy:** 0.6226

## Misclassification Table

| # | Image ID | True Label | Predicted | Confidence |
|---|----------|-----------|-----------|------------|
| 1 | FLD_002 | flooding | non_environmental | 0.875 |
| 2 | FLD_033 | flooding | non_environmental | 0.794 |
| 3 | FLD_099 | flooding | waste | 0.840 |
| 4 | NEG_033 | non_environmental | flooding | 0.837 |
| 5 | NEG_038 | non_environmental | waste | 0.681 |
| 6 | NEG_059 | non_environmental | waste | 0.984 |
| 7 | NEG_076 | non_environmental | flooding | 0.847 |
| 8 | NEG_080 | non_environmental | waste | 0.941 |
| 9 | NEG_095 | non_environmental | waste | 0.627 |
| 10 | POL_011 | pollution | waste | 0.703 |
| 11 | POL_019 | pollution | non_environmental | 0.939 |
| 12 | POL_024 | pollution | waste | 0.717 |
| 13 | WST_030 | waste | flooding | 0.903 |
| 14 | WST_033 | waste | flooding | 0.968 |
| 15 | WST_049 | waste | pollution | 0.611 |
| 16 | WST_058 | waste | non_environmental | 0.802 |
| 17 | WST_060 | waste | non_environmental | 0.988 |
| 18 | WST_064 | waste | non_environmental | 0.999 |
| 19 | WST_095 | waste | flooding | 0.950 |
| 20 | WST_099 | waste | flooding | 0.891 |

## Errors by True Class

### flooding (3 errors / 15 samples)

- Predicted as **non_environmental**: 2 time(s)
- Predicted as **waste**: 1 time(s)

### non_environmental (6 errors / 15 samples)

- Predicted as **waste**: 4 time(s)
- Predicted as **flooding**: 2 time(s)

### pollution (3 errors / 8 samples)

- Predicted as **waste**: 2 time(s)
- Predicted as **non_environmental**: 1 time(s)

### waste (8 errors / 15 samples)

- Predicted as **flooding**: 4 time(s)
- Predicted as **non_environmental**: 3 time(s)
- Predicted as **pollution**: 1 time(s)

## Analysis Notes

**Bidirectional confusion totals (from confusion matrix):**

| Pair | Direction A | Direction B | Total |
|------|------------|------------|-------|
| waste ↔ non_environmental | waste→non_env: 3 | non_env→waste: 4 | **7** |
| waste ↔ flooding | waste→flooding: 4 | flooding→waste: 1 | **5** |
| non_env ↔ flooding | non_env→flooding: 2 | flooding→non_env: 2 | 4 |
| pollution ↔ waste | pollution→waste: 2 | waste→pollution: 1 | 3 |

- **waste** has the lowest F1 (0.4828) — confused in all directions; 8 of 15 samples misclassified.
- The **dominant confusion pair is waste ↔ non_environmental** (7 cross-direction errors total), not waste ↔ flooding.
- **waste → flooding** (4 errors) dominates the single direction, but flooding → waste contributes only 1, giving 5 total.
- **flooding** is the best-performing class (highest recall: 0.80).
- **pollution** is high-precision but lower-recall — some pollution scenes misclassified as waste or non_environmental.
- Several errors are very high-confidence (WST_064 conf=0.999, WST_033 conf=0.968), suggesting the model is overfit to misleading visual features rather than simply uncertain.

## Recommendations for Experiment 2

*(Pending user approval — do not run yet.)*

1. **Targeted augmentation for waste**: random rotation, colour jitter to improve invariance.
2. **Class-weighted loss**: already applied; verify weight ratios are appropriate for current split.
3. **Label smoothing** (ε=0.1): may reduce over-confident predictions on the ambiguous non_env/waste boundary.
4. **Mixup** (α=0.2): soft-label mixing on the most confused pair.

*(Each change should be ablated individually before combining.)*
