import json
import pandas as pd

with open('artifacts/baseline_efficientnet_b0/classification_report.json') as f: cr1 = json.load(f)
with open('artifacts/exp2_efficientnet_b0/classification_report.json') as f: cr2 = json.load(f)

classes = ['flooding', 'non_environmental', 'pollution', 'waste']

print('--- F1 Scores Comparison ---')
print(f'Class              | Baseline | Exp2   | Delta')
print('-'*50)
for c in classes:
    f1_1 = cr1[c]['f1-score']
    f1_2 = cr2[c]['f1-score']
    print(f'{c:<18} | {f1_1:.4f}   | {f1_2:.4f} | {f1_2-f1_1:+.4f}')

print('\n--- Exp2 Misclassifications Summary ---')
confusions = {}
with open('artifacts/exp2_efficientnet_b0/val_predictions.json', encoding='utf-8') as f: preds = json.load(f)
for p in preds:
    if not p['correct']:
        k = f"{p['true_label']} -> {p['pred_label']}"
        confusions[k] = confusions.get(k, 0) + 1

for k, v in sorted(confusions.items(), key=lambda x: -x[1]):
    print(f'{k}: {v}')
