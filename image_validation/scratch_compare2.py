import json

print('--- Baseline Misclassifications Summary ---')
confusions = {}
with open('artifacts/baseline_efficientnet_b0/val_predictions.json', encoding='utf-8') as f: preds = json.load(f)
for p in preds:
    if not p['correct']:
        k = f"{p['true_label']} -> {p['pred_label']}"
        confusions[k] = confusions.get(k, 0) + 1

for k, v in sorted(confusions.items(), key=lambda x: -x[1]):
    print(f'{k}: {v}')
