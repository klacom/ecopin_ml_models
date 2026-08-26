import json
import pandas as pd
import numpy as np
import os

PROJECT_ROOT = r'c:\dev\ecopin_image_validation'
E3_ARTIFACT  = os.path.join(PROJECT_ROOT, 'artifacts', 'exp3_efficientnet_b0')
CB_ARTIFACT  = os.path.join(PROJECT_ROOT, 'artifacts', 'corrected_baseline_efficientnet_b0')
CLASS_NAMES  = ['flooding', 'non_environmental', 'pollution', 'waste']

# Load data
with open(os.path.join(E3_ARTIFACT, 'per_epoch_metrics.json'), encoding='utf-8') as f:
    e3_epoch_metrics = {int(k): v for k, v in json.load(f).items()}
with open(os.path.join(CB_ARTIFACT, 'per_epoch_metrics.json'), encoding='utf-8') as f:
    cb_epoch_metrics = {int(k): v for k, v in json.load(f).items()}

e3_history = pd.read_csv(os.path.join(E3_ARTIFACT, 'training_history.csv'))
cb_history = pd.read_csv(os.path.join(CB_ARTIFACT, 'training_history.csv'))

# Find Exp 3 bests
e3_best_val_loss = min(e3_epoch_metrics.items(), key=lambda x: x[1]['val_loss'])
e3_best_val_acc  = max(e3_epoch_metrics.items(), key=lambda x: x[1]['val_acc'])
e3_best_macro_f1 = max(e3_epoch_metrics.items(), key=lambda x: x[1]['macro_f1'])

print("=== A. EXPERIMENT 3 RUN INTEGRITY & EPOCHS ===")
for _, r in e3_history.iterrows():
    stage = r.get('stage', '?')
    print(f"  Ep{int(r['epoch']):02d} [S{int(stage)}] train_acc={r['train_acc']:.4f} val_loss={r['val_loss']:.4f} val_acc={r['val_acc']:.4f} macro_f1={r['macro_f1']:.4f}")

print("\n=== B. EXPERIMENT 3 BEST EPOCHS ===")
print(f"  By val_loss  : epoch {e3_best_val_loss[0]} | val_loss={e3_best_val_loss[1]['val_loss']:.4f} val_acc={e3_best_val_loss[1]['val_acc']:.4f} macro_f1={e3_best_val_loss[1]['macro_f1']:.4f}")
print(f"  By val_acc   : epoch {e3_best_val_acc[0]}  | val_loss={e3_best_val_acc[1]['val_loss']:.4f} val_acc={e3_best_val_acc[1]['val_acc']:.4f} macro_f1={e3_best_val_acc[1]['macro_f1']:.4f}")
print(f"  By macro_f1  : epoch {e3_best_macro_f1[0]}  | val_loss={e3_best_macro_f1[1]['val_loss']:.4f} val_acc={e3_best_macro_f1[1]['val_acc']:.4f} macro_f1={e3_best_macro_f1[1]['macro_f1']:.4f}")

# Extract specific epoch data
cb_ep7 = cb_epoch_metrics[7]
cb_ep9 = cb_epoch_metrics[9]
e3_best = e3_best_macro_f1[1]

print("\n=== C. COMPARISON ===")
print(f"{'Metric':<20} {'CB (Ep7 - F1)':>15} {'CB (Ep9 - Loss)':>15} {'Exp3 (Best F1)':>15}")
print("-" * 70)
print(f"{'Val Accuracy':<20} {cb_ep7['val_acc']:>15.4f} {cb_ep9['val_acc']:>15.4f} {e3_best['val_acc']:>15.4f}")
print(f"{'Macro F1':<20} {cb_ep7['macro_f1']:>15.4f} {cb_ep9['macro_f1']:>15.4f} {e3_best['macro_f1']:>15.4f}")
print(f"{'Weighted F1':<20} {cb_ep7['weighted_f1']:>15.4f} {cb_ep9['weighted_f1']:>15.4f} {e3_best['weighted_f1']:>15.4f}")

print("\nPer-class F1:")
print(f"{'Class':<20} {'CB (Ep7 - F1)':>15} {'CB (Ep9 - Loss)':>15} {'Exp3 (Best F1)':>15}")
print("-" * 70)
for cls in CLASS_NAMES:
    print(f"{cls:<20} {cb_ep7['per_class'][cls]['f1-score']:>15.4f} {cb_ep9['per_class'][cls]['f1-score']:>15.4f} {e3_best['per_class'][cls]['f1-score']:>15.4f}")

print("\nConfusion Matrices:")
print("\nCB (Ep7 - Macro F1):")
print(pd.DataFrame(np.array(cb_ep7['confusion_matrix']), index=CLASS_NAMES, columns=CLASS_NAMES).to_string())
print("\nExp3 (Best Macro F1):")
print(pd.DataFrame(np.array(e3_best['confusion_matrix']), index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

print("\nMisclassification differences (CB Ep7 vs Exp3):")
print(f"{'Error':<30} {'CB Ep7':>10} {'Exp3':>10} {'Delta':>10}")
print("-" * 65)
errors = [
    (0,1,"flooding -> non_env"),
    (0,3,"flooding -> waste"),
    (1,0,"non_env -> flooding"),
    (1,3,"non_env -> waste"),
    (2,0,"pollution -> flooding"),
    (2,3,"pollution -> waste"),
    (3,0,"waste -> flooding"),
    (3,1,"waste -> non_env"),
    (3,2,"waste -> pollution"),
]
cm_cb7 = np.array(cb_ep7['confusion_matrix'])
cm_e3 = np.array(e3_best['confusion_matrix'])
for i, j, label in errors:
    v_cb7 = cm_cb7[i, j]
    v_e3  = cm_e3[i, j]
    print(f"{label:<30} {v_cb7:>10} {v_e3:>10} {v_e3 - v_cb7:>10}")
