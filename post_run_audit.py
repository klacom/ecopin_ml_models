import json
import pandas as pd
import numpy as np
import torch
from timm import create_model
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = r'c:\dev\ecopin_image_validation'
CB_ARTIFACT  = os.path.join(PROJECT_ROOT, 'artifacts', 'corrected_baseline_efficientnet_b0')
BL_ARTIFACT  = os.path.join(PROJECT_ROOT, 'artifacts', 'baseline_efficientnet_b0')
E2_ARTIFACT  = os.path.join(PROJECT_ROOT, 'artifacts', 'exp2_efficientnet_b0')
CB_CKPT_DIR  = os.path.join(PROJECT_ROOT, 'checkpoints', 'corrected_baseline')
BEST_CKPT    = os.path.join(CB_CKPT_DIR, 'best.pt')

CLASS_NAMES = ['flooding', 'non_environmental', 'pollution', 'waste']

# ── Load data ──────────────────────────────────────────────────────────────────
with open(os.path.join(CB_ARTIFACT, 'per_epoch_metrics.json'), encoding='utf-8') as f:
    epoch_metrics = {int(k): v for k, v in json.load(f).items()}

with open(os.path.join(CB_ARTIFACT, 'classification_report.json'), encoding='utf-8') as f:
    cb_report = json.load(f)

with open(os.path.join(BL_ARTIFACT, 'classification_report.json'), encoding='utf-8') as f:
    bl_report = json.load(f)

with open(os.path.join(E2_ARTIFACT, 'classification_report.json'), encoding='utf-8') as f:
    e2_report = json.load(f)

cb_history = pd.read_csv(os.path.join(CB_ARTIFACT, 'training_history.csv'))
bl_history = pd.read_csv(os.path.join(BL_ARTIFACT, 'training_history.csv'))

# ── A: Epoch summary ───────────────────────────────────────────────────────────
print("=== A. CORRECTED BASELINE TRAINING HISTORY ===")
for _, r in cb_history.iterrows():
    stage = r.get('stage', '?')
    print(f"  Ep{int(r['epoch']):02d} [S{int(stage)}] train_acc={r['train_acc']:.4f} "
          f"val_loss={r['val_loss']:.4f} val_acc={r['val_acc']:.4f} "
          f"macro_f1={r['macro_f1']:.4f}")

# ── B: Best epoch by each criterion ───────────────────────────────────────────
best_by_val_loss  = min(epoch_metrics.items(), key=lambda x: x[1]['val_loss'])
best_by_val_acc   = max(epoch_metrics.items(), key=lambda x: x[1]['val_acc'])
best_by_macro_f1  = max(epoch_metrics.items(), key=lambda x: x[1]['macro_f1'])

print("\n=== B. BEST EPOCH BY EACH CRITERION ===")
print(f"  By val_loss  : epoch {best_by_val_loss[0]} | val_loss={best_by_val_loss[1]['val_loss']:.4f} val_acc={best_by_val_loss[1]['val_acc']:.4f} macro_f1={best_by_val_loss[1]['macro_f1']:.4f}")
print(f"  By val_acc   : epoch {best_by_val_acc[0]}  | val_loss={best_by_val_acc[1]['val_loss']:.4f} val_acc={best_by_val_acc[1]['val_acc']:.4f} macro_f1={best_by_val_acc[1]['macro_f1']:.4f}")
print(f"  By macro_f1  : epoch {best_by_macro_f1[0]}  | val_loss={best_by_macro_f1[1]['val_loss']:.4f} val_acc={best_by_macro_f1[1]['val_acc']:.4f} macro_f1={best_by_macro_f1[1]['macro_f1']:.4f}")

print("\n  Per-class F1 at each best epoch:")
for label, (ep, data) in [("val_loss-best (ep9)", best_by_val_loss),
                           ("val_acc-best  (ep7)", best_by_val_acc),
                           ("macro_f1-best (ep7)", best_by_macro_f1)]:
    print(f"\n  [{label}]")
    for c in CLASS_NAMES:
        print(f"    {c:<20} F1={data['per_class'][c]['f1-score']:.4f}  P={data['per_class'][c]['precision']:.4f}  R={data['per_class'][c]['recall']:.4f}")

# ── C: Checkpoint integrity check ──────────────────────────────────────────────
print("\n=== C. CHECKPOINT INTEGRITY ===")
# Load best.pt and epoch_09.pt, compare state dict checksums
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_state  = torch.load(BEST_CKPT, map_location='cpu')
ep9_state   = torch.load(os.path.join(CB_CKPT_DIR, 'epoch_09.pt'), map_location='cpu')
ep7_state   = torch.load(os.path.join(CB_CKPT_DIR, 'epoch_07.pt'), map_location='cpu')

# Compare first layer's first weight tensor
key = list(best_state.keys())[0]
best_matches_ep9 = torch.allclose(best_state[key], ep9_state[key])
best_matches_ep7 = torch.allclose(best_state[key], ep7_state[key])
print(f"  best.pt matches epoch_09.pt: {best_matches_ep9}  [Expected: True]")
print(f"  best.pt matches epoch_07.pt: {best_matches_ep7}  [Expected: False]")

# Spot-check last layer too
key2 = list(best_state.keys())[-1]
best_matches_ep9_last = torch.allclose(best_state[key2], ep9_state[key2])
print(f"  Last-layer match ep9: {best_matches_ep9_last}  [Expected: True]")

# ── D: Three-way comparison ────────────────────────────────────────────────────
print("\n=== D. THREE-WAY COMPARISON ===")
print(f"\n{'Metric':<20} {'Hist BL':>10} {'Corr BL':>10} {'Exp2':>10}")
print("-"*55)

hl = bl_report
cl = cb_report
e2 = e2_report

rows = [
    ("Val Accuracy", "0.6226 (33/53)", "0.6038 (32/53)", "0.5660 (30/53)"),
    ("Macro F1",     f"{hl['macro avg']['f1-score']:.4f}", f"{cl['macro avg']['f1-score']:.4f}", f"{e2['macro avg']['f1-score']:.4f}"),
    ("Weighted F1",  f"{hl['weighted avg']['f1-score']:.4f}", f"{cl['weighted avg']['f1-score']:.4f}", f"{e2['weighted avg']['f1-score']:.4f}"),
]
for label, h, c, e in rows:
    print(f"  {label:<20} {h:>12} {c:>12} {e:>12}")

print(f"\n  Per-class F1:")
print(f"  {'Class':<22} {'Hist BL':>8} {'Corr BL':>8} {'Exp2':>8}")
print("  " + "-"*50)
for cls in CLASS_NAMES:
    h_f1 = hl[cls]['f1-score']
    c_f1 = cl[cls]['f1-score']
    e_f1 = e2[cls]['f1-score']
    print(f"  {cls:<22} {h_f1:>8.4f} {c_f1:>8.4f} {e_f1:>8.4f}")

print("\n  Overfitting gap (train_acc - val_acc at best epoch):")
cb_best_row = cb_history[cb_history['epoch'] == 9].iloc[0]
print(f"  Hist BL  : train_acc=0.9714 val_acc=0.6226 gap=0.3488 (ep10)")
print(f"  Corr BL  : train_acc={cb_best_row['train_acc']:.4f} val_acc={cb_best_row['val_acc']:.4f} gap={cb_best_row['train_acc']-cb_best_row['val_acc']:.4f} (ep9)")
print(f"  Exp2     : train_acc=0.7510 val_acc=0.5660 gap=0.1850 (ep8)")

# Confusion matrices
print("\n  Corrected Baseline confusion matrix (ep9):")
cm_cb = np.array(epoch_metrics[9]['confusion_matrix'])
print(pd.DataFrame(cm_cb, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

# Directional errors comparison
print("\n  Directional error comparison:")
print(f"  {'Error':35} {'Hist BL':>8} {'Corr BL':>8} {'Exp2':>8}")
print("  " + "-"*60)

# Historical BL confusion (from known values)
bl_cm = np.array([
    [12, 2, 0, 1],
    [2, 9, 0, 4],
    [0, 1, 6, 2],
    [4, 3, 1, 7],
])
# Exp2 confusion
e2_cm = np.array([
    [11, 3, 0, 1],
    [3, 9, 0, 3],
    [3, 1, 3, 1],
    [1, 3, 4, 7],
])

errors_of_interest = [
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
for i, j, label in errors_of_interest:
    print(f"  {label:<35} {bl_cm[i,j]:>8} {cm_cb[i,j]:>8} {e2_cm[i,j]:>8}")
