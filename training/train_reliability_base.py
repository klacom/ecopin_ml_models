"""
train_reliability_base.py -- EfficientNet-B0 Multi-Seed Reliability Base Script

Shared training logic for Experiments 9-14 (multi-seed reliability protocol).
Accepts command-line arguments to avoid duplicating 6 large scripts.

Usage:
    python training/train_reliability_base.py --exp_id exp9 --seed 0 --config A
    python training/train_reliability_base.py --exp_id exp10 --seed 0 --config B

Config A: RandomHorizontalFlip(p=0.2), NO rotation
Config B: RandomHorizontalFlip(p=0.2), RandomRotation(degrees=15)

PROTOCOL (identical to Experiments 4/8):
  - Model: EfficientNet-B0, ImageNet pretrained, 4 classes
  - Dataset: same train/val CSV splits, same preprocessing
  - Loss: CrossEntropyLoss(class_weights), no label_smoothing
  - Optimizer: Adam, LR_STAGE1=1e-3 -> LR_STAGE2=1e-4 at epoch 6
  - Scheduler: ReduceLROnPlateau(mode='min', factor=0.5, patience=2)
  - Stage 1: 5 frozen epochs; Stage 2: unfrozen from epoch 6
  - Checkpoint: Macro F1 (maximize), tie-break val_acc then val_loss
  - Early stopping patience: 5
  - No ColorJitter, No weight_decay, No dropout, No label_smoothing
  - Test set: NOT LOADED OR EVALUATED
  - pin_memory=False, num_workers=0
"""

import argparse
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
import platform

import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from timm import create_model
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from training.dataset import EcopinDataset
from training.utils import compute_class_weights
from data_config.dataset_paths_and_split_config import (
    TRAIN_CSV, VAL_CSV, RAW_DIR, PROCESSED_DIR
)


# ── Fixed hyperparameters (must NOT be varied across experiments) ────────────
MAX_EPOCHS    = 20
STAGE1_EPOCHS = 5
BATCH_SIZE    = 16
LR_STAGE1     = 1e-3
LR_STAGE2     = 1e-4
PATIENCE      = 5
CLASS_NAMES   = ['flooding', 'non_environmental', 'pollution', 'waste']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Preprocessing ─────────────────────────────────────────────────────────────
def _pad_to_square(img):
    w, h = img.size
    max_side   = max(w, h)
    pad_left   = (max_side - w) // 2
    pad_top    = (max_side - h) // 2
    pad_right  = max_side - w - pad_left
    pad_bottom = max_side - h - pad_top
    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)


def get_transform(train: bool, use_rotation: bool):
    """
    Config A (use_rotation=False): RandomHorizontalFlip(p=0.2) only.
    Config B (use_rotation=True):  RandomHorizontalFlip(p=0.2) + RandomRotation(15deg).
    Validation: deterministic pad-to-square -> Resize(224) -> ToTensor -> Normalize.
    """
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(p=0.2))
        if use_rotation:
            transforms.append(T.RandomRotation(degrees=15))
    transforms.extend([
        T.Lambda(_pad_to_square),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return T.Compose(transforms)


# ── Macro F1 Early Stopping ──────────────────────────────────────────────────
class MacroF1EarlyStopping:
    """
    Monitors Macro F1 (maximize).
    Tie-breaker 1: higher val_accuracy.
    Tie-breaker 2: lower val_loss.
    """
    def __init__(self, patience: int, best_ckpt_path: str):
        self.patience      = patience
        self.counter       = 0
        self.best_macro_f1 = None
        self.best_val_acc  = None
        self.best_val_loss = None
        self.best_epoch    = None
        self.early_stop    = False
        self.best_ckpt_path = best_ckpt_path

    def _is_improvement(self, macro_f1, val_acc, val_loss):
        if self.best_macro_f1 is None:
            return True
        if macro_f1 > self.best_macro_f1 + 1e-6:
            return True
        if abs(macro_f1 - self.best_macro_f1) <= 1e-6:
            if val_acc > self.best_val_acc + 1e-6:
                return True
            if abs(val_acc - self.best_val_acc) <= 1e-6:
                if val_loss < self.best_val_loss - 1e-6:
                    return True
        return False

    def __call__(self, epoch, macro_f1, val_acc, val_loss, model):
        if self._is_improvement(macro_f1, val_acc, val_loss):
            self.best_macro_f1 = macro_f1
            self.best_val_acc  = val_acc
            self.best_val_loss = val_loss
            self.best_epoch    = epoch
            torch.save(model.state_dict(), self.best_ckpt_path)
            self.counter = 0
            print(f'  [EarlyStopping] New best Macro F1={macro_f1:.4f} '
                  f'(val_acc={val_acc:.4f}, val_loss={val_loss:.4f}) '
                  f'at epoch {epoch} -> best checkpoint saved')
        else:
            self.counter += 1
            print(f'  [EarlyStopping] No improvement. Counter: {self.counter}/{self.patience} '
                  f'(best Macro F1={self.best_macro_f1:.4f} at epoch {self.best_epoch})')
            if self.counter >= self.patience:
                self.early_stop = True


# ── Validation helper ─────────────────────────────────────────────────────────
def run_validation(model, val_loader, criterion):
    model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss_sum += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss = val_loss_sum / val_total
    val_acc  = val_correct  / val_total
    report   = classification_report(all_labels, all_preds,
                                     target_names=CLASS_NAMES,
                                     output_dict=True, zero_division=0)
    macro_f1 = report['macro avg']['f1-score']
    return val_loss, val_acc, macro_f1, np.array(all_preds), np.array(all_labels), report


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_training_curves(history_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (y1, y2, title, ylabel) in zip(axes, [
        ('train_loss', 'val_loss', 'Loss Curves', 'Loss'),
        ('train_acc',  'val_acc',  'Accuracy Curves', 'Accuracy'),
    ]):
        ax.plot(history_df['epoch'], history_df[y1], label='Train')
        ax.plot(history_df['epoch'], history_df[y2], label='Val')
        ax.axvline(x=5.5, color='grey', linestyle='--', linewidth=0.8, label='Stage 1->2')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_macro_f1_curve(history_df, output_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history_df['epoch'], history_df['macro_f1'], marker='o', label='Val Macro F1')
    best_ep = history_df.loc[history_df['macro_f1'].idxmax()]
    ax.axvline(x=best_ep['epoch'], color='red', linestyle='--', linewidth=0.8,
               label=f'Best epoch {int(best_ep["epoch"])} F1={best_ep["macro_f1"]:.4f}')
    ax.axvline(x=5.5, color='grey', linestyle='--', linewidth=0.8, label='Stage 1->2')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Macro F1')
    ax.set_title('Validation Macro F1 Curve')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_confusion_matrix(cm, class_names, output_path, title='Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(ticks); ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('True label'); ax.set_xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def train(exp_id: str, seed: int, config: str):
    use_rotation = (config == 'B')

    # Set paths
    artifact_root   = os.path.join(PROJECT_ROOT, 'artifacts', f'{exp_id}_efficientnet_b0')
    checkpoint_root = os.path.join(PROJECT_ROOT, 'checkpoints', exp_id)
    best_ckpt_path  = os.path.join(checkpoint_root, 'best.pt')
    os.makedirs(artifact_root,   exist_ok=True)
    os.makedirs(checkpoint_root, exist_ok=True)

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    config_desc = ('H-flip 0.2 + Rotation 15deg' if use_rotation
                   else 'H-flip 0.2, No Rotation')
    print(f'=== {exp_id.upper()} | Seed={seed} | Config {config} ({config_desc}) ===')
    print(f'Device              : {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU name            : {torch.cuda.get_device_name(0)}')
        print(f'GPU memory          : {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB')

    # ── Data loaders ──────────────────────────────────────────────────────────
    train_dataset = EcopinDataset(TRAIN_CSV, transform=get_transform(True,  use_rotation))
    val_dataset   = EcopinDataset(VAL_CSV,   transform=get_transform(False, use_rotation))
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=0, pin_memory=False)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0, pin_memory=False)
    print(f'Train samples       : {len(train_dataset)}')
    print(f'Val samples         : {len(val_dataset)}')

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = compute_class_weights(TRAIN_CSV).to(DEVICE)
    print('Class weights       :', {c: round(float(w), 4)
                                    for c, w in zip(CLASS_NAMES, class_weights.cpu())})

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = create_model('efficientnet_b0', pretrained=True, num_classes=4)
    model = model.to(DEVICE)
    print(f'Model params device : {next(model.parameters()).device}')

    # ── Stage 1: freeze backbone ──────────────────────────────────────────────
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f'Stage 1: backbone frozen ({frozen_count} tensors frozen)')
    print(f'STAGE1_EPOCHS={STAGE1_EPOCHS}: frozen epochs 1-{STAGE1_EPOCHS}, '
          f'unfrozen from epoch {STAGE1_EPOCHS+1}.')

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_STAGE1
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    early_stop = MacroF1EarlyStopping(patience=PATIENCE, best_ckpt_path=best_ckpt_path)

    history       = []
    epoch_metrics = {}

    for epoch in range(1, MAX_EPOCHS + 1):
        stage = 1 if epoch <= STAGE1_EPOCHS else 2

        # Training pass
        model.train()
        train_loss_sum, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * imgs.size(0)
            preds   = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)
        train_loss = train_loss_sum / total
        train_acc  = correct / total

        # Validation pass
        val_loss, val_acc, macro_f1, all_preds, all_labels, report = \
            run_validation(model, val_loader, criterion)
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(4)))
        current_lr = optimizer.param_groups[0]['lr']

        history.append({
            'epoch': epoch, 'stage': stage,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc,
            'macro_f1': macro_f1,
            'weighted_f1': report['weighted avg']['f1-score'],
            'learning_rate': current_lr,
        })

        # Per-epoch checkpoint (observability only)
        torch.save(model.state_dict(), os.path.join(checkpoint_root, f'epoch_{epoch:02d}.pt'))

        epoch_metrics[epoch] = {
            'val_loss': val_loss, 'val_acc': val_acc,
            'macro_f1': macro_f1,
            'weighted_f1': report['weighted avg']['f1-score'],
            'per_class': {c: {k: report[c][k]
                              for k in ['precision', 'recall', 'f1-score', 'support']}
                          for c in CLASS_NAMES},
            'confusion_matrix': cm.tolist(),
        }

        print(f'Epoch {epoch:02d} [S{stage}] | '
              f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} | '
              f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} | '
              f'macro_f1={macro_f1:.4f} | lr={current_lr:.6f}')

        # Scheduler monitors val_loss (identical to all previous experiments)
        scheduler.step(val_loss)

        # Macro F1 early stopping
        early_stop(epoch, macro_f1, val_acc, val_loss, model)
        if early_stop.early_stop:
            print(f'Early stopping triggered at epoch {epoch}')
            break

        # Stage transition at end of epoch STAGE1_EPOCHS
        if epoch == STAGE1_EPOCHS:
            print(f'--- Transition to Stage 2 after epoch {epoch}: '
                  f'unfreeze backbone, LR={LR_STAGE2}')
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=LR_STAGE2)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2
            )

    # ── Save artefacts ────────────────────────────────────────────────────────
    epoch_metrics_path = os.path.join(artifact_root, 'per_epoch_metrics.json')
    with open(epoch_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(epoch_metrics, f, indent=2)
    print(f'Per-epoch metrics saved: {epoch_metrics_path}')

    history_df   = pd.DataFrame(history)
    history_path = os.path.join(artifact_root, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f'Training history saved: {history_path}')

    # ── Reload best checkpoint ────────────────────────────────────────────────
    print(f'\nReloading best checkpoint (epoch {early_stop.best_epoch}, '
          f'Macro F1={early_stop.best_macro_f1:.4f}) ...')
    best_model = create_model('efficientnet_b0', pretrained=False, num_classes=4)
    best_model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
    best_model = best_model.to(DEVICE)
    criterion_eval = nn.CrossEntropyLoss(weight=compute_class_weights(TRAIN_CSV).to(DEVICE))

    val_loss_b, val_acc_b, macro_f1_b, all_preds_b, all_labels_b, report_b = \
        run_validation(best_model, val_loader, criterion_eval)
    cm_b = confusion_matrix(all_labels_b, all_preds_b, labels=list(range(4)))
    weighted_f1_b = report_b['weighted avg']['f1-score']

    n_total   = len(all_labels_b)
    n_correct = int((all_preds_b == all_labels_b).sum())

    print(f'\nBest checkpoint (epoch {early_stop.best_epoch}) final evaluation:')
    print(f'Val accuracy        : {val_acc_b:.4f} ({n_correct}/{n_total})')
    print(f'Macro F1            : {macro_f1_b:.4f}')
    print(f'Weighted F1         : {weighted_f1_b:.4f}')
    print('Confusion matrix:\n'
          + pd.DataFrame(cm_b, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

    # ── Classification report JSON ────────────────────────────────────────────
    cr_path = os.path.join(artifact_root, 'classification_report.json')
    with open(cr_path, 'w', encoding='utf-8') as f:
        json.dump(report_b, f, indent=2)

    # ── Config JSON ──────────────────────────────────────────────────────────
    best_hist_row = history_df[history_df['epoch'] == early_stop.best_epoch]
    train_acc_at_best = float(best_hist_row['train_acc'].iloc[0]) if len(best_hist_row) else float('nan')
    config_data = {
        'experiment':            f'{exp_id}_efficientnet_b0',
        'seed':                  seed,
        'config':                config,
        'augmentation_train':    (['RandomHorizontalFlip(p=0.2)', 'RandomRotation(15)',
                                   'pad_to_square', 'Resize(224,224)', 'ToTensor', 'Normalize(ImageNet)']
                                  if use_rotation else
                                  ['RandomHorizontalFlip(p=0.2)', 'pad_to_square',
                                   'Resize(224,224)', 'ToTensor', 'Normalize(ImageNet)']),
        'augmentation_val':      ['pad_to_square', 'Resize(224,224)', 'ToTensor', 'Normalize(ImageNet)'],
        'model':                 'efficientnet_b0 (timm, ImageNet pretrained)',
        'batch_size':            BATCH_SIZE,
        'max_epochs':            MAX_EPOCHS,
        'stage1_epochs':         STAGE1_EPOCHS,
        'lr_stage1':             LR_STAGE1,
        'lr_stage2':             LR_STAGE2,
        'optimizer':             'Adam',
        'scheduler':             'ReduceLROnPlateau(mode=min, factor=0.5, patience=2)',
        'early_stopping_metric': 'Macro F1 (maximize); tie-break: val_acc, val_loss',
        'early_stopping_patience': PATIENCE,
        'total_epochs_trained':  len(history),
        'best_epoch':            early_stop.best_epoch,
        'best_macro_f1':         early_stop.best_macro_f1,
        'best_val_acc':          early_stop.best_val_acc,
        'best_val_loss':         early_stop.best_val_loss,
        'best_weighted_f1':      weighted_f1_b,
        'train_acc_at_best':     train_acc_at_best,
        'test_set_accessed':     False,
    }
    config_path = os.path.join(artifact_root, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2)
    print(f'Config saved: {config_path}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_curves(history_df,
                         os.path.join(artifact_root, 'training_curves.png'))
    plot_macro_f1_curve(history_df,
                        os.path.join(artifact_root, 'macro_f1_curve.png'))
    plot_confusion_matrix(cm_b, CLASS_NAMES,
                          os.path.join(artifact_root, 'confusion_matrix.png'),
                          title=f'{exp_id.upper()} Seed={seed} Config={config} Confusion Matrix')

    # ── Per-image misclassification contact sheet ─────────────────────────────
    val_df = pd.read_csv(VAL_CSV, dtype=str)
    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    records = []
    for i, row in val_df.iterrows():
        true_idx = class_to_idx.get(str(row.get('label', row.get('class', ''))), -1)
        pred_idx = int(all_preds_b[i]) if i < len(all_preds_b) else -1
        if true_idx != pred_idx:
            img_id = row.get('image_id', row.get('filename', str(i)))
            img_path = str(row.get('filepath', row.get('image_path', '')))
            conf = float(torch.softmax(
                torch.zeros(4), dim=0
            )[pred_idx]) if pred_idx >= 0 else 0.0
            records.append({
                'image_id': img_id, 'image_path': img_path,
                'true_label': CLASS_NAMES[true_idx] if true_idx >= 0 else '?',
                'pred_label': CLASS_NAMES[pred_idx] if pred_idx >= 0 else '?',
                'confidence': conf,
            })

    if records:
        n_errors = len(records)
        n_cols = min(6, n_errors)
        n_rows = (n_errors + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(3 * n_cols, 3.5 * n_rows))
        axes = np.array(axes).reshape(-1) if n_errors > 1 else [axes]
        for ax, e in zip(axes, records):
            try:
                img = plt.imread(e['image_path'])
                ax.imshow(img)
            except Exception:
                ax.set_facecolor('#cccccc')
            ax.set_title(
                f"{e['image_id']}\nTrue: {e['true_label']}\n"
                f"Pred: {e['pred_label']}", fontsize=7)
            ax.axis('off')
        for ax in axes[n_errors:]:
            ax.axis('off')
        plt.suptitle(f'{exp_id.upper()} Seed={seed} Cfg={config} Misclassifications ({n_errors} images)',
                     fontsize=11)
        plt.tight_layout()
        cs_path = os.path.join(artifact_root, 'misclassification_contact_sheet.png')
        plt.savefig(cs_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f'Contact sheet saved: {cs_path}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n=== {exp_id.upper()} Complete ===')
    print(f'--- Checkpoint selection criterion: Macro F1 (maximize) ---')
    print(f'Best epoch                : {early_stop.best_epoch}')
    print(f'  Macro F1 at best epoch  : {macro_f1_b:.4f}')
    print(f'  val_acc  at best epoch  : {val_acc_b:.4f} ({n_correct}/{n_total})')
    print(f'  val_loss at best epoch  : {val_loss_b:.4f}')
    print(f'  Weighted F1             : {weighted_f1_b:.4f}')
    print(f'  Train-val gap           : {train_acc_at_best - val_acc_b:.4f}')
    print(f'--- Files ---')
    print(f'Per-epoch checkpoints     : {checkpoint_root}')
    print(f'Artifacts                 : {artifact_root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-seed reliability training')
    parser.add_argument('--exp_id', required=True,
                        help='Experiment ID, e.g. exp9 (determines artifact/checkpoint paths)')
    parser.add_argument('--seed',   required=True, type=int,
                        help='Random seed')
    parser.add_argument('--config', required=True, choices=['A', 'B'],
                        help='A = No rotation; B = RandomRotation(15deg)')
    args = parser.parse_args()
    train(exp_id=args.exp_id, seed=args.seed, config=args.config)
