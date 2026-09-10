"""
train_corrected_baseline.py â€” EfficientNet-B0 Corrected Baseline (Option A)

STRICT CORRECTED REPRODUCTION of the historical baseline.

INTENTIONAL PROTOCOL CORRECTION:
  1. Stage 1 freeze duration: epochs 1â€“5 frozen, epoch 6 onward unfrozen.
     The historical execution (task-398) froze only epochs 1â€“4 because the
     source at that time had STAGE1_EPOCHS = 4. This script uses STAGE1_EPOCHS = 5
     and the same 'if epoch == STAGE1_EPOCHS: unfreeze' loop structure, which
     guarantees the transition fires at the END of epoch 5.

OBSERVABILITY ADDITIONS (do not affect training or checkpoint selection):
  2. Per-epoch checkpoint saving: every epoch's weights saved to
     checkpoints/corrected_baseline/epoch_{N:02d}.pt
  3. Per-epoch metrics JSON: val_loss, val_acc, macro_f1, weighted_f1,
     per-class F1, and confusion matrix recorded every epoch.
     This allows retrospective identification of both:
       - the epoch selected by minimum val_loss (checkpoint criterion)
       - the epoch with highest Macro F1 (future experiment criterion)
  4. Macro F1 and weighted F1 added to training_history.csv columns.

ENVIRONMENT CHANGES (documented; do not affect model protocol):
  5. Device: CUDA/RTX 5050 (historical baseline ran on CPU).
     GPU vs CPU floating-point arithmetic may cause minor numerical differences
     even with the same seed. This is an inherent hardware difference.
  6. pin_memory=False (historical: True â€” caused DataLoader deadlock on
     this Windows + CUDA environment with num_workers=0).
  7. num_workers=0 (historical: 2 â€” simplified for environment compatibility).

UNCHANGED FROM HISTORICAL BASELINE:
  - Checkpoint-selection criterion: val_loss (minimize) â€” IDENTICAL to historical
  - Early stopping: based on val_loss â€” IDENTICAL to historical
  - Dataset splits: same TRAIN_CSV / VAL_CSV
  - Seed: 42
  - Model: timm EfficientNet-B0, ImageNet pretrained, num_classes=4
  - drop_rate: timm default (0.0) â€” not set
  - Loss: CrossEntropyLoss(weight=class_weights), no label smoothing
  - Optimizer: Adam, LR Stage1=1e-3, LR Stage2=1e-4
  - Scheduler: ReduceLROnPlateau(mode=min, factor=0.5, patience=2) on val_loss
  - Early stopping patience: 5
  - Augmentation: RandomHorizontalFlip(p=0.2) training only
  - Preprocessing: pad-to-square (zero-pad) â†’ Resize(224x224) â†’ Normalize(ImageNet)
  - Batch size: 16
  - Max epochs: 20
  - Class weights: inverse frequency from train split
  - Test set: NOT LOADED OR EVALUATED
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # prevent cp1252 crash on Windows
import json
import platform
import math

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
from PIL import Image

from training.dataset import EcopinDataset
from training.utils import compute_class_weights
from data_config.dataset_paths_and_split_config import (
    TRAIN_CSV, VAL_CSV, SEED, RAW_DIR, PROCESSED_DIR
)

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ARTIFACT_ROOT   = os.path.join(PROJECT_ROOT, 'artifacts', 'corrected_baseline_efficientnet_b0')
CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, 'checkpoints', 'corrected_baseline')
BEST_CKPT_PATH  = os.path.join(CHECKPOINT_ROOT, 'best.pt')
os.makedirs(ARTIFACT_ROOT,   exist_ok=True)
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

# â”€â”€ Hyperparameters (identical to historical baseline except where noted) â”€â”€â”€â”€â”€â”€
SEED            = 42
MAX_EPOCHS      = 20
STAGE1_EPOCHS   = 5          # CHANGE: corrected from 4 (actual) to 5 (intended)
BATCH_SIZE      = 16
LR_STAGE1       = 1e-3
LR_STAGE2       = 1e-4
PATIENCE        = 5
CLASS_NAMES     = ['flooding', 'non_environmental', 'pollution', 'waste']

# â”€â”€ Reproducibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
torch.manual_seed(SEED)
np.random.seed(SEED)

# â”€â”€ Device â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device              : {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU name            : {torch.cuda.get_device_name(0)}')
    print(f'GPU memory          : {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB')


# â”€â”€ Preprocessing (IDENTICAL to historical baseline) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _pad_to_square(img):
    w, h = img.size
    max_side   = max(w, h)
    pad_left   = (max_side - w) // 2
    pad_top    = (max_side - h) // 2
    pad_right  = max_side - w - pad_left
    pad_bottom = max_side - h - pad_top
    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)


def get_transform(train: bool = False):
    """Identical to historical baseline augmentation pipeline.
    Training: RandomHorizontalFlip(p=0.2) only.
    Validation: deterministic pad-to-square -> Resize(224) -> ToTensor -> Normalize.
    """
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(p=0.2))  # UNCHANGED
    transforms.extend([
        T.Lambda(_pad_to_square),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return T.Compose(transforms)


# â”€â”€ Early Stopping â€” val_loss (IDENTICAL to historical baseline protocol) â”€â”€â”€â”€â”€â”€
class EarlyStopping:
    """Monitors val_loss (minimize) â€” identical criterion to historical baseline.
    Per-epoch checkpoint saving is handled in the main training loop,
    separately from best-checkpoint selection, so observability additions
    do not affect which model is selected.
    """
    def __init__(self, patience: int = 5, best_ckpt_path: str = 'best.pt'):
        self.patience       = patience
        self.counter        = 0
        self.best_score     = None   # stored as -val_loss (higher is better)
        self.best_val_loss  = None
        self.best_val_acc   = None
        self.best_macro_f1  = None
        self.best_epoch     = None
        self.early_stop     = False
        self.best_ckpt_path = best_ckpt_path

    def __call__(self, epoch: int, val_loss: float, val_acc: float,
                 macro_f1: float, model):
        score = -val_loss  # higher score = lower val_loss = better
        if self.best_score is None or score > self.best_score:
            self.best_score    = score
            self.best_val_loss = val_loss
            self.best_val_acc  = val_acc
            self.best_macro_f1 = macro_f1
            self.best_epoch    = epoch
            torch.save(model.state_dict(), self.best_ckpt_path)
            self.counter = 0
            print(f'  [EarlyStopping] New best val_loss={val_loss:.4f} '
                  f'(val_acc={val_acc:.4f}, macro_f1={macro_f1:.4f}) '
                  f'at epoch {epoch} -> best checkpoint saved')
        else:
            self.counter += 1
            print(f'  [EarlyStopping] No improvement. Counter: {self.counter}/{self.patience} '
                  f'(best val_loss={self.best_val_loss:.4f} at epoch {self.best_epoch})')
            if self.counter >= self.patience:
                self.early_stop = True


# â”€â”€ Per-epoch validation helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_validation(model, val_loader, criterion):
    """Run full validation pass.
    criterion is passed explicitly â€” no global state dependency.
    Returns: val_loss, val_acc, macro_f1, all_preds, all_labels, per-class report.
    """
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


# â”€â”€ Plots â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def plot_training_curves(history_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (y1, y2, title, ylabel) in zip(axes, [
        ('train_loss', 'val_loss', 'Loss Curves', 'Loss'),
        ('train_acc',  'val_acc',  'Accuracy Curves', 'Accuracy'),
    ]):
        ax.plot(history_df['epoch'], history_df[y1], label='Train')
        ax.plot(history_df['epoch'], history_df[y2], label='Val')
        ax.axvline(x=5.5, color='grey', linestyle='--', linewidth=0.8, label='Stage 1â†’2')
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
    ax.axvline(x=5.5, color='grey', linestyle='--', linewidth=0.8, label='Stage 1â†’2')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Macro F1')
    ax.set_title('Validation Macro F1 Curve')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_confusion_matrix(cm, class_names, output_path, title='Corrected Baseline â€” Validation Confusion Matrix'):
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


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def train():

    # â”€â”€ Data loaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    train_dataset = EcopinDataset(TRAIN_CSV, transform=get_transform(train=True))
    val_dataset   = EcopinDataset(VAL_CSV,   transform=get_transform(train=False))
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=0, pin_memory=False)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0, pin_memory=False)
    print(f'Train samples       : {len(train_dataset)}')
    print(f'Val samples         : {len(val_dataset)}')

    # â”€â”€ Class weights (UNCHANGED) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    class_weights = compute_class_weights(TRAIN_CSV).to(DEVICE)
    print('Class weights       :', {c: round(float(w), 4)
                                    for c, w in zip(CLASS_NAMES, class_weights.cpu())})

    # â”€â”€ Loss (UNCHANGED: CrossEntropyLoss, class-weighted, no label smoothing) â”€
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # â”€â”€ Model (UNCHANGED: timm EfficientNet-B0, ImageNet pretrained, 4 classes)
    model = create_model('efficientnet_b0', pretrained=True, num_classes=4)
    model = model.to(DEVICE)
    print(f'Model params device : {next(model.parameters()).device}')
    print(f'Classifier          : {model.classifier}')

    # â”€â”€ Stage 1: freeze backbone (UNCHANGED logic; duration corrected to 5) â”€â”€â”€â”€
    # epochs 1-5: backbone frozen (classifier-only training)
    # epoch 6+  : backbone unfrozen (full fine-tuning)
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f'Stage 1: backbone frozen ({frozen_count} tensors frozen)')
    print(f'STAGE1_EPOCHS={STAGE1_EPOCHS}: backbone frozen for epochs 1-{STAGE1_EPOCHS}, '
          f'unfrozen from epoch {STAGE1_EPOCHS+1} onward.')

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_STAGE1
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    # Checkpoint selection: val_loss (minimize) â€” identical to historical baseline
    early_stop = EarlyStopping(
        patience=PATIENCE,
        best_ckpt_path=BEST_CKPT_PATH,
    )

    history    = []
    epoch_metrics = {}   # epoch -> full per-class report + cm

    for epoch in range(1, MAX_EPOCHS + 1):
        stage = 1 if epoch <= STAGE1_EPOCHS else 2

        # â”€â”€ Training pass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

        # â”€â”€ Validation pass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        val_loss, val_acc, macro_f1, all_preds, all_labels, report = \
            run_validation(model, val_loader, criterion)

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(4)))
        current_lr = optimizer.param_groups[0]['lr']

        # â”€â”€ Record history â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        history.append({
            'epoch':       epoch,
            'stage':       stage,
            'train_loss':  train_loss,
            'train_acc':   train_acc,
            'val_loss':    val_loss,
            'val_acc':     val_acc,
            'macro_f1':    macro_f1,
            'weighted_f1': report['weighted avg']['f1-score'],
            'learning_rate': current_lr,
        })

        # â”€â”€ Save per-epoch checkpoint (observability â€” does not affect selection)
        epoch_ckpt_path = os.path.join(CHECKPOINT_ROOT, f'epoch_{epoch:02d}.pt')
        torch.save(model.state_dict(), epoch_ckpt_path)

        # â”€â”€ Save per-epoch metrics (observability) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        epoch_metrics[epoch] = {
            'val_loss':    val_loss,
            'val_acc':     val_acc,
            'macro_f1':    macro_f1,
            'weighted_f1': report['weighted avg']['f1-score'],
            'per_class':   {c: {k: report[c][k] for k in
                                ['precision', 'recall', 'f1-score', 'support']}
                            for c in CLASS_NAMES},
            'confusion_matrix': cm.tolist(),
        }

        print(f'Epoch {epoch:02d} [S{stage}] | '
              f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} | '
              f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} | '
              f'macro_f1={macro_f1:.4f} | lr={current_lr:.6f}')

        # â”€â”€ Scheduler step: monitors val_loss (IDENTICAL to historical) â”€â”€â”€â”€â”€â”€â”€â”€
        scheduler.step(val_loss)

        # â”€â”€ Early stopping: val_loss criterion (IDENTICAL to historical) â”€â”€â”€â”€â”€â”€â”€
        early_stop(epoch, val_loss, val_acc, macro_f1, model)
        if early_stop.early_stop:
            print(f'Early stopping triggered at epoch {epoch}')
            break

        # â”€â”€ Stage transition (fires at END of epoch STAGE1_EPOCHS) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Epoch 6 is the FIRST epoch with backbone unfrozen.
        if epoch == STAGE1_EPOCHS:
            print(f'--- Transition to Stage 2 after epoch {epoch}: '
                  f'unfreeze backbone, LR={LR_STAGE2}')
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=LR_STAGE2)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2
            )

    # â”€â”€ Save per-epoch metrics JSON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    epoch_metrics_path = os.path.join(ARTIFACT_ROOT, 'per_epoch_metrics.json')
    with open(epoch_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(epoch_metrics, f, indent=2)
    print(f'Per-epoch metrics saved: {epoch_metrics_path}')

    # â”€â”€ Save training history CSV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    history_df   = pd.DataFrame(history)
    history_path = os.path.join(ARTIFACT_ROOT, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f'Training history saved: {history_path}')

    # â”€â”€ Reload best checkpoint for final evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Best checkpoint = epoch with minimum val_loss (identical to historical)
    print(f'\nReloading best checkpoint (epoch {early_stop.best_epoch}, '
          f'selected by val_loss={early_stop.best_val_loss:.4f}) ...')
    best_model = create_model('efficientnet_b0', pretrained=False, num_classes=4)
    best_model.load_state_dict(torch.load(BEST_CKPT_PATH, map_location=DEVICE))
    best_model = best_model.to(DEVICE)
    criterion_eval = nn.CrossEntropyLoss(weight=compute_class_weights(TRAIN_CSV).to(DEVICE))

    val_loss_b, val_acc_b, macro_f1_b, all_preds_b, all_labels_b, report_b = \
        run_validation(best_model, val_loader, criterion_eval)
    cm_b = confusion_matrix(all_labels_b, all_preds_b, labels=list(range(4)))

    n_total   = len(all_labels_b)
    n_correct = int((all_preds_b == all_labels_b).sum())

    print(f'\nBest checkpoint (epoch {early_stop.best_epoch}) final evaluation:')
    print(f'Val accuracy        : {val_acc_b:.4f} ({n_correct}/{n_total})')
    print(f'Macro F1            : {macro_f1_b:.4f}')
    print(f'Weighted F1         : {report_b["weighted avg"]["f1-score"]:.4f}')
    print(f'Confusion matrix:\n'
          + pd.DataFrame(cm_b, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

    # â”€â”€ Per-image predictions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    val_df = pd.read_csv(VAL_CSV, dtype=str)
    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    records = []
    best_model.eval()
    with torch.no_grad():
        for _, row in val_df.iterrows():
            rel    = row['relative_image_path']
            prefix, sub = rel.split('/', 1)
            base   = RAW_DIR if prefix == 'raw' else PROCESSED_DIR
            img_path = os.path.join(base, sub)
            img    = Image.open(img_path).convert('RGB')
            tensor = get_transform(train=False)(img).unsqueeze(0).to(DEVICE)
            out    = best_model(tensor)
            probs  = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()
            pred   = int(np.argmax(probs))
            true   = class_to_idx[row['primary_label']]
            records.append({
                'image_id':   row.get('image_id', os.path.basename(img_path)),
                'image_path': img_path,
                'true_label': CLASS_NAMES[true],
                'pred_label': CLASS_NAMES[pred],
                'confidence': round(float(probs[pred]), 4),
                'correct':    bool(true == pred),
                'probs':      {c: round(float(probs[j]), 4)
                               for j, c in enumerate(CLASS_NAMES)},
            })

    pred_path = os.path.join(ARTIFACT_ROOT, 'val_predictions.json')
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    errors   = [r for r in records if not r['correct']]
    n_errors = len(errors)

    # â”€â”€ Plots â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    plot_training_curves(history_df, os.path.join(ARTIFACT_ROOT, 'training_curves.png'))
    plot_macro_f1_curve(history_df,  os.path.join(ARTIFACT_ROOT, 'macro_f1_curve.png'))
    plot_confusion_matrix(cm_b, CLASS_NAMES,
                          os.path.join(ARTIFACT_ROOT, 'validation_confusion_matrix.png'))

    # â”€â”€ Classification report JSON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cr_path = os.path.join(ARTIFACT_ROOT, 'classification_report.json')
    with open(cr_path, 'w', encoding='utf-8') as f:
        json.dump(report_b, f, indent=2)

    # â”€â”€ Config JSON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    best_hist_row = history_df[history_df['epoch'] == early_stop.best_epoch]
    train_acc_at_best = float(best_hist_row['train_acc'].iloc[0]) if len(best_hist_row) else float('nan')
    config = {
        'experiment':              'corrected_baseline_efficientnet_b0',
        'python_version':          platform.python_version(),
        'torch_version':           torch.__version__,
        'torchvision_version':     __import__('torchvision').__version__,
        'cuda_version':            torch.version.cuda,
        'gpu_name':                torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu',
        'device':                  str(DEVICE),
        'seed':                    SEED,
        'model':                   'efficientnet_b0',
        'pretrained':              'ImageNet-1k via timm',
        'num_classes':             4,
        'class_names':             CLASS_NAMES,
        'drop_rate':               'timm default (0.0)',
        'optimizer':               'Adam',
        'weight_decay':            0,
        'label_smoothing':         0,
        'class_weights':           {c: round(float(w), 6) for c, w in
                                    zip(CLASS_NAMES, compute_class_weights(TRAIN_CSV))},
        'stage1_epochs':           STAGE1_EPOCHS,
        'stage1_lr':               LR_STAGE1,
        'stage2_lr':               LR_STAGE2,
        'scheduler':               'ReduceLROnPlateau(mode=min, factor=0.5, patience=2)',
        'early_stopping_patience': PATIENCE,
        'early_stopping_metric':   'val_loss (minimize) â€” identical to historical baseline',
        'checkpoint_note':         'per-epoch checkpoints saved for retrospective Macro F1 analysis',
        'batch_size':              BATCH_SIZE,
        'max_epochs':              MAX_EPOCHS,
        'total_epochs_trained':    len(history),
        'best_epoch':              early_stop.best_epoch,
        'best_macro_f1':           early_stop.best_macro_f1,
        'best_val_acc':            early_stop.best_val_acc,
        'best_val_loss':           early_stop.best_val_loss,
        'train_acc_at_best_epoch': train_acc_at_best,
        'augmentation_train':      ['RandomHorizontalFlip(p=0.2)', 'pad_to_square',
                                    'Resize(224,224)', 'ToTensor', 'Normalize(ImageNet)'],
        'augmentation_val':        ['pad_to_square', 'Resize(224,224)',
                                    'ToTensor', 'Normalize(ImageNet)'],
        'changes_vs_historical_baseline': [
            '[PROTOCOL] Stage 1 freeze: 5 epochs (historical: 4 actual epochs)',
            '[OBSERVABILITY] Per-epoch checkpoint saving added (epoch_01.pt ... epoch_NN.pt)',
            '[OBSERVABILITY] Per-epoch metrics JSON added (val_loss, macro_f1, per-class F1, cm)',
            '[OBSERVABILITY] Macro F1 and weighted F1 added to training_history.csv',
            '[ENVIRONMENT] Device: CUDA RTX5050 (historical: CPU)',
            '[ENVIRONMENT] pin_memory=False (historical: True â€” Windows/CUDA deadlock fix)',
            '[ENVIRONMENT] num_workers=0 (historical: 2)',
        ],
        'unchanged_vs_historical_baseline': [
            'Model: EfficientNet-B0, ImageNet pretrained',
            'Augmentation: RandomHorizontalFlip(p=0.2) only',
            'Loss: CrossEntropyLoss(class_weights), no label_smoothing',
            'Optimizer: Adam, LR 1e-3 / 1e-4',
            'Scheduler: ReduceLROnPlateau(mode=min, factor=0.5, patience=2)',
            'Batch size: 16', 'Max epochs: 20', 'ES patience: 5', 'Seed: 42',
            'Preprocessing: pad_to_square -> Resize(224)',
        ],
    }
    config_path = os.path.join(ARTIFACT_ROOT, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f'Config saved: {config_path}')

    # â”€â”€ Contact sheet (misclassifications) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if n_errors > 0:
        cols = 4
        rows = math.ceil(n_errors / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        axes = np.array(axes).reshape(-1)
        for ax in axes: ax.axis('off')
        for k, e in enumerate(sorted(errors, key=lambda x: (x['true_label'], x['image_id']))):
            ax = axes[k]
            try:
                img = Image.open(e['image_path']).convert('RGB')
                ax.imshow(img)
            except Exception as ex:
                ax.text(0.5, 0.5, f'Error:\n{ex}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=7)
            ax.set_title(
                f"{e['image_id']}\nTrue: {e['true_label']}\n"
                f"Pred: {e['pred_label']} ({e['confidence']:.2f})", fontsize=7)
            ax.axis('off')
        plt.suptitle(f'Corrected Baseline Misclassifications ({n_errors} images)', fontsize=11)
        plt.tight_layout()
        cs_path = os.path.join(ARTIFACT_ROOT, 'misclassification_contact_sheet.png')
        plt.savefig(cs_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f'Contact sheet saved: {cs_path}')

    # â”€â”€ Validation report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    off = sorted([(cm_b[i,j], CLASS_NAMES[i], CLASS_NAMES[j])
                  for i in range(4) for j in range(4) if i != j], reverse=True)

    report_md = f"""# EfficientNet-B0 Corrected Baseline â€” Validation Report

**Experiment:** `corrected_baseline_efficientnet_b0`
**Device:** {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})
**torch:** {torch.__version__}  **CUDA:** {torch.version.cuda}
**Batch size:** {BATCH_SIZE}  **Seed:** {SEED}
**Total epochs run:** {len(history)}
**Best checkpoint epoch:** {early_stop.best_epoch} (selected by Macro F1)
**Best Macro F1:** {early_stop.best_macro_f1:.4f}
**Best val accuracy:** {early_stop.best_val_acc:.4f}
**Best val loss:** {early_stop.best_val_loss:.4f}

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

## Validation Metrics (best checkpoint â€” epoch {early_stop.best_epoch})

| Metric | Value |
|--------|-------|
| Accuracy | {val_acc_b:.4f} ({n_correct}/{n_total}) |
| Macro F1 | {macro_f1_b:.4f} |
| Weighted F1 | {report_b['weighted avg']['f1-score']:.4f} |

## Per-Class Results

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
"""
    for cls in CLASS_NAMES:
        r = report_b[cls]
        report_md += (f"| {cls} | {r['precision']:.4f} | {r['recall']:.4f} "
                      f"| {r['f1-score']:.4f} | {int(r['support'])} |\n")

    report_md += f"""
## Confusion Matrix (rows=True, cols=Predicted)

| True \\ Pred | {' | '.join(CLASS_NAMES)} |
|-------------|{'|'.join(['------']*4)}|
"""
    for i, cls in enumerate(CLASS_NAMES):
        vals = ' | '.join(str(cm_b[i, j]) for j in range(4))
        report_md += f"| {cls} | {vals} |\n"

    report_md += "\n## Most Common Confusions\n\n| True | Predicted | Count |\n|------|-----------|-------|\n"
    for cnt, tc, pc in off[:6]:
        if cnt > 0:
            report_md += f"| {tc} | {pc} | {cnt} |\n"

    report_md += f"""
## Overfitting Indicator

| | Value |
|--|-------|
| Train accuracy at best epoch | {train_acc_at_best:.4f} |
| Val accuracy at best epoch | {val_acc_b:.4f} |
| Trainâˆ’Val gap | {train_acc_at_best - val_acc_b:.4f} |

## Misclassifications ({n_errors} / {n_total})

| Image ID | True | Predicted | Confidence |
|----------|------|-----------|------------|
"""
    for e in sorted(errors, key=lambda x: x['true_label']):
        report_md += f"| {e['image_id']} | {e['true_label']} | {e['pred_label']} | {e['confidence']:.3f} |\n"

    report_md += """
## Interpretation Notes

- Validation set is 53 images. A 1-image change â‰ˆ 1.9 pp accuracy swing.
- Best epoch selected by **val_loss (minimize)** â€” identical to historical baseline protocol.
- Per-epoch checkpoints saved for retrospective analysis: identify minimum val_loss epoch
  (selected checkpoint) and maximum Macro F1 epoch separately from the saved files.
- Test set was NOT evaluated.

All artifacts: `c:/dev/ecopin_image_validation/artifacts/corrected_baseline_efficientnet_b0/`
"""
    report_path = os.path.join(ARTIFACT_ROOT, 'corrected_baseline_validation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f'Validation report saved: {report_path}')

    # Identify Macro F1 best epoch from saved per-epoch metrics (retrospective)
    best_macro_f1_epoch = max(epoch_metrics.keys(), key=lambda e: epoch_metrics[e]['macro_f1'])
    best_macro_f1_val   = epoch_metrics[best_macro_f1_epoch]['macro_f1']

    print('\n=== Corrected Baseline Complete ===')
    print(f'--- Checkpoint selection criterion: val_loss (minimize) ---')
    print(f'Best epoch (by val_loss)  : {early_stop.best_epoch}')
    print(f'  val_loss at best epoch  : {early_stop.best_val_loss:.4f}')
    print(f'  val_acc  at best epoch  : {val_acc_b:.4f} ({n_correct}/{n_total})')
    print(f'  Macro F1 at best epoch  : {macro_f1_b:.4f}')
    print(f'  Weighted F1             : {report_b["weighted avg"]["f1-score"]:.4f}')
    print(f'  Train-val gap           : {train_acc_at_best - val_acc_b:.4f}')
    print(f'--- Retrospective Macro F1 best (observability) ---')
    print(f'Best epoch (by Macro F1)  : {best_macro_f1_epoch}')
    print(f'  Macro F1 at that epoch  : {best_macro_f1_val:.4f}')
    print(f'  val_acc  at that epoch  : {epoch_metrics[best_macro_f1_epoch]["val_acc"]:.4f}')
    print(f'  val_loss at that epoch  : {epoch_metrics[best_macro_f1_epoch]["val_loss"]:.4f}')
    print(f'--- Files ---')
    print(f'Per-epoch checkpoints     : {CHECKPOINT_ROOT}')


if __name__ == '__main__':
    train()

