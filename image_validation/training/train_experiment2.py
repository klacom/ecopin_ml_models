"""
train_experiment2.py â€” EfficientNet-B0 Experiment 2
Controlled experiment vs baseline:
  - Stage 1: freeze backbone for exactly 5 complete epochs (corrected from baseline's 4)
  - weight_decay = 1e-4 on Adam
  - label_smoothing = 0.1 on CrossEntropyLoss
  - drop_rate = 0.2 (timm param; EfficientNet-B0 default is 0.0, no duplication)
  - Training augmentation: H-flip p=0.5, RandomRotationÂ±15Â°, ColorJitter
  - CUDA training (RTX 5050)
  - Test set is NOT loaded or evaluated.
"""

import os
import sys
import json
import platform

import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from timm import create_model
from sklearn.metrics import confusion_matrix, classification_report
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from training.dataset import EcopinDataset
from training.utils import compute_class_weights, generate_classification_report

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from data_config.dataset_paths_and_split_config import TRAIN_CSV, VAL_CSV, SEED

ARTIFACT_ROOT   = os.path.join(PROJECT_ROOT, 'artifacts', 'exp2_efficientnet_b0')
CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, 'checkpoints')
CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, 'efficientnet_b0_exp2_best.pt')
os.makedirs(ARTIFACT_ROOT,   exist_ok=True)
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

# â”€â”€ Hyperparameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEED           = 42
MAX_EPOCHS     = 20
STAGE1_EPOCHS  = 5          # backbone frozen for epochs 1..5 inclusive
BATCH_SIZE     = 16
LR_STAGE1      = 1e-3
LR_STAGE2      = 1e-4
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTHING= 0.1
DROP_RATE      = 0.2        # passed to timm; default is 0.0 â€” no duplication
PATIENCE       = 5

# â”€â”€ Reproducibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
torch.manual_seed(SEED)
np.random.seed(SEED)

# â”€â”€ Device â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device              : {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU name            : {torch.cuda.get_device_name(0)}')
    print(f'GPU memory          : {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB')

# â”€â”€ Class names â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CLASS_NAMES = ['flooding', 'non_environmental', 'pollution', 'waste']

# â”€â”€ Preprocessing helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _pad_to_square(img):
    """Zero-pad shorter dimension to preserve aspect ratio."""
    w, h = img.size
    max_side  = max(w, h)
    pad_left  = (max_side - w) // 2
    pad_top   = (max_side - h) // 2
    pad_right = max_side - w - pad_left
    pad_bottom= max_side - h - pad_top
    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)


def get_transform(train: bool = False):
    """
    Training pipeline (Experiment 2):
        RandomHorizontalFlip(p=0.5)
        RandomRotation(Â±15Â°)
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0)
        pad-to-square â†’ Resize(224) â†’ ToTensor â†’ Normalize(ImageNet)

    Validation pipeline: pad-to-square â†’ Resize(224) â†’ ToTensor â†’ Normalize
    Augmentations are applied before pad-to-square so they operate on the
    natural (unpadded) image.
    """
    if train:
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0),
            T.Lambda(_pad_to_square),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Lambda(_pad_to_square),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# â”€â”€ Early stopping â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class EarlyStopping:
    def __init__(self, patience: int = 5, checkpoint_path: str = 'best.pt'):
        self.patience        = patience
        self.counter         = 0
        self.best_score      = None
        self.best_val_loss   = None
        self.best_epoch      = None
        self.early_stop      = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, epoch: int, val_loss: float, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score:
            self.best_score    = score
            self.best_val_loss = val_loss
            self.best_epoch    = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
            self.counter = 0
            print(f'  [EarlyStopping] New best val_loss={val_loss:.4f} at epoch {epoch} â†’ checkpoint saved')
        else:
            self.counter += 1
            print(f'  [EarlyStopping] No improvement. Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


# â”€â”€ Plotting helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def plot_training_curves(history_df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (y1, y2, title, ylabel) in zip(axes, [
        ('train_loss', 'val_loss', 'Loss Curves', 'Loss'),
        ('train_accuracy', 'val_accuracy', 'Accuracy Curves', 'Accuracy'),
    ]):
        ax.plot(history_df['epoch'], history_df[y1], label='Train')
        ax.plot(history_df['epoch'], history_df[y2], label='Val')
        # Mark stage boundary
        ax.axvline(x=5.5, color='grey', linestyle='--', linewidth=0.8, label='Stage 1â†’2')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f'Training curves saved: {output_path}')


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Exp2 Validation Confusion Matrix')
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
    print(f'Confusion matrix saved: {output_path}')


# â”€â”€ Main training function â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def train():
    # â”€â”€ Data loaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    train_dataset = EcopinDataset(TRAIN_CSV, transform=get_transform(train=True))
    val_dataset   = EcopinDataset(VAL_CSV,   transform=get_transform(train=False))
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=False)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0, pin_memory=False)
    print(f'Train samples       : {len(train_dataset)}')
    print(f'Val samples         : {len(val_dataset)}')

    # â”€â”€ Class weights â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    class_weights = compute_class_weights(TRAIN_CSV).to(DEVICE)
    print('Class weights       :', {c: round(float(w), 4)
                                    for c, w in zip(CLASS_NAMES, class_weights.cpu())})

    # â”€â”€ Loss â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    # â”€â”€ Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # drop_rate=0.2 adds dropout before the classifier head.
    # EfficientNet-B0 timm default drop_rate is 0.0 â€” no duplication.
    model = create_model('efficientnet_b0', pretrained=True,
                         num_classes=4, drop_rate=DROP_RATE)
    model = model.to(DEVICE)
    print(f'Model params device : {next(model.parameters()).device}')
    print(f'drop_rate           : {model.drop_rate}')
    print(f'classifier          : {model.classifier}')

    # â”€â”€ Stage 1: freeze backbone â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f'Stage 1: backbone frozen ({frozen_count} parameter tensors frozen)')

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_STAGE1, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    early_stop = EarlyStopping(patience=PATIENCE, checkpoint_path=CHECKPOINT_PATH)

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        stage = 1 if epoch <= STAGE1_EPOCHS else 2

        # â”€â”€ Train â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        model.train()
        train_loss_sum, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)
        train_loss = train_loss_sum / total
        train_acc  = correct / total

        # â”€â”€ Validate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
                val_loss_sum += loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)
        val_loss = val_loss_sum / val_total
        val_acc  = val_correct  / val_total
        current_lr = optimizer.param_groups[0]['lr']

        history.append({
            'epoch':        epoch,
            'stage':        stage,
            'train_loss':   train_loss,
            'train_accuracy': train_acc,
            'val_loss':     val_loss,
            'val_accuracy': val_acc,
            'learning_rate': current_lr,
        })

        print(f'Epoch {epoch:02d} [S{stage}] | '
              f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} | '
              f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} | '
              f'lr={current_lr:.6f}')

        scheduler.step(val_loss)
        early_stop(epoch, val_loss, model)
        if early_stop.early_stop:
            print(f'Early stopping triggered at epoch {epoch}')
            break

        # â”€â”€ Stage transition (fires at END of epoch STAGE1_EPOCHS) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # The backbone is unfrozen AFTER epoch 5's training+validation+ES check.
        # Epoch 6 will be the first epoch with backbone unfrozen.
        if epoch == STAGE1_EPOCHS:
            print(f'--- Transition to Stage 2 after epoch {epoch}: unfreeze backbone, LR={LR_STAGE2}, weight_decay={WEIGHT_DECAY}')
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(
                model.parameters(), lr=LR_STAGE2, weight_decay=WEIGHT_DECAY
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2
            )

    # â”€â”€ Save training history â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    history_df = pd.DataFrame(history)
    history_path = os.path.join(ARTIFACT_ROOT, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f'Training history saved: {history_path}')

    # â”€â”€ Save config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    config = {
        'experiment':      'exp2_efficientnet_b0',
        'python_version':  platform.python_version(),
        'torch_version':   torch.__version__,
        'torchvision_version': __import__('torchvision').__version__,
        'cuda_version':    torch.version.cuda,
        'gpu_name':        torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu',
        'device':          str(DEVICE),
        'seed':            SEED,
        'model':           'efficientnet_b0',
        'pretrained':      'ImageNet-1k via timm',
        'num_classes':     4,
        'class_names':     CLASS_NAMES,
        'drop_rate':       DROP_RATE,
        'optimizer':       'Adam',
        'weight_decay':    WEIGHT_DECAY,
        'label_smoothing': LABEL_SMOOTHING,
        'class_weights':   {c: round(float(w), 6) for c, w in
                            zip(CLASS_NAMES, compute_class_weights(TRAIN_CSV))},
        'stage1_epochs':   STAGE1_EPOCHS,
        'stage1_lr':       LR_STAGE1,
        'stage2_lr':       LR_STAGE2,
        'scheduler':       'ReduceLROnPlateau(mode=min, factor=0.5, patience=2)',
        'early_stopping_patience': PATIENCE,
        'batch_size':      BATCH_SIZE,
        'max_epochs':      MAX_EPOCHS,
        'total_epochs_trained': len(history),
        'best_epoch':      early_stop.best_epoch,
        'best_val_loss':   early_stop.best_val_loss,
        'augmentation_train': [
            'RandomHorizontalFlip(p=0.5)',
            'RandomRotation(degrees=15)',
            'ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0)',
            'pad_to_square (zero-pad)',
            'Resize(224, 224)',
            'ToTensor',
            'Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])',
        ],
        'augmentation_val': [
            'pad_to_square (zero-pad)',
            'Resize(224, 224)',
            'ToTensor',
            'Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])',
        ],
        'stage_boundary_note': (
            f'Backbone frozen for epochs 1-{STAGE1_EPOCHS}. '
            f'Transition fires at end of epoch {STAGE1_EPOCHS}. '
            f'Epoch {STAGE1_EPOCHS+1} onward: backbone unfrozen.'
        ),
    }
    config_path = os.path.join(ARTIFACT_ROOT, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'Config saved: {config_path}')

    # â”€â”€ Plot training curves â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    plot_training_curves(history_df, os.path.join(ARTIFACT_ROOT, 'training_curves.png'))

    # â”€â”€ Reload best checkpoint for final evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f'\\nReloading best checkpoint from epoch {early_stop.best_epoch} ...')
    best_model = create_model('efficientnet_b0', pretrained=False,
                              num_classes=4, drop_rate=DROP_RATE)
    best_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    best_model = best_model.to(DEVICE)
    best_model.eval()

    # â”€â”€ Validation inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    all_preds, all_labels, all_probs, all_image_ids, all_img_paths = [], [], [], [], []
    # Re-use a plain (no label_smoothing) eval loader
    eval_val_dataset = EcopinDataset(VAL_CSV, transform=get_transform(train=False))
    eval_val_loader  = DataLoader(eval_val_dataset, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0)

    # We need image_id and path for per-image records â€” use a custom loop
    val_df = pd.read_csv(VAL_CSV, dtype=str)
    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

    from PIL import Image
    import os as _os
    from data_config.dataset_paths_and_split_config import RAW_DIR, PROCESSED_DIR

    with torch.no_grad():
        for _, row in val_df.iterrows():
            rel = row['relative_image_path']
            prefix, sub = rel.split('/', 1)
            base = RAW_DIR if prefix == 'raw' else PROCESSED_DIR
            img_path = _os.path.join(base, sub)
            img = Image.open(img_path).convert('RGB')
            tensor = get_transform(train=False)(img).unsqueeze(0).to(DEVICE)
            out    = best_model(tensor)
            probs  = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()
            pred   = int(np.argmax(probs))
            true   = class_to_idx[row['primary_label']]
            all_preds.append(pred)
            all_labels.append(true)
            all_probs.append(probs)
            all_image_ids.append(row.get('image_id', _os.path.basename(img_path)))
            all_img_paths.append(img_path)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    n_total    = len(all_labels)
    n_correct  = int((all_preds == all_labels).sum())
    accuracy   = n_correct / n_total

    print(f'\\nVal samples         : {n_total}')
    print(f'Correct predictions : {n_correct}')
    print(f'Accuracy            : {accuracy:.4f}')

    # â”€â”€ Confusion matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(4)))
    print('\\nConfusion matrix (rows=True, cols=Predicted):')
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())
    plot_confusion_matrix(cm, CLASS_NAMES,
                          os.path.join(ARTIFACT_ROOT, 'validation_confusion_matrix.png'))

    # â”€â”€ Classification report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    report = classification_report(all_labels, all_preds,
                                   target_names=CLASS_NAMES, output_dict=True)
    cr_path = os.path.join(ARTIFACT_ROOT, 'classification_report.json')
    with open(cr_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Classification report saved: {cr_path}')

    macro_f1    = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    waste_f1    = report['waste']['f1-score']

    # â”€â”€ Per-image predictions JSON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    records = []
    for i in range(n_total):
        pred_idx = int(all_preds[i])
        true_idx = int(all_labels[i])
        records.append({
            'image_id':   all_image_ids[i],
            'image_path': all_img_paths[i],
            'true_label': CLASS_NAMES[true_idx],
            'pred_label': CLASS_NAMES[pred_idx],
            'confidence': round(float(all_probs[i][pred_idx]), 4),
            'correct':    bool(true_idx == pred_idx),
            'probs':      {c: round(float(all_probs[i][j]), 4)
                           for j, c in enumerate(CLASS_NAMES)},
        })
    pred_path = os.path.join(ARTIFACT_ROOT, 'val_predictions.json')
    with open(pred_path, 'w') as f:
        json.dump(records, f, indent=2)

    errors    = [r for r in records if not r['correct']]
    n_errors  = len(errors)

    # â”€â”€ Baseline comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    BASELINE = {
        'val_accuracy': 0.6226, 'macro_f1': 0.6311,
        'weighted_f1':  0.6201, 'waste_f1': 0.4828,
        'best_val_loss': 1.7228, 'best_epoch': 10,
        'train_acc_at_best': 0.9714,
    }
    best_epoch_row = history_df[history_df['epoch'] == early_stop.best_epoch]
    exp2_train_acc = float(best_epoch_row['train_accuracy'].iloc[0]) if len(best_epoch_row) else float('nan')
    train_val_gap  = exp2_train_acc - accuracy

    # â”€â”€ Validation report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def fmt_delta(new, old, higher_is_better=True):
        delta = new - old
        sign  = '+' if delta >= 0 else ''
        arrow = 'â†‘' if (delta > 0) == higher_is_better else ('â†“' if delta != 0 else '=')
        return f'{sign}{delta:+.4f} {arrow}'

    report_md = f"""# EfficientNet-B0 Experiment 2 â€” Validation Report

**Dataset:** Ecopin Pilot v1.0 (train/val only â€” test set untouched)
**Device:** {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})
**torch:** {torch.__version__}  **CUDA:** {torch.version.cuda}
**Batch size:** {BATCH_SIZE}  **Seed:** {SEED}
**Total epochs run:** {len(history)}
**Best checkpoint epoch:** {early_stop.best_epoch}
**Best val loss:** {early_stop.best_val_loss:.4f}

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
| Validation Accuracy | 0.6226 | {accuracy:.4f} | {fmt_delta(accuracy, BASELINE['val_accuracy'])} |
| Macro F1 | 0.6311 | {macro_f1:.4f} | {fmt_delta(macro_f1, BASELINE['macro_f1'])} |
| Weighted F1 | 0.6201 | {weighted_f1:.4f} | {fmt_delta(weighted_f1, BASELINE['weighted_f1'])} |
| Waste F1 | 0.4828 | {waste_f1:.4f} | {fmt_delta(waste_f1, BASELINE['waste_f1'])} |
| Best Val Loss | 1.7228 | {early_stop.best_val_loss:.4f} | {fmt_delta(early_stop.best_val_loss, BASELINE['best_val_loss'], higher_is_better=False)} |
| Best Epoch | 10 | {early_stop.best_epoch} | {early_stop.best_epoch - BASELINE['best_epoch']:+d} |

## Overfitting Comparison

| | Baseline @ best epoch | Experiment 2 @ best epoch |
|--|----------------------|--------------------------|
| Train accuracy | 0.9714 | {exp2_train_acc:.4f} |
| Val accuracy | 0.6226 | {accuracy:.4f} |
| Trainâˆ’Val gap | 0.3488 | {train_val_gap:.4f} |

## Overall Validation Metrics (best checkpoint)

| Metric | Value |
|--------|-------|
| Validation accuracy | {accuracy:.4f} ({n_correct}/{n_total}) |
| Macro F1 | {macro_f1:.4f} |
| Weighted F1 | {weighted_f1:.4f} |

## Per-Class Results

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
"""
    for cls in CLASS_NAMES:
        r = report[cls]
        report_md += f"| {cls} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support'])} |\\n"

    report_md += f"""
## Confusion Matrix (rows=True, cols=Predicted)

| True \\ Pred | {' | '.join(CLASS_NAMES)} |
|-------------|{'|'.join(['------']*4)}|
"""
    for i, cls in enumerate(CLASS_NAMES):
        vals = ' | '.join(str(cm[i, j]) for j in range(4))
        report_md += f"| {cls} | {vals} |\\n"

    # Off-diagonal sorted
    off = sorted([(cm[i,j], CLASS_NAMES[i], CLASS_NAMES[j])
                  for i in range(4) for j in range(4) if i != j], reverse=True)
    report_md += "\\n## Most Common Confusions\\n\\n| True | Predicted | Count |\\n|------|-----------|-------|\\n"
    for cnt, tc, pc in off[:6]:
        if cnt > 0:
            report_md += f"| {tc} | {pc} | {cnt} |\\n"

    report_md += f"""
## Misclassifications ({n_errors} / {n_total})

| Image ID | True | Predicted | Confidence |
|----------|------|-----------|------------|
"""
    for e in sorted(errors, key=lambda x: x['true_label']):
        report_md += f"| {e['image_id']} | {e['true_label']} | {e['pred_label']} | {e['confidence']:.3f} |\\n"

    report_md += """
## Interpretation Notes

- **Validation set is 53 images.** A 1-image change corresponds to ~1.9 pp accuracy swing.
  Any comparison should be interpreted carefully â€” individual image differences are meaningful.
- Comparison should focus on pattern changes (which classes improved/degraded), not just headline accuracy.
- The overfitting gap (trainâˆ’val accuracy) is the primary indicator of whether regularization helped.

## Visuals
- Training curves: `training_curves.png`
- Confusion matrix: `validation_confusion_matrix.png`

All artifacts in `c:/dev/ecopin_image_validation/artifacts/exp2_efficientnet_b0/`
"""
    report_path = os.path.join(ARTIFACT_ROOT, 'experiment2_validation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f'Validation report saved: {report_path}')

    # â”€â”€ Error analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    import math
    from itertools import product as iproduct

    error_md = f"""# Experiment 2 Validation Error Analysis

**Total validation images:** {n_total}
**Correct predictions:** {n_correct}
**Misclassifications:** {n_errors}
**Accuracy:** {accuracy:.4f}

## Misclassification Table

| # | Image ID | True | Predicted | Confidence |
|---|----------|------|-----------|------------|
"""
    for k, e in enumerate(sorted(errors, key=lambda x: (x['true_label'], x['image_id'])), 1):
        error_md += f"| {k} | {e['image_id']} | {e['true_label']} | {e['pred_label']} | {e['confidence']:.3f} |\\n"

    error_md += "\\n## Errors by True Class\\n\\n"
    for cls in CLASS_NAMES:
        cls_errors = [e for e in errors if e['true_label'] == cls]
        support    = sum(1 for r in records if r['true_label'] == cls)
        error_md  += f"### {cls} ({len(cls_errors)} errors / {support} samples)\\n\\n"
        if cls_errors:
            counts = {}
            for e in cls_errors:
                counts[e['pred_label']] = counts.get(e['pred_label'], 0) + 1
            for pc, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                error_md += f"- Predicted as **{pc}**: {cnt}\\n"
        else:
            error_md += "- No errors\\n"
        error_md += "\\n"

    # Bidirectional confusion table
    error_md += "## Bidirectional Confusion Totals\\n\\n| Pair | Aâ†’B | Bâ†’A | Total |\\n|------|-----|-----|-------|\\n"
    pairs = [(0,3),(0,1),(0,2),(1,3),(1,2),(2,3)]
    pair_totals = []
    for i, j in pairs:
        ab = int(cm[i,j]); ba = int(cm[j,i])
        pair_totals.append((ab+ba, CLASS_NAMES[i], CLASS_NAMES[j], ab, ba))
    for total_cnt, cn_i, cn_j, ab, ba in sorted(pair_totals, reverse=True):
        if total_cnt > 0:
            error_md += f"| {cn_i} â†” {cn_j} | {ab} | {ba} | **{total_cnt}** |\\n"

    error_md += f"\\n*This analysis covers the validation set only. Test set is untouched.*\\n"
    ea_path = os.path.join(ARTIFACT_ROOT, 'validation_error_analysis.md')
    with open(ea_path, 'w', encoding='utf-8') as f:
        f.write(error_md)
    print(f'Error analysis saved: {ea_path}')

    # â”€â”€ Contact sheet â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if n_errors > 0:
        cols = 4
        rows = math.ceil(n_errors / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        axes = np.array(axes).reshape(-1)
        for ax in axes: ax.axis('off')
        for k, e in enumerate(sorted(errors, key=lambda x: (x['true_label'], x['image_id']))):
            ax = axes[k]
            try:
                from PIL import Image as PILImage
                img = PILImage.open(e['image_path']).convert('RGB')
                ax.imshow(img)
            except Exception as ex:
                ax.text(0.5, 0.5, f'Error:\\n{ex}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=7)
            ax.set_title(
                f"{e['image_id']}\\nTrue: {e['true_label']}\\n"
                f"Pred: {e['pred_label']} ({e['confidence']:.2f})",
                fontsize=7)
            ax.axis('off')
        plt.suptitle(f'Exp2 Misclassification Contact Sheet ({n_errors} images)', fontsize=11)
        plt.tight_layout()
        cs_path = os.path.join(ARTIFACT_ROOT, 'misclassification_contact_sheet.png')
        plt.savefig(cs_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f'Contact sheet saved: {cs_path}')

    print('\\n=== Experiment 2 Complete ===')
    print(f'Best epoch          : {early_stop.best_epoch}')
    print(f'Best val loss       : {early_stop.best_val_loss:.4f}')
    print(f'Val accuracy        : {accuracy:.4f} ({n_correct}/{n_total})')
    print(f'Macro F1            : {macro_f1:.4f}')
    print(f'Weighted F1         : {weighted_f1:.4f}')
    print(f'Waste F1            : {waste_f1:.4f}')
    print(f'Train-val gap       : {train_val_gap:.4f} (baseline: 0.3488)')


if __name__ == '__main__':
    train()


