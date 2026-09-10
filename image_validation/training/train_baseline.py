import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
# Ensure project root is on sys.path for absolute imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from timm import create_model
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Local imports
from training.dataset import EcopinDataset
from training.utils import (
    compute_class_weights,
    plot_training_curves,
    plot_confusion_matrix,
    generate_classification_report,
    save_training_config,
)

# Additional utilities defined locally
import torchvision.transforms as T
import torchvision.transforms.functional as TF

def get_transform(train: bool = False):
    """Create transform pipeline.
    Training: light horizontal flip.
    Validation: deterministic.
    """
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(p=0.2))
    transforms.extend([
        T.Lambda(_pad_to_square),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return T.Compose(transforms)

def _pad_to_square(img):
    w, h = img.size
    max_side = max(w, h)
    pad_left = (max_side - w) // 2
    pad_top = (max_side - h) // 2
    pad_right = max_side - w - pad_left
    pad_bottom = max_side - h - pad_top
    padding = (pad_left, pad_top, pad_right, pad_bottom)
    return TF.pad(img, padding, fill=0)

# Simple EarlyStopping implementation
class EarlyStopping:
    def __init__(self, patience: int = 5, verbose: bool = False, checkpoint_path: str = 'best.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path
        self.best_epoch = None
        self.best_metric = None

    def __call__(self, epoch: int, val_loss: float, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_metric = val_loss
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_metric = val_loss
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
            self.counter = 0

# Paths config (reuse same config as dataset_paths_and_split_config)
from data_config.dataset_paths_and_split_config import TRAIN_CSV, VAL_CSV, SEED

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Approved hyper‑parameters
MAX_EPOCHS = 20
STAGE1_EPOCHS = 5
BATCH_SIZE = 16  # will be reduced if OOM
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-4
PATIENCE = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output locations
ARTIFACT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'baseline_efficientnet_b0'))
CHECKPOINT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
os.makedirs(ARTIFACT_ROOT, exist_ok=True)
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

# Helper to build DataLoader (retry with smaller batch if OOM)
def build_loader(csv_path, transform, batch_size):
    dataset = EcopinDataset(csv_path, transform=transform)
    while True:
        try:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            # test a batch to detect OOM early
            next(iter(loader))
            return loader
        except RuntimeError as e:
            if 'out of memory' in str(e):
                batch_size = max(1, batch_size // 2)
                print(f'GPU OOM – reducing batch size to {batch_size}')
                torch.cuda.empty_cache()
            else:
                raise

def train():
    # Data loaders
    train_loader = build_loader(TRAIN_CSV, get_transform(train=True), BATCH_SIZE)
    val_loader = build_loader(VAL_CSV, get_transform(train=False), BATCH_SIZE)

    # Class weights
    class_weights = compute_class_weights(TRAIN_CSV).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Model
    model = create_model('efficientnet_b0', pretrained=True, num_classes=4)
    model = model.to(DEVICE)

    # Stage 1: freeze backbone
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_STAGE1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stop = EarlyStopping(patience=PATIENCE, verbose=True,
                               checkpoint_path=os.path.join(CHECKPOINT_ROOT, 'efficientnet_b0_best.pt'))

    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    total_epochs = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        # Training epoch
        model.train()
        train_loss_sum = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
        train_loss = train_loss_sum / total
        train_acc = correct / total

        # Validation epoch
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        # Record history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Scheduler step
        scheduler.step(val_loss)

        # Early stopping check
        early_stop(epoch, val_loss, model)
        if early_stop.early_stop:
            print(f'Early stopping triggered at epoch {epoch}')
            break

        # Transition to Stage 2 after completing STAGE1_EPOCHS epochs (backbone remains frozen for these epochs)
        if epoch == STAGE1_EPOCHS:
            print('--- Transition to Stage 2: unfreeze backbone, LR set to', LR_STAGE2)
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=LR_STAGE2)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        print(f'Epoch {epoch:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | train acc {train_acc:.4f} | val acc {val_acc:.4f}')
        total_epochs = epoch

    # Load best checkpoint
    best_model = create_model('efficientnet_b0', pretrained=False, num_classes=4)
    best_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_ROOT, 'efficientnet_b0_best.pt')))
    best_model = best_model.to(DEVICE)
    best_model.eval()

    # Final validation with best model
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = best_model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    class_names = ['flooding', 'non_environmental', 'pollution', 'waste']
    cm = confusion_matrix(all_labels, all_preds)
    report = generate_classification_report(all_labels, all_preds, class_names,
                                            os.path.join(ARTIFACT_ROOT, 'classification_report.json'))
    # Save history and plots
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(ARTIFACT_ROOT, 'training_history.csv'), index=False)
    plot_training_curves(history_df, os.path.join(ARTIFACT_ROOT, 'training_curves.png'))
    plot_confusion_matrix(cm, class_names, os.path.join(ARTIFACT_ROOT, 'validation_confusion_matrix.png'))
    # Save config
    config = {
        'model': 'efficientnet_b0',
        'pretrained': True,
        'max_epochs': MAX_EPOCHS,
        'stage1_epochs': STAGE1_EPOCHS,
        'stage1_lr': LR_STAGE1,
        'stage2_lr': LR_STAGE2,
        'batch_size': BATCH_SIZE,
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'early_stopping_patience': PATIENCE,
        'seed': SEED,
        'device': str(DEVICE),
        'total_epochs_trained': total_epochs,
        'early_stop_epoch': early_stop.best_epoch,
        'best_val_loss': early_stop.best_metric,
    }
    save_training_config(config, os.path.join(ARTIFACT_ROOT, 'config.json'))

    # Markdown validation report
    report_md = f"""# EfficientNet‑B0 Baseline – Validation Report

**Dataset:** Ecopin Pilot v1.0 (train/val only)\n**Device:** {DEVICE}\n**Final batch size:** {BATCH_SIZE}\n**Total epochs run:** {total_epochs}\n**Early‑stop epoch:** {early_stop.best_epoch}\n**Best validation loss:** {early_stop.best_metric:.4f}\n\n## Metrics\n| Metric | Value |\n|--------|-------|\n| Validation Accuracy | {val_acc:.4f} |\n| Macro F1 | {report['macro avg']['f1-score']:.4f} |\n| Weighted F1 | {report['weighted avg']['f1-score']:.4f} |\n\n## Per‑class results (precision, recall, f1‑score, support)\n```
{json.dumps(report, indent=2)}
```\n\n## Visuals\n- Training / validation curves: ![Training Curves]({os.path.join(ARTIFACT_ROOT, 'training_curves.png')})\n- Confusion matrix: ![Confusion Matrix]({os.path.join(ARTIFACT_ROOT, 'validation_confusion_matrix.png')})\n\nAll artifacts are stored in `{ARTIFACT_ROOT}`.\n"""
    report_path = os.path.join(ARTIFACT_ROOT, 'baseline_validation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print('Validation report saved to', report_path)

if __name__ == '__main__':
    train()
