import os
import sys
import json
import random
import time
import platform
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder
from timm import create_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

# ── Ensure UTF-8 output on Windows ──
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Configuration ──
SEED = 42
BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_CLASSES = 4
CLASS_NAMES = ['flooding', 'non_environmental', 'pollution', 'waste']
MAX_EPOCHS = 20
INITIAL_LR = 1e-5
PATIENCE = 5

HISTORICAL_CKPT_PATH = r'c:\dev\ecopin_ml_models\baseline_original_350\best_model.pt'
SPLIT_DIR = r'c:\dev\datasets\ecopin_dataset\split'
TRAIN_DIR = os.path.join(SPLIT_DIR, 'train')
VAL_DIR = os.path.join(SPLIT_DIR, 'val')
TEST_DIR = os.path.join(SPLIT_DIR, 'test')

EXP_DIR = r'c:\dev\ecopin_ml_models\final_1000_dataset\experiments\experiment_02_baseline_finetune'
os.makedirs(EXP_DIR, exist_ok=True)

# ── Seed everything for reproducibility ──
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ── Pad to Square Transform ──
class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        pad_l = (max_side - w) // 2
        pad_t = (max_side - h) // 2
        pad_r = max_side - w - pad_l
        pad_b = max_side - h - pad_t
        return TF.pad(img, (pad_l, pad_t, pad_r, pad_b), fill=0)

# ── Transforms (Identical to Experiment 01 for controlled comparison) ──
train_transforms = T.Compose([
    PadToSquare(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    PadToSquare(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── Data Loaders ──
train_dataset = ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = ImageFolder(VAL_DIR, transform=eval_transforms)
test_dataset = ImageFolder(TEST_DIR, transform=eval_transforms)

# Verify class to index mapping
print("Class to idx:", train_dataset.class_to_idx)
assert list(train_dataset.class_to_idx.keys()) == CLASS_NAMES

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"Loaded: {len(train_dataset)} Train, {len(val_dataset)} Validation, {len(test_dataset)} Test images.")

# ── Model Initialization from Historical Exp 8 Checkpoint ──
print(f"\nLoading historical baseline checkpoint: {HISTORICAL_CKPT_PATH}")
checkpoint = torch.load(HISTORICAL_CKPT_PATH, map_location='cpu', weights_only=False)

model = create_model('efficientnet_b0', pretrained=False, num_classes=NUM_CLASSES)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"Loaded checkpoint metadata: Epoch={checkpoint.get('epoch')}, Best Macro F1={checkpoint.get('best_macro_f1'):.4f}, Best Val Acc={checkpoint.get('best_val_acc'):.4f}")
else:
    state_dict = checkpoint

missing, unexpected = model.load_state_dict(state_dict, strict=True)
print(f"Checkpoint loaded successfully into EfficientNet-B0 (Missing: {missing}, Unexpected: {unexpected})")

model = model.to(device)

# Unweighted CrossEntropyLoss since the 1,000-image dataset is perfectly balanced (250 images / class)
criterion = nn.CrossEntropyLoss()

# Conservative fine-tuning: all layers trainable at low LR 1e-5
optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    cls_report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds).tolist()
    
    return avg_loss, acc, macro_f1, weighted_f1, cls_report, cm, all_preds, all_probs, all_targets

# ── Zero-shot / Pre-Fine-Tuning Evaluation of Historical Model on New Splits ──
print("\n--- Zero-Shot Performance of Historical Model on Final Dataset (Before Fine-Tuning) ---")
init_val_loss, init_val_acc, init_val_macro_f1, _, _, _, _, _, _ = evaluate(model, val_loader)
init_test_loss, init_test_acc, init_test_macro_f1, _, _, _, _, _, _ = evaluate(model, test_loader)
print(f"Pre-Fine-Tuning Val Acc: {init_val_acc*100:.2f}%, Val Macro F1: {init_val_macro_f1:.4f}, Val Loss: {init_val_loss:.4f}")
print(f"Pre-Fine-Tuning Test Acc: {init_test_acc*100:.2f}%, Test Macro F1: {init_test_macro_f1:.4f}, Test Loss: {init_test_loss:.4f}")

# ── Training Loop ──
history = []
per_epoch_metrics = []

best_macro_f1 = -1.0
best_val_acc = -1.0
best_val_loss = float('inf')
best_epoch = -1
epochs_without_improvement = 0

print("\n=======================================================")
print("  FINE-TUNING: Historical Exp 8 on 1,000-Image Dataset ")
print("=======================================================")

for epoch in range(1, MAX_EPOCHS + 1):
    current_lr = optimizer.param_groups[0]['lr']
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
    train_loss = running_loss / total
    train_acc = correct / total
    
    val_loss, val_acc, val_macro_f1, val_weighted_f1, cls_report, cm, _, _, _ = evaluate(model, val_loader)
    scheduler.step(val_loss)
    
    is_best = False
    if val_macro_f1 > best_macro_f1:
        best_macro_f1 = val_macro_f1
        best_val_acc = val_acc
        best_val_loss = val_loss
        best_epoch = epoch
        is_best = True
        epochs_without_improvement = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_macro_f1': best_macro_f1,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'class_names': CLASS_NAMES,
            'base_checkpoint': HISTORICAL_CKPT_PATH
        }, os.path.join(EXP_DIR, 'best_model.pt'))
    else:
        epochs_without_improvement += 1
        
    log_entry = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_macro_f1': val_macro_f1,
        'val_weighted_f1': val_weighted_f1,
        'lr': current_lr,
        'is_best': is_best
    }
    history.append(log_entry)
    per_epoch_metrics.append({
        'epoch': epoch,
        'metrics': log_entry,
        'classification_report': cls_report,
        'confusion_matrix': cm
    })
    
    print(f"Epoch [{epoch:02d}/{MAX_EPOCHS:02d}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val Macro F1: {val_macro_f1:.4f} (LR: {current_lr:.1e}) {'*' if is_best else ''}")
    
    if epochs_without_improvement >= PATIENCE:
        print(f"\n[Early Stopping] Triggered after {epochs_without_improvement} epochs without improvement in validation Macro F1.")
        break

# Save final checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'class_names': CLASS_NAMES,
    'base_checkpoint': HISTORICAL_CKPT_PATH
}, os.path.join(EXP_DIR, 'final_model.pt'))

# ── Load Best Checkpoint for Final Validation & Test Evaluation ──
print("\n=======================================================")
print(f"  Evaluating Best Model Checkpoint (from Epoch {best_epoch})  ")
print("=======================================================")

best_ckpt = torch.load(os.path.join(EXP_DIR, 'best_model.pt'), map_location=device, weights_only=False)
model.load_state_dict(best_ckpt['model_state_dict'])

# Validation evaluation
val_loss, val_acc, val_macro_f1, val_weighted_f1, val_report, val_cm, val_preds, val_probs, val_targets = evaluate(model, val_loader)

# Test evaluation (single run on split/test)
test_loss, test_acc, test_macro_f1, test_weighted_f1, test_report, test_cm, test_preds, test_probs, test_targets = evaluate(model, test_loader)

print(f"\n[Best Validation Results]")
print(f"  Accuracy: {val_acc*100:.2f}%")
print(f"  Macro F1: {val_macro_f1:.4f}")
print(f"  Loss:     {val_loss:.4f}")

print(f"\n[Final Test Results]")
print(f"  Accuracy: {test_acc*100:.2f}%")
print(f"  Macro F1: {test_macro_f1:.4f}")
print(f"  Loss:     {test_loss:.4f}")

# ── Save Predictions and Artifacts ──

# 1. Validation predictions JSON
val_pred_records = []
for idx, (path, target) in enumerate(val_dataset.samples):
    val_pred_records.append({
        'filename': os.path.basename(path),
        'true_label': CLASS_NAMES[target],
        'pred_label': CLASS_NAMES[val_preds[idx]],
        'correct': bool(val_preds[idx] == target),
        'probabilities': {CLASS_NAMES[i]: float(val_probs[idx][i]) for i in range(NUM_CLASSES)}
    })

with open(os.path.join(EXP_DIR, 'val_predictions.json'), 'w', encoding='utf-8') as f:
    json.dump(val_pred_records, f, indent=2)

# 2. Test predictions JSON
test_pred_records = []
for idx, (path, target) in enumerate(test_dataset.samples):
    test_pred_records.append({
        'filename': os.path.basename(path),
        'true_label': CLASS_NAMES[target],
        'pred_label': CLASS_NAMES[test_preds[idx]],
        'correct': bool(test_preds[idx] == target),
        'probabilities': {CLASS_NAMES[i]: float(test_probs[idx][i]) for i in range(NUM_CLASSES)}
    })

with open(os.path.join(EXP_DIR, 'test_predictions.json'), 'w', encoding='utf-8') as f:
    json.dump(test_pred_records, f, indent=2)

# 3. Save training history CSV
pd.DataFrame(history).to_csv(os.path.join(EXP_DIR, 'training_history.csv'), index=False)

# 4. Save per-epoch metrics JSON
with open(os.path.join(EXP_DIR, 'per_epoch_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(per_epoch_metrics, f, indent=2)

# 5. Save classification reports
reports_data = {
    'validation': val_report,
    'test': test_report
}
with open(os.path.join(EXP_DIR, 'classification_report.json'), 'w', encoding='utf-8') as f:
    json.dump(reports_data, f, indent=2)

# 6. Save confusion matrices
cm_data = {
    'class_names': CLASS_NAMES,
    'validation_confusion_matrix': val_cm,
    'test_confusion_matrix': test_cm
}
with open(os.path.join(EXP_DIR, 'confusion_matrix.json'), 'w', encoding='utf-8') as f:
    json.dump(cm_data, f, indent=2)

# 7. Save configuration JSON
config_data = {
    'experiment': 'experiment_02_baseline_finetune',
    'date_trained': time.strftime('%Y-%m-%d %H:%M:%S'),
    'environment': {
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'
    },
    'dataset': {
        'path': SPLIT_DIR,
        'total_images': 1000,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'num_classes': NUM_CLASSES,
        'classes': CLASS_NAMES,
        'distribution': '200 train / 25 val / 25 test per class (balanced)'
    },
    'model': {
        'architecture': 'efficientnet_b0',
        'initialization_source': 'Historical Exp 8 Checkpoint (c:/dev/ecopin_ml_models/baseline_original_350/best_model.pt)',
        'initialization_type': 'Pre-trained on original ~350 EcoPin dataset'
    },
    'hyperparameters': {
        'seed': SEED,
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'loss_function': 'CrossEntropyLoss (unweighted, perfectly balanced dataset)',
        'optimizer': 'Adam',
        'initial_learning_rate': INITIAL_LR,
        'max_epochs': MAX_EPOCHS,
        'scheduler': 'ReduceLROnPlateau(mode=min, factor=0.5, patience=2)',
        'early_stopping_patience': PATIENCE,
        'early_stopping_metric': 'Macro F1 (maximize)',
        'augmentations': [
            'PadToSquare(fill=0)',
            'Resize(224, 224)',
            'RandomHorizontalFlip(p=0.2)',
            'Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])'
        ]
    },
    'pre_finetuning_metrics': {
        'val_acc': init_val_acc,
        'val_macro_f1': init_val_macro_f1,
        'val_loss': init_val_loss,
        'test_acc': init_test_acc,
        'test_macro_f1': init_test_macro_f1,
        'test_loss': init_test_loss
    },
    'results': {
        'best_epoch': best_epoch,
        'total_epochs_trained': epoch,
        'validation': {
            'accuracy': val_acc,
            'macro_f1': val_macro_f1,
            'weighted_f1': val_weighted_f1,
            'loss': val_loss
        },
        'test': {
            'accuracy': test_acc,
            'macro_f1': test_macro_f1,
            'weighted_f1': test_weighted_f1,
            'loss': test_loss
        }
    },
    'comparison_to_exp01': {
        'exp01_test_acc': 0.5200,
        'exp01_test_macro_f1': 0.5174,
        'exp02_test_acc': test_acc,
        'exp02_test_macro_f1': test_macro_f1,
        'test_acc_difference': test_acc - 0.5200,
        'test_macro_f1_difference': test_macro_f1 - 0.5174
    },
    'historical_comparison': {
        'historical_exp8_val_acc': 0.6981,
        'historical_exp8_val_macro_f1': 0.6743,
        'note': 'Historical Exp 8 was trained on original ~350-image dataset (53 val images). Provided for context only.'
    }
}

with open(os.path.join(EXP_DIR, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(config_data, f, indent=2)

# 8. Create README.md
readme_content = f"""# Experiment 02 — Historical Exp 8 Fine-Tune on Final 1,000-Image Dataset

## 1. Experiment Objective
Determine whether fine-tuning the historical best EcoPin model (Experiment 8, trained on the original ~350-image dataset) yields superior generalization on the finalized 1,000-image dataset compared to the fresh ImageNet-pretrained model from Experiment 01.

## 2. Dataset and Split
- **Total Images:** 1,000 images
- **Dataset Path:** `c:/dev/datasets/ecopin_dataset/split/`
- **Partition:** 800 train / 100 validation / 100 test (80/10/10 stratified split)
- **Class Distribution:** Exactly 250 images per class (200 train / 25 val / 25 test per class)
  - `flooding`: 250 images
  - `non_environmental`: 250 images
  - `pollution`: 250 images
  - `waste`: 250 images

## 3. Model Architecture
- **Architecture:** EfficientNet-B0 (`timm`)
- **Number of Classes:** 4 (`flooding`, `non_environmental`, `pollution`, `waste`)

## 4. Historical Checkpoint Used
- **Source Checkpoint:** `c:/dev/ecopin_ml_models/baseline_original_350/best_model.pt` (Experiment 8)
- **Original Training:** Trained on the 350-image dataset (245 train / 53 val / 52 test)
- **Historical Benchmark:** 69.81% Val Acc, 0.6743 Macro F1 on the old 53-image validation split.

## 5. Training Configuration
- **Random Seed:** {SEED}
- **Image Size:** 224 × 224 (aspect-preserving square padding)
- **Batch Size:** {BATCH_SIZE}
- **Loss Function:** `CrossEntropyLoss` (unweighted; uniform class representation)
- **Optimizer:** Adam
- **Initial Learning Rate:** {INITIAL_LR:.1e} (conservative fine-tuning)
- **Scheduler:** `ReduceLROnPlateau(mode='min', factor=0.5, patience=2)`
- **Early Stopping:** Patience = {PATIENCE} epochs monitoring validation Macro F1
- **Augmentation:** `PadToSquare`, `Resize(224, 224)`, `RandomHorizontalFlip(p=0.2)`, `ImageNet Normalization`
- **Total Epochs Trained:** {epoch}

## 6. Fine-Tuning Strategy
Direct end-to-end conservative fine-tuning initialized from the historical Exp 8 weights at a low learning rate (`1e-5`). Since the classification head was already trained for the 4 EcoPin classes, unfreezing the full network immediately at `1e-5` allows seamless adaptation across feature extraction and classification layers without catastrophic forgetting.

## 7. Best Epoch
- **Best Epoch:** Epoch {best_epoch} (out of {epoch} epochs)

## 8. Best Validation Metrics
- **Validation Accuracy:** {val_acc*100:.2f}% ({val_acc:.4f})
- **Validation Macro F1:** {val_macro_f1:.4f}
- **Validation Loss:** {val_loss:.4f}

## 9. Final Test Metrics (Evaluated Once on `split/test`)
- **Test Accuracy:** {test_acc*100:.2f}% ({test_acc:.4f})
- **Test Macro F1:** {test_macro_f1:.4f}
- **Test Loss:** {test_loss:.4f}

## 10. Per-Class Results

### Validation Set (100 samples)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **flooding** | {val_report['flooding']['precision']:.4f} | {val_report['flooding']['recall']:.4f} | {val_report['flooding']['f1-score']:.4f} | {val_report['flooding']['support']} |
| **non_environmental** | {val_report['non_environmental']['precision']:.4f} | {val_report['non_environmental']['recall']:.4f} | {val_report['non_environmental']['f1-score']:.4f} | {val_report['non_environmental']['support']} |
| **pollution** | {val_report['pollution']['precision']:.4f} | {val_report['pollution']['recall']:.4f} | {val_report['pollution']['f1-score']:.4f} | {val_report['pollution']['support']} |
| **waste** | {val_report['waste']['precision']:.4f} | {val_report['waste']['recall']:.4f} | {val_report['waste']['f1-score']:.4f} | {val_report['waste']['support']} |
| **Macro Avg** | {val_report['macro avg']['precision']:.4f} | {val_report['macro avg']['recall']:.4f} | **{val_report['macro avg']['f1-score']:.4f}** | {val_report['macro avg']['support']} |

### Test Set (100 samples)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **flooding** | {test_report['flooding']['precision']:.4f} | {test_report['flooding']['recall']:.4f} | {test_report['flooding']['f1-score']:.4f} | {test_report['flooding']['support']} |
| **non_environmental** | {test_report['non_environmental']['precision']:.4f} | {test_report['non_environmental']['recall']:.4f} | {test_report['non_environmental']['f1-score']:.4f} | {test_report['non_environmental']['support']} |
| **pollution** | {test_report['pollution']['precision']:.4f} | {test_report['pollution']['recall']:.4f} | {test_report['pollution']['f1-score']:.4f} | {test_report['pollution']['support']} |
| **waste** | {test_report['waste']['precision']:.4f} | {test_report['waste']['recall']:.4f} | {test_report['waste']['f1-score']:.4f} | {test_report['waste']['support']} |
| **Macro Avg** | {test_report['macro avg']['precision']:.4f} | {test_report['macro avg']['recall']:.4f} | **{test_report['macro avg']['f1-score']:.4f}** | {test_report['macro avg']['support']} |

## 11. Confusion Matrix

### Validation Confusion Matrix
```text
               Predicted
           FLD  NON  POL  WST
True  FLD  [{val_cm[0][0]:2d},  {val_cm[0][1]:2d},  {val_cm[0][2]:2d},  {val_cm[0][3]:2d}]
      NON  [{val_cm[1][0]:2d},  {val_cm[1][1]:2d},  {val_cm[1][2]:2d},  {val_cm[1][3]:2d}]
      POL  [{val_cm[2][0]:2d},  {val_cm[2][1]:2d},  {val_cm[2][2]:2d},  {val_cm[2][3]:2d}]
      WST  [{val_cm[3][0]:2d},  {val_cm[3][1]:2d},  {val_cm[3][2]:2d},  {val_cm[3][3]:2d}]
```

### Test Confusion Matrix
```text
               Predicted
           FLD  NON  POL  WST
True  FLD  [{test_cm[0][0]:2d},  {test_cm[0][1]:2d},  {test_cm[0][2]:2d},  {test_cm[0][3]:2d}]
      NON  [{test_cm[1][0]:2d},  {test_cm[1][1]:2d},  {test_cm[1][2]:2d},  {test_cm[1][3]:2d}]
      POL  [{test_cm[2][0]:2d},  {test_cm[2][1]:2d},  {test_cm[2][2]:2d},  {test_cm[2][3]:2d}]
      WST  [{test_cm[3][0]:2d},  {test_cm[3][1]:2d},  {test_cm[3][2]:2d},  {test_cm[3][3]:2d}]
```

## 12. Comparison with Experiment 01

| Metric (Test Set - 100 images) | Experiment 01 (Fresh Pretrained) | Experiment 02 (Historical Exp 8 Fine-Tune) | Difference (Exp 02 - Exp 01) |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 52.00% | {test_acc*100:.2f}% | {(test_acc - 0.5200)*100:+.2f}% |
| **Test Macro F1** | 0.5174 | {test_macro_f1:.4f} | {(test_macro_f1 - 0.5174):+.4f} |
| **Test Loss** | 1.6902 | {test_loss:.4f} | {(test_loss - 1.6902):+.4f} |

## 13. Comparison with Historical Exp 8
- **Historical Exp 8 (Validation Benchmark):** 69.81% Val Acc / 0.6743 Macro F1 on 53 validation images from original ~350-image dataset.
- **Experiment 02 (Validation):** {val_acc*100:.2f}% Val Acc / {val_macro_f1:.4f} Macro F1 on 100 validation images from finalized 1,000-image dataset.
- *Context Distinction:* Historical Exp 8 had only 53 validation samples with known leakage before the 96 replacement protocol. Experiment 02 evaluates against a clean, leakage-free 100-sample validation set and 100-sample test set.

## 14. Final Conclusion
{'Experiment 02 demonstrated superior performance over Experiment 01' if test_macro_f1 > 0.5174 else 'Experiment 02 showed comparable/lower performance compared to Experiment 01'} on the finalized 100-image test set, achieving **{test_acc*100:.2f}% test accuracy** and **{test_macro_f1:.4f} test Macro F1**.
"""

with open(os.path.join(EXP_DIR, 'README.md'), 'w', encoding='utf-8') as f:
    f.write(readme_content)

print(f"\n[Artifacts Successfully Written to {EXP_DIR}]")
