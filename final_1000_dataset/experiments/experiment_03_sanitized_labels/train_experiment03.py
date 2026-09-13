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
sys.stdout.reconfigure(line_buffering=True)

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
SPLIT_DIR = r'c:\dev\datasets\ecopin_dataset\experiment_03_sanitized'
TRAIN_DIR = os.path.join(SPLIT_DIR, 'train')
VAL_DIR = os.path.join(SPLIT_DIR, 'val')
TEST_DIR = os.path.join(SPLIT_DIR, 'test')

EXP_DIR = r'c:\dev\ecopin_ml_models\final_1000_dataset\experiments\experiment_03_sanitized_labels'
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

# ── Transforms (Identical to Experiment 01 and 02 for controlled comparison) ──
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

print(f"Loaded Sanitized Dataset: {len(train_dataset)} Train, {len(val_dataset)} Validation, {len(test_dataset)} Test images.")

# ── Save Label-Change Manifest ──
manifest_data = {
    'sanitization_date': time.strftime('%Y-%m-%d %H:%M:%S'),
    'source_dataset': r'c:\dev\datasets\ecopin_dataset\split',
    'sanitized_dataset': SPLIT_DIR,
    'total_images_changed': 5,
    'total_candidates_audited': 7,
    'applied_changes': [
        {
            'image_id': 'NEG_126',
            'filename': 'NEG_126.jpg',
            'split': 'train',
            'original_label': 'non_environmental',
            'new_label': 'waste',
            'reason': 'Open rickshaw overflowing with uncollected trash bags and loose garbage.',
            'verified': True
        },
        {
            'image_id': 'NEG_136',
            'filename': 'NEG_136.jpg',
            'split': 'train',
            'original_label': 'non_environmental',
            'new_label': 'waste',
            'reason': 'Pick-up truck overloaded with a high pile of junk, mattresses, and discarded household waste.',
            'verified': True
        },
        {
            'image_id': 'NEG_141',
            'filename': 'NEG_141.jpg',
            'split': 'train',
            'original_label': 'non_environmental',
            'new_label': 'waste',
            'reason': 'Motorized tricycle loaded high with unbagged loose garbage and refuse.',
            'verified': True
        },
        {
            'image_id': 'NEG_142',
            'filename': 'NEG_142.jpg',
            'split': 'val',
            'original_label': 'non_environmental',
            'new_label': 'waste',
            'reason': 'Outdoor dumpsters with overflowing trash and loose garbage scattered on surrounding ground.',
            'verified': True
        },
        {
            'image_id': 'NEG_248',
            'filename': 'NEG_248.jpg',
            'split': 'train',
            'original_label': 'non_environmental',
            'new_label': 'waste',
            'reason': 'Discarded plastic garbage, empty bottles, and litter scattered across ground.',
            'verified': True
        }
    ],
    'protected_test_candidates_unchanged': [
        {
            'image_id': 'NEG_222',
            'filename': 'NEG_222.jpg',
            'split': 'test',
            'original_label': 'non_environmental',
            'new_label': 'non_environmental (UNCHANGED)',
            'reason': 'Protected by Test-Set Protection Rule to ensure test set remains 100% identical across Exp 01/02/03.'
        },
        {
            'image_id': 'NEG_249',
            'filename': 'NEG_249.jpg',
            'split': 'test',
            'original_label': 'non_environmental',
            'new_label': 'non_environmental (UNCHANGED)',
            'reason': 'Protected by Test-Set Protection Rule to ensure test set remains 100% identical across Exp 01/02/03.'
        }
    ]
}

with open(os.path.join(EXP_DIR, 'label_change_manifest.json'), 'w', encoding='utf-8') as f:
    json.dump(manifest_data, f, indent=2)

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

criterion = nn.CrossEntropyLoss()
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

# ── Zero-shot / Pre-Fine-Tuning Evaluation of Historical Model ──
print("\n--- Zero-Shot Performance of Historical Model on Sanitized Splits (Before Fine-Tuning) ---")
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
print("  EXPERIMENT 03: Sanitized-Label Fine-Tuning         ")
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
    'experiment': 'experiment_03_sanitized_labels',
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
        'distribution': {
            'train': {'flooding': 200, 'non_environmental': 196, 'pollution': 200, 'waste': 204},
            'val': {'flooding': 25, 'non_environmental': 24, 'pollution': 25, 'waste': 26},
            'test': {'flooding': 25, 'non_environmental': 25, 'pollution': 25, 'waste': 25}
        }
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
        'loss_function': 'CrossEntropyLoss (unweighted)',
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
    'comparison_summary': {
        'exp01_test_acc': 0.5200,
        'exp01_test_macro_f1': 0.5174,
        'exp02_test_acc': 0.6200,
        'exp02_test_macro_f1': 0.6105,
        'exp03_test_acc': test_acc,
        'exp03_test_macro_f1': test_macro_f1,
        'diff_exp03_minus_exp02_acc': test_acc - 0.6200,
        'diff_exp03_minus_exp02_macro_f1': test_macro_f1 - 0.6105
    }
}

with open(os.path.join(EXP_DIR, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(config_data, f, indent=2)

print(f"\n[Training and Evaluation Complete. Artifacts Saved to {EXP_DIR}]")
