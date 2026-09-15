"""
Experiment 03 — Lightweight Binary Classifiers on Frozen CLIP Embeddings
=========================================================================
Pipeline : Frozen CLIP ViT-B/32  →  512-D embeddings  →  classifier
Input    : c:/dev/datasets/ecopin_dataset/embeddings/exp03/clip_vitb32_{train,val,test}.npz
Output   : c:/dev/ecopin_ml_models/image_validation/artifacts/exp03_clip_classifiers/

Three classifiers trained and compared:
  1. Linear      — 512 → 1  (single linear layer, BCEWithLogitsLoss)
  2. SmallMLP    — 512 → 128 → 32 → 1  (ReLU, no dropout)
  3. RegMLP      — 512 → 128 → 32 → 1  (ReLU + Dropout(0.3), weight_decay=1e-3)

Labels  : valid=1, invalid=0
Seed    : 42
Device  : CUDA if available, else CPU
"""

import json
import os
import random
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    f1_score, confusion_matrix, classification_report,
)

# ── PATHS ─────────────────────────────────────────────────────────────────────
EMB_DIR   = Path(r"C:\dev\datasets\ecopin_dataset\embeddings\exp03")
OUT_DIR   = Path(r"C:\dev\ecopin_ml_models\image_validation\artifacts\exp03_clip_classifiers")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────────
SEED          = 42
BATCH_SIZE    = 64
MAX_EPOCHS    = 200
LR            = 1e-3
PATIENCE      = 15       # early stopping on val macro-F1
DROPOUT       = 0.3
WEIGHT_DECAY  = 1e-3
EMB_DIM       = 512
CLASS_NAMES   = ["invalid", "valid"]   # indices 0, 1

# ── REPRODUCIBILITY ───────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

# ── LOAD EMBEDDINGS ───────────────────────────────────────────────────────────
def load_split(name: str):
    d = np.load(str(EMB_DIR / f"clip_vitb32_{name}.npz"), allow_pickle=True)
    X = torch.tensor(d["embeddings"], dtype=torch.float32)
    y = torch.tensor(d["label_ints"], dtype=torch.float32)
    fns = d["filenames"].tolist()
    return X, y, fns

print("Loading embeddings …")
X_train, y_train, fns_train = load_split("train")
X_val,   y_val,   fns_val   = load_split("val")
X_test,  y_test,  fns_test  = load_split("test")

assert X_train.shape == (840, EMB_DIM), X_train.shape
assert X_val.shape   == (180, EMB_DIM), X_val.shape
assert X_test.shape  == (180, EMB_DIM), X_test.shape
print(f"Train : {X_train.shape}  |  pos={int(y_train.sum())}  neg={int((y_train==0).sum())}")
print(f"Val   : {X_val.shape}  |  pos={int(y_val.sum())}  neg={int((y_val==0).sum())}")
print(f"Test  : {X_test.shape}  |  pos={int(y_test.sum())}  neg={int((y_test==0).sum())}")

train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False,
)

# Class-weighted loss (handles 2:1 imbalance)
pos_weight = torch.tensor([y_train.shape[0] / y_train.sum() * 0.5]).to(device)  # ~0.75
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ── ARCHITECTURES ─────────────────────────────────────────────────────────────
class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(EMB_DIM, 1)
    def forward(self, x):
        return self.fc(x).squeeze(1)

class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMB_DIM, 128), nn.ReLU(),
            nn.Linear(128, 32),      nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

class RegMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMB_DIM, 128), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(128, 32),      nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

MODELS = {
    "linear":   LinearClassifier,
    "small_mlp": SmallMLP,
    "reg_mlp":  RegMLP,
}

# ── EVALUATION ────────────────────────────────────────────────────────────────
def evaluate(model, X, y_true_tensor, split_name: str) -> dict:
    model.eval()
    with torch.no_grad():
        logits_gpu = model(X.to(device))
        loss       = criterion(logits_gpu, y_true_tensor.to(device)).item()
        logits     = logits_gpu.cpu()
        probs  = torch.sigmoid(logits).numpy()
        preds  = (probs >= 0.5).astype(int)

    y_true = y_true_tensor.numpy().astype(int)
    acc    = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, preds, average=None, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    report = classification_report(
        y_true, preds, target_names=CLASS_NAMES, zero_division=0, output_dict=True
    )
    return {
        "split":     split_name,
        "loss":      round(float(loss), 6),
        "accuracy":  round(float(acc), 6),
        "macro_f1":  round(float(macro_f1), 6),
        "precision_invalid": round(float(prec[0]), 6),
        "recall_invalid":    round(float(rec[0]),  6),
        "f1_invalid":        round(float(f1[0]),   6),
        "precision_valid":   round(float(prec[1]), 6),
        "recall_valid":      round(float(rec[1]),  6),
        "f1_valid":          round(float(f1[1]),   6),
        "confusion_matrix":  cm.tolist(),
        "classification_report": report,
        "predictions":       preds.tolist(),
        "probabilities":     probs.tolist(),
    }

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train_model(name: str, model_cls) -> dict:
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    set_seed(SEED)
    model = model_cls().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    wd = WEIGHT_DECAY if name == "reg_mlp" else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=wd)

    artifact_dir = OUT_DIR / name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = artifact_dir / "best.pt"

    best_macro_f1   = -1.0
    best_epoch      = 0
    patience_count  = 0
    history         = []

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train pass
        model.train()
        train_loss_sum, n_correct, n_total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * X_b.size(0)
            preds_b = (torch.sigmoid(logits) >= 0.5).long()
            n_correct += (preds_b == y_b.long()).sum().item()
            n_total   += X_b.size(0)

        train_loss = train_loss_sum / n_total
        train_acc  = n_correct / n_total

        # ── val pass
        val_metrics = evaluate(model, X_val, y_val, "val")
        val_macro_f1 = val_metrics["macro_f1"]
        val_loss     = val_metrics["loss"]
        val_acc      = val_metrics["accuracy"]

        history.append({
            "epoch":       epoch,
            "train_loss":  round(train_loss, 6),
            "train_acc":   round(train_acc,  6),
            "val_loss":    val_loss,
            "val_acc":     val_acc,
            "val_macro_f1": val_macro_f1,
        })

        # Print every 10 epochs + first/last
        if epoch == 1 or epoch % 10 == 0:
            print(f"  Ep{epoch:03d}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                  f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_mF1={val_macro_f1:.4f}")

        # ── early stopping / checkpoint
        if val_macro_f1 > best_macro_f1:
            best_macro_f1  = val_macro_f1
            best_epoch     = epoch
            patience_count = 0
            torch.save(model.state_dict(), str(ckpt_path))
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (best epoch {best_epoch}, best val mF1={best_macro_f1:.4f})")
                break

    print(f"  Best epoch: {best_epoch}  |  Best val macro-F1: {best_macro_f1:.4f}")

    # ── Load best checkpoint for final evaluation
    model.load_state_dict(torch.load(str(ckpt_path), weights_only=True))

    val_results  = evaluate(model, X_val,   y_val,   "val")
    test_results = evaluate(model, X_test,  y_test,  "test")

    # ── Print results
    for res in [val_results, test_results]:
        print(f"\n  [{res['split']}]")
        print(f"    accuracy   : {res['accuracy']:.4f}")
        print(f"    macro_f1   : {res['macro_f1']:.4f}")
        print(f"    invalid  P/R/F1 : {res['precision_invalid']:.4f} / {res['recall_invalid']:.4f} / {res['f1_invalid']:.4f}")
        print(f"    valid    P/R/F1 : {res['precision_valid']:.4f} / {res['recall_valid']:.4f} / {res['f1_valid']:.4f}")
        cm = res['confusion_matrix']
        print(f"    confusion matrix (rows=true, cols=pred):")
        print(f"                  pred_invalid  pred_valid")
        print(f"      true_invalid    {cm[0][0]:5d}        {cm[0][1]:5d}")
        print(f"      true_valid      {cm[1][0]:5d}        {cm[1][1]:5d}")

    # ── Save artifacts
    config = {
        "model_name":   name,
        "architecture": str(model),
        "n_params":     n_params,
        "seed":         SEED,
        "batch_size":   BATCH_SIZE,
        "max_epochs":   MAX_EPOCHS,
        "lr":           LR,
        "weight_decay": wd,
        "dropout":      DROPOUT if name == "reg_mlp" else 0.0,
        "patience":     PATIENCE,
        "early_stop_metric": "val_macro_f1",
        "pos_weight":   float(pos_weight.item()),
        "best_epoch":   best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "clip_model":   "ViT-B/32",
        "emb_dim":      EMB_DIM,
    }

    with open(artifact_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(artifact_dir / "val_results.json", "w") as f:
        json.dump(val_results, f, indent=2)
    with open(artifact_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # Save training history as CSV
    import csv
    with open(artifact_dir / "training_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"\n  Artifacts saved → {artifact_dir}")
    return {
        "name":          name,
        "best_epoch":    best_epoch,
        "val_results":   val_results,
        "test_results":  test_results,
        "config":        config,
    }

# ── RUN ALL THREE ─────────────────────────────────────────────────────────────
all_results = {}
for model_name, model_cls in MODELS.items():
    all_results[model_name] = train_model(model_name, model_cls)

# ── COMPARISON TABLE ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("COMPARISON TABLE")
print(f"{'='*70}")
print(f"{'Model':<14} | {'Val Acc':>8} {'Val mF1':>8} | {'Test Acc':>9} {'Test mF1':>9}")
print("-" * 60)
for name, res in all_results.items():
    v = res["val_results"]
    t = res["test_results"]
    print(f"{name:<14} | {v['accuracy']:>8.4f} {v['macro_f1']:>8.4f} | {t['accuracy']:>9.4f} {t['macro_f1']:>9.4f}")

# Select best model by val macro-F1
best_name = max(all_results, key=lambda k: all_results[k]["val_results"]["macro_f1"])
print(f"\nBest model (val macro-F1): {best_name}")
best = all_results[best_name]
print(f"  Val  accuracy={best['val_results']['accuracy']:.4f}  macro_f1={best['val_results']['macro_f1']:.4f}")
print(f"  Test accuracy={best['test_results']['accuracy']:.4f}  macro_f1={best['test_results']['macro_f1']:.4f}")

# ── SAVE COMPARISON SUMMARY ───────────────────────────────────────────────────
summary = {
    "seed":        SEED,
    "clip_model":  "ViT-B/32",
    "emb_dim":     EMB_DIM,
    "best_model":  best_name,
    "models": {
        name: {
            "best_epoch":     res["config"]["best_epoch"],
            "val_accuracy":   res["val_results"]["accuracy"],
            "val_macro_f1":   res["val_results"]["macro_f1"],
            "val_loss":       res["val_results"]["loss"],
            "val_precision_invalid": res["val_results"]["precision_invalid"],
            "val_recall_invalid":    res["val_results"]["recall_invalid"],
            "val_f1_invalid":        res["val_results"]["f1_invalid"],
            "val_precision_valid":   res["val_results"]["precision_valid"],
            "val_recall_valid":      res["val_results"]["recall_valid"],
            "val_f1_valid":          res["val_results"]["f1_valid"],
            "val_confusion_matrix":  res["val_results"]["confusion_matrix"],
            "test_accuracy":  res["test_results"]["accuracy"],
            "test_macro_f1":  res["test_results"]["macro_f1"],
            "test_loss":      res["test_results"]["loss"],
            "test_precision_invalid": res["test_results"]["precision_invalid"],
            "test_recall_invalid":    res["test_results"]["recall_invalid"],
            "test_f1_invalid":        res["test_results"]["f1_invalid"],
            "test_precision_valid":   res["test_results"]["precision_valid"],
            "test_recall_valid":      res["test_results"]["recall_valid"],
            "test_f1_valid":          res["test_results"]["f1_valid"],
            "test_confusion_matrix":  res["test_results"]["confusion_matrix"],
        }
        for name, res in all_results.items()
    }
}

with open(OUT_DIR / "comparison_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nComparison summary → {OUT_DIR / 'comparison_summary.json'}")
print("Done.")
