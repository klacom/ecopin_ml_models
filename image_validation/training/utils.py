import os
import json
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np


def compute_class_weights(csv_path: str) -> torch.Tensor:
    """Compute class weights inversely proportional to class frequencies.
    Returns a tensor suitable for torch.nn.CrossEntropyLoss(weight=...).
    """
    df = pd.read_csv(csv_path, dtype=str)
    label_counts = df['primary_label'].value_counts().sort_index()
    total = label_counts.sum()
    # Inverse frequency weighting
    weights = total / (len(label_counts) * label_counts)
    return torch.tensor(weights.values, dtype=torch.float)


def plot_training_curves(history: pd.DataFrame, output_path: str):
    """Plot loss and accuracy curves from training history DataFrame.
    History must contain columns: epoch, train_loss, val_loss, train_acc, val_acc.
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['train_acc'], label='Train Acc')
    plt.plot(history['epoch'], history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str):
    """Save a confusion matrix heatmap using matplotlib.
    """
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Validation Confusion Matrix')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_classification_report(y_true, y_pred, class_names, output_path: str):
    """Generate sklearn classification report and save as JSON."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    return report


def save_training_config(config: dict, output_path: str):
    """Save training configuration as JSON for reproducibility."""
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
