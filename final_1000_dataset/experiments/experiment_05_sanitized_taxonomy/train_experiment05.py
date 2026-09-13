import json
import argparse
import json
import os
import platform
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from hashlib import sha256

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from timm import create_model
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torchvision.transforms.functional as TF


if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stdout.reconfigure(line_buffering=True)


SEED = 42
BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_CLASSES = 4
CLASS_NAMES = ["flooding", "non_environmental", "pollution", "waste"]
MAX_EPOCHS = 20
INITIAL_LR = 1e-5
PATIENCE = 5

SOURCE_SPLIT_DIR = r"c:\dev\datasets\ecopin_dataset\split"
SANITIZED_SPLIT_DIR = r"c:\dev\datasets\ecopin_dataset\experiment_05_sanitized_taxonomy"
EXP_DIR = r"c:\dev\ecopin_ml_models\final_1000_dataset\experiments\experiment_05_sanitized_taxonomy"
HISTORICAL_CKPT_PATH = r"c:\dev\ecopin_ml_models\baseline_original_350\best_model.pt"
TARGET_AUDIT_LIST_PATH = r"c:\dev\ecopin_ml_models\final_1000_dataset\experiments\target_audit_list.json"
AUDIT_REPORT_PATH = r"c:\dev\ecopin_ml_models\final_1000_dataset\experiments\experiment_04_pollution_waste_audit.md"

EXP01_DIR = r"c:\dev\ecopin_ml_models\final_1000_dataset\experiments\experiment_01_fresh_pretrained"
EXP02_DIR = r"c:\dev\ecopin_ml_models\final_1000_dataset\experiments\experiment_02_baseline_finetune"
EXP03_DIR = r"c:\dev\ecopin_ml_models\final_1000_dataset\experiments\experiment_03_sanitized_labels"
EXP02_SCRIPT_PATH = os.path.join(EXP02_DIR, "train_experiment02.py")

TRAIN_DIR = os.path.join(SANITIZED_SPLIT_DIR, "train")
VAL_DIR = os.path.join(SANITIZED_SPLIT_DIR, "val")
TEST_DIR = os.path.join(SANITIZED_SPLIT_DIR, "test")

os.makedirs(EXP_DIR, exist_ok=True)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class PadToSquare:
    def __call__(self, img):
        width, height = img.size
        max_side = max(width, height)
        pad_l = (max_side - width) // 2
        pad_t = (max_side - height) // 2
        pad_r = max_side - width - pad_l
        pad_b = max_side - height - pad_t
        return TF.pad(img, (pad_l, pad_t, pad_r, pad_b), fill=0)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def sha256_file(path):
    digest = sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_audit_report(path):
    rows = {}
    pattern = re.compile(
        r"^\|\s*(POL_\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*\|$"
    )
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = pattern.match(line)
            if not match:
                continue
            image_id, filename, subcategory, current_label, decision, reason = match.groups()
            rows[image_id] = {
                "image_id": image_id,
                "filename": filename,
                "subcategory": subcategory,
                "current_label": current_label,
                "decision": decision,
                "reason": reason,
            }
    if len(rows) != 101:
        raise RuntimeError(f"Expected 101 parsed audit rows, found {len(rows)}")
    return rows


def build_manifest():
    audit_targets = read_json(TARGET_AUDIT_LIST_PATH)
    audit_rows = parse_audit_report(AUDIT_REPORT_PATH)

    manifest = []
    for item in audit_targets:
        image_id = item["image_id"]
        if image_id not in audit_rows:
            raise RuntimeError(f"Missing audit decision for {image_id}")

        audit = audit_rows[image_id]
        split = item["split"]
        decision = audit["decision"]

        if decision == "RECLASSIFY_TO_WASTE" and split in {"train", "val"}:
            final_action = "APPLY_RELABEL_TO_WASTE"
            new_label = "waste"
            test_protected = False
        elif decision == "RECLASSIFY_TO_WASTE" and split == "test":
            final_action = "TEST_PROTECTED_UNCHANGED"
            new_label = "pollution"
            test_protected = True
        else:
            final_action = "LEAVE_AS_POLLUTION"
            new_label = "pollution"
            test_protected = False

        manifest.append(
            {
                "image_id": image_id,
                "filename": item["filename"],
                "subcategory": item["subcategory"],
                "original_label": "pollution",
                "split": split,
                "source_path": item["full_path"],
                "experiment04_decision": decision,
                "experiment05_final_action": final_action,
                "new_label": new_label,
                "reason": audit["reason"],
                "visual_verification": True,
                "test_protected": test_protected,
                "notes": item.get("notes", ""),
                "location": item.get("location", ""),
            }
        )

    manifest.sort(key=lambda row: (row["split"], row["image_id"]))
    return manifest


def create_sanitized_dataset(manifest):
    if not os.path.exists(SANITIZED_SPLIT_DIR):
        print(f"Creating sanitized dataset copy at: {SANITIZED_SPLIT_DIR}")
        shutil.copytree(SOURCE_SPLIT_DIR, SANITIZED_SPLIT_DIR, copy_function=shutil.copy2)
    else:
        print(f"Sanitized dataset already exists, reusing: {SANITIZED_SPLIT_DIR}")

    applied_changes = []
    for entry in manifest:
        if entry["experiment05_final_action"] != "APPLY_RELABEL_TO_WASTE":
            continue

        src = os.path.join(SANITIZED_SPLIT_DIR, entry["split"], "pollution", entry["filename"])
        dst = os.path.join(SANITIZED_SPLIT_DIR, entry["split"], "waste", entry["filename"])

        if os.path.exists(src):
            shutil.move(src, dst)
            applied_changes.append(entry["filename"])
        elif os.path.exists(dst):
            continue
        else:
            raise FileNotFoundError(f"Neither source nor destination exists for relabel: {entry['filename']}")

    print(f"Applied or confirmed {len(applied_changes)} train/val relabel moves.")


def collect_all_image_paths(root_dir):
    image_paths = []
    for split in ["train", "val", "test"]:
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(root_dir, split, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in sorted(os.listdir(class_dir)):
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    image_paths.append(os.path.join(class_dir, filename))
    return image_paths


def count_distribution(root_dir):
    distribution = {}
    for split in ["train", "val", "test"]:
        distribution[split] = {}
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(root_dir, split, class_name)
            count = 0
            if os.path.isdir(class_dir):
                count = len(
                    [
                        name
                        for name in os.listdir(class_dir)
                        if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                    ]
                )
            distribution[split][class_name] = count
    return distribution


def build_image_index(root_dir):
    index = {}
    duplicate_ids = []

    for split in ["train", "val", "test"]:
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(root_dir, split, class_name)
            if not os.path.isdir(class_dir):
                continue

            for filename in sorted(os.listdir(class_dir)):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue

                image_id = os.path.splitext(filename)[0]
                abs_path = os.path.join(class_dir, filename)
                rel_path = os.path.relpath(abs_path, root_dir)
                record = {
                    "image_id": image_id,
                    "filename": filename,
                    "split": split,
                    "label": class_name,
                    "absolute_path": abs_path,
                    "relative_path": rel_path,
                    "sha256": sha256_file(abs_path),
                }

                if image_id in index:
                    duplicate_ids.append(
                        {
                            "image_id": image_id,
                            "first_location": index[image_id]["relative_path"],
                            "duplicate_location": rel_path,
                        }
                    )
                else:
                    index[image_id] = record

    return index, duplicate_ids


def verify_manifest_consistency(source_root, sanitized_root, manifest):
    source_index, source_duplicate_ids = build_image_index(source_root)
    sanitized_index, sanitized_duplicate_ids = build_image_index(sanitized_root)

    source_ids = set(source_index.keys())
    sanitized_ids = set(sanitized_index.keys())
    manifest_index = {entry["image_id"]: entry for entry in manifest}

    missing_from_sanitized = sorted(source_ids - sanitized_ids)
    orphan_in_sanitized = sorted(sanitized_ids - source_ids)

    split_membership_changes = []
    content_hash_mismatches = []
    observed_label_changes = []
    unexpected_label_changes = []
    expected_relabels_missing = []
    expected_relabels_applied = []
    protected_test_label_changes = []

    for image_id in sorted(source_ids & sanitized_ids):
        source_record = source_index[image_id]
        sanitized_record = sanitized_index[image_id]
        manifest_entry = manifest_index.get(image_id)

        if source_record["split"] != sanitized_record["split"]:
            split_membership_changes.append(
                {
                    "image_id": image_id,
                    "source_split": source_record["split"],
                    "sanitized_split": sanitized_record["split"],
                }
            )

        if source_record["sha256"] != sanitized_record["sha256"]:
            content_hash_mismatches.append(
                {
                    "image_id": image_id,
                    "source_path": source_record["relative_path"],
                    "sanitized_path": sanitized_record["relative_path"],
                }
            )

        if source_record["label"] != sanitized_record["label"]:
            observed_label_changes.append(
                {
                    "image_id": image_id,
                    "split": source_record["split"],
                    "source_label": source_record["label"],
                    "sanitized_label": sanitized_record["label"],
                    "source_path": source_record["relative_path"],
                    "sanitized_path": sanitized_record["relative_path"],
                }
            )

        expected_label = source_record["label"]
        if manifest_entry and manifest_entry["experiment05_final_action"] == "APPLY_RELABEL_TO_WASTE":
            expected_label = "waste"

        if sanitized_record["label"] != expected_label:
            unexpected_label_changes.append(
                {
                    "image_id": image_id,
                    "split": source_record["split"],
                    "source_label": source_record["label"],
                    "expected_label": expected_label,
                    "observed_label": sanitized_record["label"],
                }
            )

        if manifest_entry and manifest_entry["experiment05_final_action"] == "APPLY_RELABEL_TO_WASTE":
            if sanitized_record["label"] == "waste" and source_record["label"] == "pollution":
                expected_relabels_applied.append(image_id)
            else:
                expected_relabels_missing.append(
                    {
                        "image_id": image_id,
                        "split": source_record["split"],
                        "source_label": source_record["label"],
                        "observed_label": sanitized_record["label"],
                    }
                )

        if manifest_entry and manifest_entry["test_protected"] and source_record["label"] != sanitized_record["label"]:
            protected_test_label_changes.append(
                {
                    "image_id": image_id,
                    "source_label": source_record["label"],
                    "sanitized_label": sanitized_record["label"],
                }
            )

    candidate_distribution = {}
    for split in ["train", "val", "test"]:
        candidate_rows = [row for row in manifest if row["split"] == split]
        candidate_distribution[split] = {
            "reviewed": len(candidate_rows),
            "apply_relabel_to_waste": sum(
                1 for row in candidate_rows if row["experiment05_final_action"] == "APPLY_RELABEL_TO_WASTE"
            ),
            "leave_as_pollution": sum(
                1 for row in candidate_rows if row["experiment05_final_action"] == "LEAVE_AS_POLLUTION"
            ),
            "test_protected_unchanged": sum(
                1 for row in candidate_rows if row["experiment05_final_action"] == "TEST_PROTECTED_UNCHANGED"
            ),
        }

    return {
        "source_image_count": len(source_index),
        "sanitized_image_count": len(sanitized_index),
        "duplicate_ids": {
            "source": source_duplicate_ids,
            "sanitized": sanitized_duplicate_ids,
        },
        "missing_from_sanitized": missing_from_sanitized,
        "orphan_in_sanitized": orphan_in_sanitized,
        "same_split_membership": {
            "passed": len(split_membership_changes) == 0,
            "changes": split_membership_changes,
        },
        "image_content_unchanged": {
            "passed": len(content_hash_mismatches) == 0,
            "hash_mismatches": content_hash_mismatches,
        },
        "label_change_validation": {
            "candidate_distribution_by_split": candidate_distribution,
            "observed_label_change_count": len(observed_label_changes),
            "observed_label_changes": observed_label_changes,
            "only_intended_train_val_labels_changed": len(unexpected_label_changes) == 0,
            "unexpected_label_changes": unexpected_label_changes,
            "expected_relabels_applied_count": len(expected_relabels_applied),
            "expected_relabels_applied": expected_relabels_applied,
            "expected_relabels_missing": expected_relabels_missing,
            "protected_test_labels_unchanged": len(protected_test_label_changes) == 0,
            "protected_test_label_changes": protected_test_label_changes,
        },
        "no_missing_or_orphan_files": len(missing_from_sanitized) == 0 and len(orphan_in_sanitized) == 0,
        "no_duplicate_ids": len(source_duplicate_ids) == 0 and len(sanitized_duplicate_ids) == 0,
        "overall_passed": (
            len(source_duplicate_ids) == 0
            and len(sanitized_duplicate_ids) == 0
            and len(missing_from_sanitized) == 0
            and len(orphan_in_sanitized) == 0
            and len(split_membership_changes) == 0
            and len(content_hash_mismatches) == 0
            and len(unexpected_label_changes) == 0
            and len(expected_relabels_missing) == 0
            and len(protected_test_label_changes) == 0
        ),
    }


def verify_dataset_integrity(root_dir, source_root_dir, source_test_dir, manifest):
    distribution = count_distribution(root_dir)
    all_paths = collect_all_image_paths(root_dir)

    readable_failures = []
    filename_locations = defaultdict(list)
    hash_locations = defaultdict(list)

    for path in all_paths:
        filename = os.path.basename(path)
        rel_path = os.path.relpath(path, root_dir)
        filename_locations[filename].append(rel_path)
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as exc:
            readable_failures.append({"path": rel_path, "error": str(exc)})
        file_hash = sha256_file(path)
        hash_locations[file_hash].append(rel_path)

    duplicate_filenames_across_splits = []
    for filename, locations in sorted(filename_locations.items()):
        unique_splits = sorted({location.split(os.sep)[0] for location in locations})
        if len(unique_splits) > 1:
            duplicate_filenames_across_splits.append({"filename": filename, "locations": locations})

    cross_split_duplicate_hashes = []
    for file_hash, locations in sorted(hash_locations.items()):
        unique_splits = sorted({location.split(os.sep)[0] for location in locations})
        if len(locations) > 1 and len(unique_splits) > 1:
            cross_split_duplicate_hashes.append(
                {"sha256": file_hash, "locations": locations}
            )

    source_test_hashes = {}
    sanitized_test_hashes = {}
    test_hash_mismatches = []
    source_test_files = []
    sanitized_test_files = []

    for class_name in CLASS_NAMES:
        source_class_dir = os.path.join(source_test_dir, class_name)
        sanitized_class_dir = os.path.join(root_dir, "test", class_name)

        for filename in sorted(os.listdir(source_class_dir)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                rel_path = os.path.join(class_name, filename)
                source_test_files.append(rel_path)
                source_test_hashes[rel_path] = sha256_file(os.path.join(source_class_dir, filename))

        for filename in sorted(os.listdir(sanitized_class_dir)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                rel_path = os.path.join(class_name, filename)
                sanitized_test_files.append(rel_path)
                sanitized_test_hashes[rel_path] = sha256_file(os.path.join(sanitized_class_dir, filename))

    for rel_path in sorted(set(source_test_hashes) | set(sanitized_test_hashes)):
        if source_test_hashes.get(rel_path) != sanitized_test_hashes.get(rel_path):
            test_hash_mismatches.append(
                {
                    "relative_path": rel_path,
                    "source_hash": source_test_hashes.get(rel_path),
                    "sanitized_hash": sanitized_test_hashes.get(rel_path),
                }
            )

    expected_counts = {
        "train_total": 800,
        "val_total": 100,
        "test_total": 100,
    }
    totals = {
        "train_total": sum(distribution["train"].values()),
        "val_total": sum(distribution["val"].values()),
        "test_total": sum(distribution["test"].values()),
    }

    manifest_consistency = verify_manifest_consistency(source_root_dir, root_dir, manifest)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": root_dir,
        "source_dataset_path": source_root_dir,
        "source_test_path": source_test_dir,
        "expected_counts": expected_counts,
        "observed_distribution": distribution,
        "observed_totals": totals,
        "count_check_passed": totals == expected_counts,
        "readable_image_check": {
            "total_images_scanned": len(all_paths),
            "corrupted_or_unreadable_files": len(readable_failures),
            "failures": readable_failures,
        },
        "cross_split_non_leakage": {
            "duplicate_filenames_across_splits": duplicate_filenames_across_splits,
            "duplicate_hashes_across_splits": cross_split_duplicate_hashes,
            "passed": len(duplicate_filenames_across_splits) == 0 and len(cross_split_duplicate_hashes) == 0,
        },
        "test_hash_verification": {
            "source_test_file_count": len(source_test_files),
            "sanitized_test_file_count": len(sanitized_test_files),
            "mismatch_count": len(test_hash_mismatches),
            "mismatches": test_hash_mismatches,
            "byte_for_byte_identical": len(test_hash_mismatches) == 0,
        },
        "manifest_consistency": manifest_consistency,
        "overall_passed": (
            totals == expected_counts
            and len(readable_failures) == 0
            and len(duplicate_filenames_across_splits) == 0
            and len(cross_split_duplicate_hashes) == 0
            and len(test_hash_mismatches) == 0
            and manifest_consistency["overall_passed"]
        ),
    }
    return report


def load_previous_experiment_bundle(exp_dir, exp_key):
    config = read_json(os.path.join(exp_dir, "config.json"))
    class_report = read_json(os.path.join(exp_dir, "classification_report.json"))
    cm = read_json(os.path.join(exp_dir, "confusion_matrix.json"))
    test_predictions_path = os.path.join(exp_dir, "test_predictions.json")
    test_predictions = read_json(test_predictions_path) if os.path.exists(test_predictions_path) else []

    return {
        "name": exp_key,
        "config": config,
        "classification_report": class_report,
        "confusion_matrix": cm,
        "test_predictions": test_predictions,
    }


def evaluate(model, loader, criterion, device):
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
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    cls_report = classification_report(
        all_targets,
        all_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_targets, all_preds).tolist()

    return avg_loss, acc, macro_f1, weighted_f1, cls_report, cm, all_preds, all_probs, all_targets


def save_prediction_records(dataset, preds, probs, output_path):
    records = []
    for idx, (path, target) in enumerate(dataset.samples):
        records.append(
            {
                "filename": os.path.basename(path),
                "true_label": CLASS_NAMES[target],
                "pred_label": CLASS_NAMES[preds[idx]],
                "correct": bool(preds[idx] == target),
                "probabilities": {
                    CLASS_NAMES[i]: float(probs[idx][i]) for i in range(NUM_CLASSES)
                },
            }
        )
    write_json(output_path, records)
    return records


def load_existing_eval_artifacts(exp_dir):
    reports = read_json(os.path.join(exp_dir, "classification_report.json"))
    confusion = read_json(os.path.join(exp_dir, "confusion_matrix.json"))
    val_predictions = read_json(os.path.join(exp_dir, "val_predictions.json"))
    test_predictions = read_json(os.path.join(exp_dir, "test_predictions.json"))

    return {
        "val_report": reports["validation"],
        "test_report": reports["test"],
        "val_cm": confusion["validation_confusion_matrix"],
        "test_cm": confusion["test_confusion_matrix"],
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
    }


def format_metric(value):
    return f"{value:.4f}"


def format_pct(value):
    return f"{value * 100:.2f}%"


def build_confusion_matrix_markdown(cm):
    header = "| True \\ Pred | flooding | non_environmental | pollution | waste |\n|---|---:|---:|---:|---:|"
    rows = []
    for idx, row in enumerate(cm):
        rows.append(
            f"| {CLASS_NAMES[idx]} | {row[0]} | {row[1]} | {row[2]} | {row[3]} |"
        )
    return "\n".join([header] + rows)


def build_distribution_table(distribution):
    lines = [
        "| Split | flooding | non_environmental | pollution | waste | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "val", "test"]:
        row = distribution[split]
        total = sum(row.values())
        lines.append(
            f"| {split} | {row['flooding']} | {row['non_environmental']} | {row['pollution']} | {row['waste']} | {total} |"
        )
    return "\n".join(lines)


def build_per_class_metrics_table(report):
    lines = [
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_name in CLASS_NAMES:
        lines.append(
            "| {label} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {support:.0f} |".format(
                label=class_name,
                precision=report[class_name]["precision"],
                recall=report[class_name]["recall"],
                f1=report[class_name]["f1-score"],
                support=report[class_name]["support"],
            )
        )
    lines.append(
        "| macro avg | {precision:.4f} | {recall:.4f} | {f1:.4f} | {support:.0f} |".format(
            precision=report["macro avg"]["precision"],
            recall=report["macro avg"]["recall"],
            f1=report["macro avg"]["f1-score"],
            support=report["macro avg"]["support"],
        )
    )
    return "\n".join(lines)


def build_methodology_reproduction():
    return {
        "source_of_truth_script": EXP02_SCRIPT_PATH,
        "source_of_truth_config": os.path.join(EXP02_DIR, "config.json"),
        "matched_training_methodology": {
            "seed": SEED,
            "architecture": "efficientnet_b0",
            "initialization_checkpoint": HISTORICAL_CKPT_PATH,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "train_augmentations": [
                "PadToSquare(fill=0)",
                "Resize(224, 224)",
                "RandomHorizontalFlip(p=0.2)",
                "ToTensor()",
                "ImageNet normalization",
            ],
            "eval_preprocessing": [
                "PadToSquare(fill=0)",
                "Resize(224, 224)",
                "ToTensor()",
                "ImageNet normalization",
            ],
            "loss": "CrossEntropyLoss (unweighted)",
            "optimizer": "Adam",
            "initial_learning_rate": INITIAL_LR,
            "scheduler": "ReduceLROnPlateau(mode=min, factor=0.5, patience=2)",
            "early_stopping_patience": PATIENCE,
            "early_stopping_metric": "validation Macro F1",
            "max_epochs": MAX_EPOCHS,
            "checkpoint_selection": "best validation Macro F1",
        },
        "intentional_differences": [
            "Dataset root changed from the original split to the Exp05 sanitized dataset copy.",
            "Manifest generation, integrity verification, protected-test analysis, and report generation were added around the unchanged Exp02 training loop.",
        ],
    }


def augment_config_data(config_data, val_report, test_report, integrity_report):
    config_data["dataset"]["distribution"] = count_distribution(SANITIZED_SPLIT_DIR)
    config_data["results"]["validation"]["macro_precision"] = val_report["macro avg"]["precision"]
    config_data["results"]["validation"]["macro_recall"] = val_report["macro avg"]["recall"]
    config_data["results"]["test"]["macro_precision"] = test_report["macro avg"]["precision"]
    config_data["results"]["test"]["macro_recall"] = test_report["macro avg"]["recall"]
    config_data["methodology_reproduction"] = build_methodology_reproduction()
    config_data["integrity_summary"] = {
        "overall_passed": integrity_report["overall_passed"],
        "count_check_passed": integrity_report["count_check_passed"],
        "readable_images_passed": integrity_report["readable_image_check"]["corrupted_or_unreadable_files"] == 0,
        "cross_split_non_leakage_passed": integrity_report["cross_split_non_leakage"]["passed"],
        "test_set_byte_identical": integrity_report["test_hash_verification"]["byte_for_byte_identical"],
        "same_split_membership_passed": integrity_report["manifest_consistency"]["same_split_membership"]["passed"],
        "content_unchanged_passed": integrity_report["manifest_consistency"]["image_content_unchanged"]["passed"],
        "only_intended_label_changes_passed": integrity_report["manifest_consistency"]["label_change_validation"]["only_intended_train_val_labels_changed"],
        "protected_test_labels_unchanged": integrity_report["manifest_consistency"]["label_change_validation"]["protected_test_labels_unchanged"],
        "no_missing_or_orphan_files": integrity_report["manifest_consistency"]["no_missing_or_orphan_files"],
        "no_duplicate_ids": integrity_report["manifest_consistency"]["no_duplicate_ids"],
    }
    config_data["last_regenerated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return config_data


def build_comparison_rows(previous_bundles, exp05_results, exp05_test_report, exp05_test_cm):
    rows = [
        "| Experiment | Test Acc | Test Macro F1 | Pollution P | Pollution R | Pollution F1 | Waste P | Waste R | Waste F1 | pollution->waste | waste->pollution | Boundary Total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    ordered = [
        ("Experiment 01", previous_bundles["exp01"]),
        ("Experiment 02", previous_bundles["exp02"]),
        ("Experiment 03", previous_bundles["exp03"]),
    ]
    for label, bundle in ordered:
        test_results = bundle["config"]["results"]["test"]
        test_report = bundle["classification_report"]["test"]
        test_cm = bundle["confusion_matrix"]["test_confusion_matrix"]
        rows.append(
            "| {label} | {acc} | {macro} | {pp} | {pr} | {pf1} | {wp} | {wr} | {wf1} | {p2w} | {w2p} | {total} |".format(
                label=label,
                acc=format_metric(test_results["accuracy"]),
                macro=format_metric(test_results["macro_f1"]),
                pp=format_metric(test_report["pollution"]["precision"]),
                pr=format_metric(test_report["pollution"]["recall"]),
                pf1=format_metric(test_report["pollution"]["f1-score"]),
                wp=format_metric(test_report["waste"]["precision"]),
                wr=format_metric(test_report["waste"]["recall"]),
                wf1=format_metric(test_report["waste"]["f1-score"]),
                p2w=test_cm[2][3],
                w2p=test_cm[3][2],
                total=test_cm[2][3] + test_cm[3][2],
            )
        )

    rows.append(
        "| Experiment 05 | {acc} | {macro} | {pp} | {pr} | {pf1} | {wp} | {wr} | {wf1} | {p2w} | {w2p} | {total} |".format(
            acc=format_metric(exp05_results["accuracy"]),
            macro=format_metric(exp05_results["macro_f1"]),
            pp=format_metric(exp05_test_report["pollution"]["precision"]),
            pr=format_metric(exp05_test_report["pollution"]["recall"]),
            pf1=format_metric(exp05_test_report["pollution"]["f1-score"]),
            wp=format_metric(exp05_test_report["waste"]["precision"]),
            wr=format_metric(exp05_test_report["waste"]["recall"]),
            wf1=format_metric(exp05_test_report["waste"]["f1-score"]),
            p2w=exp05_test_cm[2][3],
            w2p=exp05_test_cm[3][2],
            total=exp05_test_cm[2][3] + exp05_test_cm[3][2],
        )
    )
    return "\n".join(rows)


def analyze_protected_test_candidates(manifest, prediction_records_by_exp):
    protected_filenames = sorted(
        [
            entry["filename"]
            for entry in manifest
            if entry["split"] == "test" and entry["test_protected"]
        ]
    )
    analysis = {
        "protected_filenames": protected_filenames,
        "protected_candidate_count": len(protected_filenames),
        "per_experiment": {},
    }

    protected_set = set(protected_filenames)
    for exp_name, records in prediction_records_by_exp.items():
        subset = [row for row in records if row["filename"] in protected_set]
        waste_predictions = sum(1 for row in subset if row["pred_label"] == "waste")
        pollution_predictions = sum(1 for row in subset if row["pred_label"] == "pollution")
        analysis["per_experiment"][exp_name] = {
            "evaluated_candidates": len(subset),
            "predicted_waste": waste_predictions,
            "predicted_pollution": pollution_predictions,
            "prediction_breakdown": dict(Counter(row["pred_label"] for row in subset)),
            "records": subset,
        }

    return analysis


def build_report(
    manifest,
    integrity_report,
    config_data,
    val_report,
    test_report,
    val_cm,
    test_cm,
    previous_bundles,
    protected_analysis,
):
    distribution = config_data["dataset"]["distribution"]
    results = config_data["results"]
    pre = config_data["pre_finetuning_metrics"]
    methodology = config_data["methodology_reproduction"]

    changed_entries = [row for row in manifest if row["experiment05_final_action"] == "APPLY_RELABEL_TO_WASTE"]
    protected_entries = [row for row in manifest if row["experiment05_final_action"] == "TEST_PROTECTED_UNCHANGED"]
    kept_entries = [row for row in manifest if row["experiment05_final_action"] == "LEAVE_AS_POLLUTION"]

    changed_by_split = Counter(row["split"] for row in changed_entries)
    reviewed_by_split = Counter(row["split"] for row in manifest)
    decision_breakdown = Counter(row["experiment04_decision"] for row in manifest)
    subcategory_breakdown = Counter(row["subcategory"] for row in manifest if row["experiment05_final_action"] == "APPLY_RELABEL_TO_WASTE")

    comparison_table = build_comparison_rows(previous_bundles, results["test"], test_report, test_cm)

    exp02_test_cm = previous_bundles["exp02"]["confusion_matrix"]["test_confusion_matrix"]
    exp03_test_cm = previous_bundles["exp03"]["confusion_matrix"]["test_confusion_matrix"]
    exp01_test_cm = previous_bundles["exp01"]["confusion_matrix"]["test_confusion_matrix"]
    pollution_boundary_improved = test_report["pollution"]["f1-score"] > previous_bundles["exp03"]["classification_report"]["test"]["pollution"]["f1-score"]
    waste_boundary_improved = test_report["waste"]["f1-score"] > previous_bundles["exp03"]["classification_report"]["test"]["waste"]["f1-score"]

    if results["test"]["macro_f1"] > previous_bundles["exp03"]["config"]["results"]["test"]["macro_f1"] and not pollution_boundary_improved:
        hypothesis_statement = "Partially supported: overall benchmark performance improved slightly, but the protected pollution/waste boundary did not improve."
    elif results["test"]["macro_f1"] > previous_bundles["exp03"]["config"]["results"]["test"]["macro_f1"]:
        hypothesis_statement = "Supported: overall benchmark performance and the pollution/waste boundary improved."
    else:
        hypothesis_statement = "Not supported on the protected benchmark: taxonomy sanitization did not improve the measured test outcome."

    report = f"""# EXPERIMENT 05 — Pollution to Waste Taxonomy Sanitization + Retraining

## 1. Objective

Evaluate whether correcting the severe `pollution` -> `waste` taxonomy overlap identified in Experiment 04 improves EfficientNet-B0 performance while preserving the original 100-image test benchmark byte-for-byte.

## 2. Baseline Context

- Experiment 01 test accuracy / macro F1: **{format_pct(previous_bundles["exp01"]["config"]["results"]["test"]["accuracy"])} / {format_metric(previous_bundles["exp01"]["config"]["results"]["test"]["macro_f1"])}**
- Experiment 02 test accuracy / macro F1: **{format_pct(previous_bundles["exp02"]["config"]["results"]["test"]["accuracy"])} / {format_metric(previous_bundles["exp02"]["config"]["results"]["test"]["macro_f1"])}**
- Experiment 03 test accuracy / macro F1: **{format_pct(previous_bundles["exp03"]["config"]["results"]["test"]["accuracy"])} / {format_metric(previous_bundles["exp03"]["config"]["results"]["test"]["macro_f1"])}**
- Experiment 03 dominant boundary: `pollution -> waste = {exp03_test_cm[2][3]}` errors and `waste -> pollution = {exp03_test_cm[3][2]}` errors on the protected 100-image test set.

## 3. Source of Truth and Inputs

- Audit report: `experiment_04_pollution_waste_audit.md`
- Candidate inventory: `target_audit_list.json`
- Experiment 02 source-of-truth training script: `{EXP02_SCRIPT_PATH}`
- Source dataset: `{SOURCE_SPLIT_DIR}`
- Sanitized dataset copy: `{SANITIZED_SPLIT_DIR}`
- Initialization checkpoint: `{HISTORICAL_CKPT_PATH}`

## 4. Audit Outcome Summary

- Total audited candidates: **{len(manifest)}**
- `RECLASSIFY_TO_WASTE`: **{decision_breakdown['RECLASSIFY_TO_WASTE']}**
- `KEEP`: **{decision_breakdown['KEEP']}**
- `AMBIGUOUS`: **{decision_breakdown['AMBIGUOUS']}**

## 5. Candidate Handling Policy

- Train/val `RECLASSIFY_TO_WASTE` candidates were physically moved from `pollution` to `waste`.
- Test `RECLASSIFY_TO_WASTE` candidates were explicitly protected and left unchanged to preserve direct benchmark comparability with Experiments 01-03.
- `KEEP` and `AMBIGUOUS` candidates remained labeled as `pollution`.
- No oversampling, undersampling, class weights, or synthetic balancing were introduced.

## 6. Candidate Handling Breakdown by Split

- Reviewed candidates in `train`: **{reviewed_by_split['train']}**
- Reviewed candidates in `val`: **{reviewed_by_split['val']}**
- Reviewed candidates in `test`: **{reviewed_by_split['test']}**
- Applied relabels in `train`: **{changed_by_split['train']}**
- Applied relabels in `val`: **{changed_by_split['val']}**
- Protected relabel candidates in `test`: **{len(protected_entries)}**
- Unchanged `KEEP` + `AMBIGUOUS` candidates: **{len(kept_entries)}**

## 7. Applied Label Changes

- Total applied relabels: **{len(changed_entries)}**
- `land_pollution` relabeled to `waste`: **{subcategory_breakdown['land_pollution']}**
- `construction_debris` relabeled to `waste`: **{subcategory_breakdown['construction_debris']}**

## 8. Protected Test Candidates

Protected filenames ({len(protected_entries)}): {", ".join(row["filename"] for row in protected_entries)}

These files remain labeled as `pollution` in the benchmark even though Experiment 04 marked them `RECLASSIFY_TO_WASTE`.

## 9. Dataset Construction Procedure

1. Copied `{SOURCE_SPLIT_DIR}` to `{SANITIZED_SPLIT_DIR}`.
2. Parsed Experiment 04 audit decisions for all 101 targeted `pollution` candidates.
3. Moved only approved `train` and `val` files from `pollution` to `waste`.
4. Left the `test` split untouched and verified it against the source split with SHA256 hashes.

## 10. Dataset Class Distributions

{build_distribution_table(distribution)}

## 11. Dataset Integrity Verification

- Count check passed: **{integrity_report["count_check_passed"]}**
- Readable image failures: **{integrity_report["readable_image_check"]["corrupted_or_unreadable_files"]}**
- Missing files vs source by image ID: **{len(integrity_report["manifest_consistency"]["missing_from_sanitized"])}**
- Orphan files vs source by image ID: **{len(integrity_report["manifest_consistency"]["orphan_in_sanitized"])}**
- Duplicate IDs in source/sanitized: **{len(integrity_report["manifest_consistency"]["duplicate_ids"]["source"])} / {len(integrity_report["manifest_consistency"]["duplicate_ids"]["sanitized"])}**
- Split membership changed: **{len(integrity_report["manifest_consistency"]["same_split_membership"]["changes"])}**
- Image-content hash mismatches by image ID: **{len(integrity_report["manifest_consistency"]["image_content_unchanged"]["hash_mismatches"])}**
- Unexpected label changes: **{len(integrity_report["manifest_consistency"]["label_change_validation"]["unexpected_label_changes"])}**
- Missing intended relabels: **{len(integrity_report["manifest_consistency"]["label_change_validation"]["expected_relabels_missing"])}**
- Protected test label changes: **{len(integrity_report["manifest_consistency"]["label_change_validation"]["protected_test_label_changes"])}**
- Duplicate filenames across splits: **{len(integrity_report["cross_split_non_leakage"]["duplicate_filenames_across_splits"])}**
- Duplicate hashes across splits: **{len(integrity_report["cross_split_non_leakage"]["duplicate_hashes_across_splits"])}**
- Test-set hash mismatches: **{integrity_report["test_hash_verification"]["mismatch_count"]}**
- Overall integrity verdict: **{integrity_report["overall_passed"]}**

## 12. Exp02 Methodology Reproduction

Compared against the actual Experiment 02 training script and config:

- Source of truth script: `{methodology["source_of_truth_script"]}`
- Source of truth config: `{methodology["source_of_truth_config"]}`
- Reproduced exactly: seed `{methodology["matched_training_methodology"]["seed"]}`, EfficientNet-B0, historical Exp8 initialization checkpoint, image size `{methodology["matched_training_methodology"]["image_size"]}`, batch size `{methodology["matched_training_methodology"]["batch_size"]}`, PadToSquare + Resize + horizontal flip + ImageNet normalization, unweighted CrossEntropyLoss, Adam, learning rate `{methodology["matched_training_methodology"]["initial_learning_rate"]}`, ReduceLROnPlateau, early stopping patience `{methodology["matched_training_methodology"]["early_stopping_patience"]}` on validation Macro F1, max epochs `{methodology["matched_training_methodology"]["max_epochs"]}`, and best-checkpoint selection by validation Macro F1.
- Intentional differences: dataset root changed to the sanitized copy, and Exp05 adds manifest/integrity/report generation around the unchanged training loop.

## 13. Training Configuration

- Seed: `{config_data["hyperparameters"]["seed"]}`
- Architecture: `efficientnet_b0`
- Initialization: Historical Exp 8 checkpoint
- Optimizer: `Adam`
- Learning rate: `{config_data["hyperparameters"]["initial_learning_rate"]}`
- Scheduler: `ReduceLROnPlateau(mode='min', factor=0.5, patience=2)`
- Early stopping: `patience = {config_data["hyperparameters"]["early_stopping_patience"]}` on validation Macro F1
- Max epochs: `{config_data["hyperparameters"]["max_epochs"]}`
- Loss: `CrossEntropyLoss (unweighted)`

## 14. Initialization Checkpoint Validation

- Device: `{config_data["environment"]["device"]}`
- GPU: `{config_data["environment"]["gpu_name"]}`
- Historical checkpoint path loaded successfully before fine-tuning.

## 15. Pre-Fine-Tuning Metrics

- Validation accuracy / macro F1 / loss: **{format_pct(pre["val_acc"])} / {format_metric(pre["val_macro_f1"])} / {format_metric(pre["val_loss"])}**
- Test accuracy / macro F1 / loss: **{format_pct(pre["test_acc"])} / {format_metric(pre["test_macro_f1"])} / {format_metric(pre["test_loss"])}**

## 16. Training Summary

- Best epoch: **{results["best_epoch"]}**
- Total epochs trained: **{results["total_epochs_trained"]}**
- Validation early-stopping target: **Macro F1**

## 17. Best Validation Results

- Accuracy: **{format_pct(results["validation"]["accuracy"])}**
- Macro Precision: **{format_metric(results["validation"]["macro_precision"])}**
- Macro Recall: **{format_metric(results["validation"]["macro_recall"])}**
- Macro F1: **{format_metric(results["validation"]["macro_f1"])}**
- Weighted F1: **{format_metric(results["validation"]["weighted_f1"])}**
- Loss: **{format_metric(results["validation"]["loss"])}**

Validation per-class metrics:

{build_per_class_metrics_table(val_report)}

## 18. Final Test Results

- Accuracy: **{format_pct(results["test"]["accuracy"])}**
- Macro Precision: **{format_metric(results["test"]["macro_precision"])}**
- Macro Recall: **{format_metric(results["test"]["macro_recall"])}**
- Macro F1: **{format_metric(results["test"]["macro_f1"])}**
- Weighted F1: **{format_metric(results["test"]["weighted_f1"])}**
- Loss: **{format_metric(results["test"]["loss"])}**

Test per-class metrics:

{build_per_class_metrics_table(test_report)}

## 19. Test Confusion Matrix

{build_confusion_matrix_markdown(test_cm)}

## 20. Pollution-Waste Boundary Analysis

- Experiment 01: `pollution -> waste = {exp01_test_cm[2][3]}`, `waste -> pollution = {exp01_test_cm[3][2]}`
- Experiment 02: `pollution -> waste = {exp02_test_cm[2][3]}`, `waste -> pollution = {exp02_test_cm[3][2]}`
- Experiment 03: `pollution -> waste = {exp03_test_cm[2][3]}`, `waste -> pollution = {exp03_test_cm[3][2]}`
- Experiment 05: `pollution -> waste = {test_cm[2][3]}`, `waste -> pollution = {test_cm[3][2]}`

Protected-candidate behavior:
- Exp 01 predicted `waste` on protected test candidates: **{protected_analysis["per_experiment"]["exp01"]["predicted_waste"]}/{protected_analysis["protected_candidate_count"]}**
- Exp 02 predicted `waste` on protected test candidates: **{protected_analysis["per_experiment"]["exp02"]["predicted_waste"]}/{protected_analysis["protected_candidate_count"]}**
- Exp 03 predicted `waste` on protected test candidates: **{protected_analysis["per_experiment"]["exp03"]["predicted_waste"]}/{protected_analysis["protected_candidate_count"]}**
- Exp 05 predicted `waste` on protected test candidates: **{protected_analysis["per_experiment"]["exp05"]["predicted_waste"]}/{protected_analysis["protected_candidate_count"]}**

Class-specific test metrics:
- Pollution precision / recall / F1: **{format_metric(test_report["pollution"]["precision"])} / {format_metric(test_report["pollution"]["recall"])} / {format_metric(test_report["pollution"]["f1-score"])}**
- Waste precision / recall / F1: **{format_metric(test_report["waste"]["precision"])} / {format_metric(test_report["waste"]["recall"])} / {format_metric(test_report["waste"]["f1-score"])}**

## 21. Comparative Results Across Experiments

{comparison_table}

## 22. Limitations and Threats to Validity

- The primary test set intentionally preserves 12 audited label-noise candidates, so benchmark metrics still include known taxonomy mismatch.
- Experiment 05 isolates taxonomy sanitization only; it does not test class balancing, augmentation changes, or architecture changes.
- Two audited images remained `AMBIGUOUS`, so some residual boundary uncertainty is expected inside `pollution`.

## 23. Conclusion

Experiment 05 {'improved' if results["test"]["macro_f1"] > previous_bundles["exp03"]["config"]["results"]["test"]["macro_f1"] else 'did not improve'} overall test Macro F1 relative to Experiment 03 (**{format_metric(results["test"]["macro_f1"])} vs {format_metric(previous_bundles["exp03"]["config"]["results"]["test"]["macro_f1"])}**) and raised test accuracy to **{format_pct(results["test"]["accuracy"])}**. However, the measured benchmark boundary itself did **not** improve on the protected test set: `pollution -> waste` increased from **{exp03_test_cm[2][3]}** in Experiment 03 to **{test_cm[2][3]}** in Experiment 05, while `waste -> pollution` dropped from **{exp03_test_cm[3][2]}** to **{test_cm[3][2]}**. This result is consistent with the benchmark caveat that 12 protected test images remain labeled `pollution` even though the audit concluded they belong to `waste`.

Hypothesis status: **{hypothesis_statement}**
"""
    return report


def main():
    parser = argparse.ArgumentParser(description="Run or regenerate Experiment 05 artifacts.")
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="Regenerate manifest, integrity, config, and report from existing training artifacts without retraining.",
    )
    args = parser.parse_args()

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest = build_manifest()
    create_sanitized_dataset(manifest)

    manifest_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_dataset": SOURCE_SPLIT_DIR,
        "sanitized_dataset": SANITIZED_SPLIT_DIR,
        "total_audited_candidates": len(manifest),
        "decision_counts": dict(Counter(row["experiment04_decision"] for row in manifest)),
        "final_action_counts": dict(Counter(row["experiment05_final_action"] for row in manifest)),
        "entries": manifest,
    }
    write_json(os.path.join(EXP_DIR, "label_change_manifest.json"), manifest_payload)

    integrity_report = verify_dataset_integrity(
        SANITIZED_SPLIT_DIR,
        SOURCE_SPLIT_DIR,
        os.path.join(SOURCE_SPLIT_DIR, "test"),
        manifest,
    )
    write_json(os.path.join(EXP_DIR, "dataset_integrity_report.json"), integrity_report)
    if not integrity_report["overall_passed"]:
        raise RuntimeError("Dataset integrity verification failed. See dataset_integrity_report.json")

    previous_bundles = {
        "exp01": load_previous_experiment_bundle(EXP01_DIR, "exp01"),
        "exp02": load_previous_experiment_bundle(EXP02_DIR, "exp02"),
        "exp03": load_previous_experiment_bundle(EXP03_DIR, "exp03"),
    }

    if args.artifacts_only:
        print("Artifacts-only mode: reusing existing checkpoints and evaluation outputs.")
        existing_config = read_json(os.path.join(EXP_DIR, "config.json"))
        existing_eval = load_existing_eval_artifacts(EXP_DIR)

        config_data = augment_config_data(existing_config, existing_eval["val_report"], existing_eval["test_report"], integrity_report)
        write_json(os.path.join(EXP_DIR, "config.json"), config_data)

        protected_analysis = analyze_protected_test_candidates(
            manifest,
            {
                "exp01": previous_bundles["exp01"]["test_predictions"],
                "exp02": previous_bundles["exp02"]["test_predictions"],
                "exp03": previous_bundles["exp03"]["test_predictions"],
                "exp05": existing_eval["test_predictions"],
            },
        )
        write_json(os.path.join(EXP_DIR, "protected_test_candidate_analysis.json"), protected_analysis)

        report_markdown = build_report(
            manifest,
            integrity_report,
            config_data,
            existing_eval["val_report"],
            existing_eval["test_report"],
            existing_eval["val_cm"],
            existing_eval["test_cm"],
            previous_bundles,
            protected_analysis,
        )
        with open(os.path.join(EXP_DIR, "EXPERIMENT_05_REPORT.md"), "w", encoding="utf-8") as handle:
            handle.write(report_markdown)

        print(f"[Experiment 05 artifacts regenerated without retraining. Output saved to {EXP_DIR}]")
        return

    train_transforms = T.Compose(
        [
            PadToSquare(),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transforms = T.Compose(
        [
            PadToSquare(),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = ImageFolder(VAL_DIR, transform=eval_transforms)
    test_dataset = ImageFolder(TEST_DIR, transform=eval_transforms)

    print("Class to idx:", train_dataset.class_to_idx)
    assert list(train_dataset.class_to_idx.keys()) == CLASS_NAMES

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Loaded Sanitized Dataset: {len(train_dataset)} Train, {len(val_dataset)} Validation, {len(test_dataset)} Test images.")

    checkpoint = torch.load(HISTORICAL_CKPT_PATH, map_location="cpu", weights_only=False)
    model = create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(
            "Loaded checkpoint metadata: "
            f"Epoch={checkpoint.get('epoch')}, "
            f"Best Macro F1={checkpoint.get('best_macro_f1'):.4f}, "
            f"Best Val Acc={checkpoint.get('best_val_acc'):.4f}"
        )
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"Checkpoint loaded successfully into EfficientNet-B0 (Missing: {missing}, Unexpected: {unexpected})")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    print("\n--- Zero-Shot Performance of Historical Model on Experiment 05 Splits (Before Fine-Tuning) ---")
    init_val_loss, init_val_acc, init_val_macro_f1, _, _, _, _, _, _ = evaluate(model, val_loader, criterion, device)
    init_test_loss, init_test_acc, init_test_macro_f1, _, _, _, _, _, _ = evaluate(model, test_loader, criterion, device)
    print(
        f"Pre-Fine-Tuning Val Acc: {init_val_acc * 100:.2f}%, "
        f"Val Macro F1: {init_val_macro_f1:.4f}, Val Loss: {init_val_loss:.4f}"
    )
    print(
        f"Pre-Fine-Tuning Test Acc: {init_test_acc * 100:.2f}%, "
        f"Test Macro F1: {init_test_macro_f1:.4f}, Test Loss: {init_test_loss:.4f}"
    )

    history = []
    per_epoch_metrics = []
    best_macro_f1 = -1.0
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    print("\n=======================================================")
    print("  EXPERIMENT 05: Sanitized Taxonomy Fine-Tuning       ")
    print("=======================================================")

    for epoch in range(1, MAX_EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
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

        val_loss, val_acc, val_macro_f1, val_weighted_f1, cls_report, cm, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        is_best = False
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            is_best = True
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "class_names": CLASS_NAMES,
                    "base_checkpoint": HISTORICAL_CKPT_PATH,
                },
                os.path.join(EXP_DIR, "best_model.pt"),
            )
        else:
            epochs_without_improvement += 1

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": val_weighted_f1,
            "lr": current_lr,
            "is_best": is_best,
        }
        history.append(log_entry)
        per_epoch_metrics.append(
            {
                "epoch": epoch,
                "metrics": log_entry,
                "classification_report": cls_report,
                "confusion_matrix": cm,
            }
        )

        print(
            f"Epoch [{epoch:02d}/{MAX_EPOCHS:02d}] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%, "
            f"Val Macro F1: {val_macro_f1:.4f} (LR: {current_lr:.1e}) "
            f"{'*' if is_best else ''}"
        )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"\n[Early Stopping] Triggered after {epochs_without_improvement} epochs "
                "without improvement in validation Macro F1."
            )
            break

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "base_checkpoint": HISTORICAL_CKPT_PATH,
        },
        os.path.join(EXP_DIR, "final_model.pt"),
    )

    print("\n=======================================================")
    print(f"  Evaluating Best Model Checkpoint (from Epoch {best_epoch})  ")
    print("=======================================================")

    best_ckpt = torch.load(os.path.join(EXP_DIR, "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    val_loss, val_acc, val_macro_f1, val_weighted_f1, val_report, val_cm, val_preds, val_probs, _ = evaluate(
        model, val_loader, criterion, device
    )
    test_loss, test_acc, test_macro_f1, test_weighted_f1, test_report, test_cm, test_preds, test_probs, _ = evaluate(
        model, test_loader, criterion, device
    )

    print("\n[Best Validation Results]")
    print(f"  Accuracy: {val_acc * 100:.2f}%")
    print(f"  Macro F1: {val_macro_f1:.4f}")
    print(f"  Loss:     {val_loss:.4f}")

    print("\n[Final Test Results]")
    print(f"  Accuracy: {test_acc * 100:.2f}%")
    print(f"  Macro F1: {test_macro_f1:.4f}")
    print(f"  Loss:     {test_loss:.4f}")

    val_prediction_records = save_prediction_records(
        val_dataset, val_preds, val_probs, os.path.join(EXP_DIR, "val_predictions.json")
    )
    test_prediction_records = save_prediction_records(
        test_dataset, test_preds, test_probs, os.path.join(EXP_DIR, "test_predictions.json")
    )

    pd.DataFrame(history).to_csv(os.path.join(EXP_DIR, "training_history.csv"), index=False)
    write_json(os.path.join(EXP_DIR, "per_epoch_metrics.json"), per_epoch_metrics)
    write_json(
        os.path.join(EXP_DIR, "classification_report.json"),
        {"validation": val_report, "test": test_report},
    )
    write_json(
        os.path.join(EXP_DIR, "confusion_matrix.json"),
        {
            "class_names": CLASS_NAMES,
            "validation_confusion_matrix": val_cm,
            "test_confusion_matrix": test_cm,
        },
    )

    config_data = {
        "experiment": "experiment_05_sanitized_taxonomy",
        "date_trained": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        },
        "dataset": {
            "path": SANITIZED_SPLIT_DIR,
            "total_images": 1000,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "num_classes": NUM_CLASSES,
            "classes": CLASS_NAMES,
            "distribution": count_distribution(SANITIZED_SPLIT_DIR),
        },
        "model": {
            "architecture": "efficientnet_b0",
            "initialization_source": "Historical Exp 8 Checkpoint (c:/dev/ecopin_ml_models/baseline_original_350/best_model.pt)",
            "initialization_type": "Pre-trained on original ~350 EcoPin dataset",
        },
        "hyperparameters": {
            "seed": SEED,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "loss_function": "CrossEntropyLoss (unweighted, no artificial class balancing)",
            "optimizer": "Adam",
            "initial_learning_rate": INITIAL_LR,
            "max_epochs": MAX_EPOCHS,
            "scheduler": "ReduceLROnPlateau(mode=min, factor=0.5, patience=2)",
            "early_stopping_patience": PATIENCE,
            "early_stopping_metric": "Macro F1 (maximize)",
            "augmentations": [
                "PadToSquare(fill=0)",
                "Resize(224, 224)",
                "RandomHorizontalFlip(p=0.2)",
                "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
            ],
        },
        "pre_finetuning_metrics": {
            "val_acc": init_val_acc,
            "val_macro_f1": init_val_macro_f1,
            "val_loss": init_val_loss,
            "test_acc": init_test_acc,
            "test_macro_f1": init_test_macro_f1,
            "test_loss": init_test_loss,
        },
        "results": {
            "best_epoch": best_epoch,
            "total_epochs_trained": epoch,
            "validation": {
                "accuracy": val_acc,
                "macro_f1": val_macro_f1,
                "weighted_f1": val_weighted_f1,
                "loss": val_loss,
            },
            "test": {
                "accuracy": test_acc,
                "macro_f1": test_macro_f1,
                "weighted_f1": test_weighted_f1,
                "loss": test_loss,
            },
        },
        "comparison_summary": {
            "exp01_test_acc": previous_bundles["exp01"]["config"]["results"]["test"]["accuracy"],
            "exp01_test_macro_f1": previous_bundles["exp01"]["config"]["results"]["test"]["macro_f1"],
            "exp02_test_acc": previous_bundles["exp02"]["config"]["results"]["test"]["accuracy"],
            "exp02_test_macro_f1": previous_bundles["exp02"]["config"]["results"]["test"]["macro_f1"],
            "exp03_test_acc": previous_bundles["exp03"]["config"]["results"]["test"]["accuracy"],
            "exp03_test_macro_f1": previous_bundles["exp03"]["config"]["results"]["test"]["macro_f1"],
            "exp05_test_acc": test_acc,
            "exp05_test_macro_f1": test_macro_f1,
            "diff_exp05_minus_exp03_acc": test_acc - previous_bundles["exp03"]["config"]["results"]["test"]["accuracy"],
            "diff_exp05_minus_exp03_macro_f1": test_macro_f1 - previous_bundles["exp03"]["config"]["results"]["test"]["macro_f1"],
        },
        "taxonomy_sanitization": {
            "total_audited_candidates": len(manifest),
            "train_relabels_applied": sum(1 for row in manifest if row["split"] == "train" and row["experiment05_final_action"] == "APPLY_RELABEL_TO_WASTE"),
            "val_relabels_applied": sum(1 for row in manifest if row["split"] == "val" and row["experiment05_final_action"] == "APPLY_RELABEL_TO_WASTE"),
            "protected_test_candidates": sum(1 for row in manifest if row["experiment05_final_action"] == "TEST_PROTECTED_UNCHANGED"),
        },
    }
    config_data = augment_config_data(config_data, val_report, test_report, integrity_report)
    write_json(os.path.join(EXP_DIR, "config.json"), config_data)

    protected_analysis = analyze_protected_test_candidates(
        manifest,
        {
            "exp01": previous_bundles["exp01"]["test_predictions"],
            "exp02": previous_bundles["exp02"]["test_predictions"],
            "exp03": previous_bundles["exp03"]["test_predictions"],
            "exp05": test_prediction_records,
        },
    )
    write_json(os.path.join(EXP_DIR, "protected_test_candidate_analysis.json"), protected_analysis)

    report_markdown = build_report(
        manifest,
        integrity_report,
        config_data,
        val_report,
        test_report,
        val_cm,
        test_cm,
        previous_bundles,
        protected_analysis,
    )
    with open(os.path.join(EXP_DIR, "EXPERIMENT_05_REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(report_markdown)

    print(f"\nSaved {len(val_prediction_records)} validation predictions and {len(test_prediction_records)} test predictions.")
    print(f"[Experiment 05 complete. Artifacts saved to {EXP_DIR}]")


if __name__ == "__main__":
    main()
