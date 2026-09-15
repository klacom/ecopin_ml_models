"""
Experiment 03 — Error Analysis & Threshold Study
=================================================
Answers:
  1. Which VALID images are rejected (FN)?
  2. Which INVALID images are accepted (FP)?
  3. Confidence scores on those mistakes.
  4. Optimal threshold per operational trade-off.
  5. Manual review zone (uncertainty band).
  6. Does Linear remain competitive after threshold tuning?
  7. Does Reg MLP justify its complexity over Small MLP?

Outputs all results to stdout + saves analysis JSON + threshold CSV.
"""

import json
import csv
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    precision_recall_fscore_support, f1_score,
    accuracy_score, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix,
)

# ── PATHS ─────────────────────────────────────────────────────────────────────
ART_DIR  = Path(r"C:\dev\ecopin_ml_models\image_validation\artifacts\exp03_clip_classifiers")
EMB_DIR  = Path(r"C:\dev\datasets\ecopin_dataset\embeddings\exp03")

MODELS   = ["linear", "small_mlp", "reg_mlp"]
SPLITS   = ["val", "test"]

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_results(model, split):
    with open(ART_DIR / model / f"{split}_results.json") as f:
        return json.load(f)

def load_filenames(split):
    d = np.load(str(EMB_DIR / f"clip_vitb32_{split}.npz"), allow_pickle=True)
    return d["filenames"].tolist(), d["label_ints"].tolist()

val_fns,  val_labels  = load_filenames("val")
test_fns, test_labels = load_filenames("test")

FN_DATA = {"val": (val_fns, val_labels), "test": (test_fns, test_labels)}

# ── THRESHOLD SWEEP ───────────────────────────────────────────────────────────
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.01), 2).tolist()
REVIEW_LO  = 0.35   # below this → auto-reject
REVIEW_HI  = 0.65   # above this → auto-accept
# images in [0.35, 0.65] → manual review zone

def metrics_at_threshold(probs, labels, t):
    preds = (np.array(probs) >= t).astype(int)
    y     = np.array(labels, dtype=int)
    acc   = accuracy_score(y, preds)
    mf1   = f1_score(y, preds, average="macro", zero_division=0)
    p, r, f, _ = precision_recall_fscore_support(
        y, preds, average=None, labels=[0,1], zero_division=0)
    cm    = confusion_matrix(y, preds, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": round(float(t), 2),
        "accuracy":  round(float(acc), 4),
        "macro_f1":  round(float(mf1), 4),
        "precision_invalid": round(float(p[0]),4),
        "recall_invalid":    round(float(r[0]),4),
        "f1_invalid":        round(float(f[0]),4),
        "precision_valid":   round(float(p[1]),4),
        "recall_valid":      round(float(r[1]),4),
        "f1_valid":          round(float(f[1]),4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }

def find_best_threshold(probs, labels, metric="macro_f1"):
    best_t, best_val = 0.5, -1.0
    for t in THRESHOLDS:
        m = metrics_at_threshold(probs, labels, t)
        if m[metric] > best_val:
            best_val = m[metric]
            best_t   = t
    return best_t, best_val

def review_zone(probs, lo=REVIEW_LO, hi=REVIEW_HI):
    probs = np.array(probs)
    return int(((probs >= lo) & (probs <= hi)).sum())

# ── ANALYSIS ──────────────────────────────────────────────────────────────────
SEP = "=" * 70
sep = "-" * 70

print(SEP)
print("EXPERIMENT 03 — ERROR ANALYSIS & THRESHOLD STUDY")
print(SEP)

analysis = {}

for model in MODELS:
    print(f"\n{SEP}")
    print(f"MODEL: {model.upper()}")
    print(SEP)

    analysis[model] = {}

    for split in SPLITS:
        res   = load_results(model, split)
        fns, labels = FN_DATA[split]
        probs  = np.array(res["probabilities"])
        preds  = np.array(res["predictions"])
        labels = np.array(labels, dtype=int)
        n      = len(labels)

        print(f"\n  [{split.upper()}]  n={n}  @threshold=0.5")

        # ── 1. Misclassified images ───────────────────────────────────────────
        # False Negatives: true=VALID(1), pred=INVALID(0)  → rejected valid images
        fn_mask = (labels == 1) & (preds == 0)
        fn_idx  = np.where(fn_mask)[0]
        fn_items = sorted(
            [{"filename": fns[i], "true": "valid",   "pred": "invalid",
              "confidence_of_pred": round(float(1 - probs[i]), 4),
              "prob_valid": round(float(probs[i]), 4)}
             for i in fn_idx],
            key=lambda x: x["prob_valid"]   # lowest confidence first
        )

        # False Positives: true=INVALID(0), pred=VALID(1) → accepted invalid images
        fp_mask = (labels == 0) & (preds == 1)
        fp_idx  = np.where(fp_mask)[0]
        fp_items = sorted(
            [{"filename": fns[i], "true": "invalid", "pred": "valid",
              "confidence_of_pred": round(float(probs[i]), 4),
              "prob_valid": round(float(probs[i]), 4)}
             for i in fp_idx],
            key=lambda x: -x["prob_valid"]  # highest (most confident wrong) first
        )

        print(f"\n  FALSE NEGATIVES (VALID rejected as INVALID): {len(fn_items)}")
        print(f"  {'Filename':<25} {'P(valid)':>10}  {'Conf of wrong pred':>18}")
        for item in fn_items:
            flag = " ← very low conf" if item["prob_valid"] < 0.35 else \
                   " ← borderline"    if item["prob_valid"] < 0.50 else ""
            print(f"    {item['filename']:<23} {item['prob_valid']:>10.4f}  {item['confidence_of_pred']:>18.4f}{flag}")

        print(f"\n  FALSE POSITIVES (INVALID accepted as VALID): {len(fp_items)}")
        print(f"  {'Filename':<25} {'P(valid)':>10}  {'Conf of wrong pred':>18}")
        for item in fp_items:
            flag = " ← very high conf" if item["prob_valid"] > 0.85 else \
                   " ← borderline"     if item["prob_valid"] > 0.50 else ""
            print(f"    {item['filename']:<23} {item['prob_valid']:>10.4f}  {item['confidence_of_pred']:>18.4f}{flag}")

        # ── 2. Confidence distribution of errors ─────────────────────────────
        all_errors = fn_items + fp_items
        if all_errors:
            err_probs = [abs(x["prob_valid"] - 0.5) for x in all_errors]  # dist from boundary
            high_conf_errors = sum(1 for d in err_probs if d > 0.25)   # prob outside [0.25,0.75]
            low_conf_errors  = sum(1 for d in err_probs if d <= 0.25)
            print(f"\n  Error confidence breakdown:")
            print(f"    High-confidence mistakes (prob outside 0.25–0.75): {high_conf_errors}")
            print(f"    Borderline mistakes      (prob inside  0.25–0.75): {low_conf_errors}")

        # ── 3. Manual review zone ─────────────────────────────────────────────
        n_review = review_zone(probs, REVIEW_LO, REVIEW_HI)
        auto_reject = int((probs < REVIEW_LO).sum())
        auto_accept = int((probs > REVIEW_HI).sum())
        print(f"\n  MANUAL REVIEW ZONE  [P(valid) in {REVIEW_LO}–{REVIEW_HI}]:")
        print(f"    Auto-accept (P > {REVIEW_HI})  : {auto_accept:>4}  ({auto_accept/n*100:.1f}%)")
        print(f"    Manual review zone       : {n_review:>4}  ({n_review/n*100:.1f}%)")
        print(f"    Auto-reject (P < {REVIEW_LO})  : {auto_reject:>4}  ({auto_reject/n*100:.1f}%)")
        # What's the composition of the review zone?
        rev_mask = (probs >= REVIEW_LO) & (probs <= REVIEW_HI)
        rev_labels = labels[rev_mask]
        if len(rev_labels) > 0:
            rev_tp = int((rev_labels == 1).sum())
            rev_tn = int((rev_labels == 0).sum())
            print(f"    Review zone composition  : {rev_tp} valid, {rev_tn} invalid")

        # ── 4. Threshold sweep ────────────────────────────────────────────────
        # Only do full sweep on val (no test peeking for selection)
        if split == "val":
            best_t_mf1,  best_mf1_val  = find_best_threshold(probs, labels, "macro_f1")
            best_t_rec,  best_rec_val   = find_best_threshold(probs, labels, "recall_invalid")

            m_default = metrics_at_threshold(probs, labels, 0.50)
            m_best_mf1 = metrics_at_threshold(probs, labels, best_t_mf1)
            m_best_rec = metrics_at_threshold(probs, labels, best_t_rec)

            print(f"\n  THRESHOLD SWEEP (val — for model selection only):")
            print(f"  {'Threshold':<12} {'mF1':>7} {'Acc':>7} {'Inv_Rec':>9} {'Val_Rec':>9}  {'FN':>4} {'FP':>4}")
            print(f"  {sep[:62]}")
            # Print key thresholds: default, best mF1, best invalid recall, plus surrounding range
            key_ts = sorted({0.50, best_t_mf1, best_t_rec,
                             round(best_t_mf1-0.05,2), round(best_t_mf1+0.05,2),
                             0.35, 0.40, 0.45, 0.55, 0.60})
            for t in key_ts:
                m = metrics_at_threshold(probs, labels, t)
                marker = ""
                if t == best_t_mf1: marker = " ← best mF1"
                elif t == best_t_rec: marker = " ← best inv recall"
                elif t == 0.50:     marker = " ← default"
                print(f"  {t:<12.2f} {m['macro_f1']:>7.4f} {m['accuracy']:>7.4f} "
                      f"{m['recall_invalid']:>9.4f} {m['recall_valid']:>9.4f}  "
                      f"{m['FN']:>4} {m['FP']:>4}{marker}")

            # Save threshold curve for val
            threshold_rows = [metrics_at_threshold(probs, labels, t) for t in THRESHOLDS]
            csv_path = ART_DIR / model / "val_threshold_sweep.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=threshold_rows[0].keys())
                writer.writeheader()
                writer.writerows(threshold_rows)

            # ROC AUC
            auc = roc_auc_score(labels, probs)
            print(f"\n  ROC-AUC (val): {auc:.4f}")

            # Evaluate at best_t_mf1 on test (to answer Q6/Q7 without selection bias)
            test_res   = load_results(model, "test")
            test_probs = np.array(test_res["probabilities"])
            test_labels_arr = np.array(FN_DATA["test"][1], dtype=int)
            m_test_tuned = metrics_at_threshold(test_probs, test_labels_arr, best_t_mf1)
            m_test_default = metrics_at_threshold(test_probs, test_labels_arr, 0.50)
            print(f"\n  THRESHOLD TUNING EFFECT ON TEST SET:")
            print(f"    @0.50 (default) : mF1={m_test_default['macro_f1']:.4f}  acc={m_test_default['accuracy']:.4f}  "
                  f"inv_rec={m_test_default['recall_invalid']:.4f}  val_rec={m_test_default['recall_valid']:.4f}")
            print(f"    @{best_t_mf1:.2f} (tuned)  : mF1={m_test_tuned['macro_f1']:.4f}  acc={m_test_tuned['accuracy']:.4f}  "
                  f"inv_rec={m_test_tuned['recall_invalid']:.4f}  val_rec={m_test_tuned['recall_valid']:.4f}")
            delta = m_test_tuned["macro_f1"] - m_test_default["macro_f1"]
            print(f"    Delta mF1       : {delta:+.4f}")

            analysis[model]["best_threshold_mf1"] = best_t_mf1
            analysis[model]["best_threshold_inv_recall"] = best_t_rec
            analysis[model]["val_roc_auc"] = round(float(auc), 4)
            analysis[model]["val_at_default"] = m_default
            analysis[model]["val_at_best_t"]  = m_best_mf1
            analysis[model]["test_at_default_t"] = m_test_default
            analysis[model]["test_at_best_t"]    = m_test_tuned

        analysis[model][split] = {
            "false_negatives": fn_items,
            "false_positives": fp_items,
            "n_fn": len(fn_items),
            "n_fp": len(fp_items),
            "review_zone_count": n_review,
            "auto_accept": auto_accept,
            "auto_reject": auto_reject,
        }

# ── CROSS-MODEL SUMMARY ───────────────────────────────────────────────────────
print(f"\n{SEP}")
print("CROSS-MODEL SUMMARY — DEFAULT THRESHOLD (0.50)")
print(SEP)
print(f"\n  TEST SET (final evaluation)")
print(f"  {'Model':<14} {'mF1':>7} {'Acc':>7} {'Inv P/R/F1':>22} {'Val P/R/F1':>22}  {'FN':>4} {'FP':>4}")
print(f"  {sep[:90]}")
for model in MODELS:
    m = analysis[model]["test_at_default_t"]
    print(f"  {model:<14} {m['macro_f1']:>7.4f} {m['accuracy']:>7.4f} "
          f"  {m['precision_invalid']:.3f}/{m['recall_invalid']:.3f}/{m['f1_invalid']:.3f}"
          f"   {m['precision_valid']:.3f}/{m['recall_valid']:.3f}/{m['f1_valid']:.3f}"
          f"  {m['FN']:>4} {m['FP']:>4}")

print(f"\n  TEST SET — TUNED THRESHOLD (best val macro-F1 threshold per model)")
print(f"  {'Model':<14} {'Thresh':>7} {'mF1':>7} {'Acc':>7} {'Inv P/R/F1':>22} {'Val P/R/F1':>22}  {'FN':>4} {'FP':>4}  {'Delta mF1':>10}")
print(f"  {sep[:110]}")
for model in MODELS:
    t  = analysis[model]["best_threshold_mf1"]
    m  = analysis[model]["test_at_best_t"]
    m0 = analysis[model]["test_at_default_t"]
    d  = m["macro_f1"] - m0["macro_f1"]
    print(f"  {model:<14} {t:>7.2f} {m['macro_f1']:>7.4f} {m['accuracy']:>7.4f} "
          f"  {m['precision_invalid']:.3f}/{m['recall_invalid']:.3f}/{m['f1_invalid']:.3f}"
          f"   {m['precision_valid']:.3f}/{m['recall_valid']:.3f}/{m['f1_valid']:.3f}"
          f"  {m['FN']:>4} {m['FP']:>4}  {d:>+10.4f}")

print(f"\n  MANUAL REVIEW ZONE [{REVIEW_LO}–{REVIEW_HI}] on TEST SET")
print(f"  {'Model':<14} {'Auto-accept':>12} {'Review zone':>13} {'Auto-reject':>12}  AUC (val)")
print(f"  {sep[:75]}")
for model in MODELS:
    aa  = analysis[model]["test"]["auto_accept"]
    rz  = analysis[model]["test"]["review_zone_count"]
    ar  = analysis[model]["test"]["auto_reject"]
    auc = analysis[model]["val_roc_auc"]
    print(f"  {model:<14} {aa:>12} {rz:>13} {ar:>12}  {auc:.4f}")

# Q6: Does Linear remain competitive after tuning?
print(f"\n  Q6 — Linear vs MLP after tuning (test macro-F1):")
lin_def  = analysis["linear"]["test_at_default_t"]["macro_f1"]
lin_tun  = analysis["linear"]["test_at_best_t"]["macro_f1"]
sml_def  = analysis["small_mlp"]["test_at_default_t"]["macro_f1"]
sml_tun  = analysis["small_mlp"]["test_at_best_t"]["macro_f1"]
reg_def  = analysis["reg_mlp"]["test_at_default_t"]["macro_f1"]
reg_tun  = analysis["reg_mlp"]["test_at_best_t"]["macro_f1"]
print(f"    Linear     default={lin_def:.4f}  tuned={lin_tun:.4f}")
print(f"    Small MLP  default={sml_def:.4f}  tuned={sml_tun:.4f}")
print(f"    Reg MLP    default={reg_def:.4f}  tuned={reg_tun:.4f}")
gap_lin_reg_def = reg_def - lin_def
gap_lin_reg_tun = reg_tun - lin_tun
print(f"    Reg MLP - Linear  @ default : {gap_lin_reg_def:+.4f}")
print(f"    Reg MLP - Linear  @ tuned   : {gap_lin_reg_tun:+.4f}")

# Q7: Does Reg MLP justify complexity?
print(f"\n  Q7 — Regularized MLP vs Small MLP complexity trade-off:")
gap_def = reg_def - sml_def
gap_tun = reg_tun - sml_tun
params_mlp = 69825
params_lin = 513
print(f"    Small MLP params  : {params_mlp:,}")
print(f"    Reg MLP   params  : {params_mlp:,}  (same count, +dropout, +weight_decay)")
print(f"    Linear    params  : {params_lin:,}")
print(f"    Reg vs Small mF1  @ default  : {gap_def:+.4f}")
print(f"    Reg vs Small mF1  @ tuned    : {gap_tun:+.4f}")

# ── SAVE ─────────────────────────────────────────────────────────────────────
out_path = ART_DIR / "error_analysis.json"
# Convert numpy ints for JSON
def to_serializable(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_serializable(i) for i in obj]
    return obj

with open(out_path, "w") as f:
    json.dump(to_serializable(analysis), f, indent=2)
print(f"\nFull analysis saved → {out_path}")
