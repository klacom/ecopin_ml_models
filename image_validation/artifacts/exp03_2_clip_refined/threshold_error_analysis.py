"""
Experiment 3.2 — Phase 5: Threshold & Error Analysis
=====================================================
Selected model : Regularized MLP
Source data    : saved val_results.json / test_results.json  (no retraining)
Filenames      : from exp03_2 split CSVs and embedding .npz files
Output         : artifacts/exp03_2_clip_refined/analysis/
"""

import csv, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    f1_score, confusion_matrix,
)

# ── PATHS ─────────────────────────────────────────────────────────────────────
ART_DIR  = Path(r"C:\dev\ecopin_ml_models\image_validation\artifacts\exp03_2_clip_refined")
EMB_DIR  = Path(r"C:\dev\datasets\ecopin_dataset\embeddings\exp03_2")
ART_31   = Path(r"C:\dev\ecopin_ml_models\image_validation\artifacts\exp03_clip_classifiers")
OUT_DIR  = ART_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "reg_mlp"   # selected model

# ── LOAD SAVED PREDICTIONS ────────────────────────────────────────────────────
def load_results(split):
    with open(ART_DIR / MODEL / f"{split}_results.json") as f:
        return json.load(f)

val_res  = load_results("val")
test_res = load_results("test")

val_probs  = np.array(val_res["probabilities"])   # P(valid)
test_probs = np.array(test_res["probabilities"])

val_labels  = np.array([1 if p >= 0.5 else 0 for p in val_probs], dtype=int)  # placeholder
# Load ground-truth labels from embeddings
def load_labels(split):
    d = np.load(str(EMB_DIR / f"clip_vitb32_{split}.npz"), allow_pickle=True)
    return d["label_ints"].astype(int), d["filenames"].tolist()

val_labels,  val_fns  = load_labels("val")
test_labels, test_fns = load_labels("test")

# Sanity: confirm probs align with saved accuracy at 0.5
def check_alignment(probs, labels, saved_acc):
    preds = (probs >= 0.5).astype(int)
    acc   = accuracy_score(labels, preds)
    assert abs(acc - saved_acc) < 1e-4, f"Prob/label mismatch: {acc:.4f} vs {saved_acc:.4f}"

check_alignment(val_probs,  val_labels,  val_res["accuracy"])
check_alignment(test_probs, test_labels, test_res["accuracy"])
print("Sanity check passed: saved probabilities align with reported accuracy.")

# ── THRESHOLD SWEEP ───────────────────────────────────────────────────────────
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

def metrics_at(probs, labels, t):
    preds = (probs >= t).astype(int)
    y     = np.array(labels, dtype=int)
    acc   = accuracy_score(y, preds)
    mf1   = f1_score(y, preds, average="macro", zero_division=0)
    p, r, f, _ = precision_recall_fscore_support(y, preds, average=None,
                                                  labels=[0, 1], zero_division=0)
    cm    = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "t":       round(float(t), 2),
        "acc":     round(float(acc), 4),
        "mf1":     round(float(mf1), 4),
        "p_inv":   round(float(p[0]), 4),
        "r_inv":   round(float(r[0]), 4),
        "f1_inv":  round(float(f[0]), 4),
        "p_val":   round(float(p[1]), 4),
        "r_val":   round(float(r[1]), 4),
        "f1_val":  round(float(f[1]), 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "false_valid":   int(fp),   # INVALID predicted as VALID
        "false_invalid": int(fn),   # VALID predicted as INVALID
    }

val_sweep  = [metrics_at(val_probs,  val_labels,  t) for t in THRESHOLDS]
test_sweep = [metrics_at(test_probs, test_labels, t) for t in THRESHOLDS]

# Best val threshold by macro-F1
best_t_mf1   = max(val_sweep, key=lambda m: m["mf1"])["t"]
# Best val threshold by balanced recall (minimise |inv_recall - val_recall|)
best_t_bal   = min(val_sweep, key=lambda m: abs(m["r_inv"] - m["r_val"]))["t"]
# Best val threshold by invalid recall >= 0.90 while keeping mF1 as high as possible
inv_rec_90   = [m for m in val_sweep if m["r_inv"] >= 0.90]
best_t_inv90 = (max(inv_rec_90, key=lambda m: m["mf1"])["t"]) if inv_rec_90 else None

SEP  = "=" * 72
sep  = "-" * 72

print(f"\n{SEP}")
print("PHASE 2 — THRESHOLD SWEEP (VALIDATION SET)")
print(SEP)
print(f"{'Thresh':>7} {'Acc':>7} {'mF1':>7} {'InvP':>7} {'InvR':>7} {'InvF1':>7} "
      f"{'ValP':>7} {'ValR':>7} {'ValF1':>7} {'FV':>4} {'FI':>4}")
print(sep)
for m in val_sweep:
    markers = []
    if m["t"] == best_t_mf1:  markers.append("* best mF1")
    if m["t"] == best_t_bal:  markers.append("* balanced")
    if best_t_inv90 and m["t"] == best_t_inv90: markers.append("* inv_rec>=0.90")
    tag = "  " + " | ".join(markers) if markers else ""
    print(f"{m['t']:>7.2f} {m['acc']:>7.4f} {m['mf1']:>7.4f} "
          f"{m['p_inv']:>7.4f} {m['r_inv']:>7.4f} {m['f1_inv']:>7.4f} "
          f"{m['p_val']:>7.4f} {m['r_val']:>7.4f} {m['f1_val']:>7.4f} "
          f"{m['false_valid']:>4} {m['false_invalid']:>4}{tag}")

# Save sweep CSV
with open(OUT_DIR / "val_threshold_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=val_sweep[0].keys()); w.writeheader(); w.writerows(val_sweep)

print(f"\nBest threshold (val mF1)         : {best_t_mf1}")
print(f"Best threshold (balanced recall) : {best_t_bal}")
print(f"Best threshold (inv recall>=0.90): {best_t_inv90}")

# ── PHASE 3 — ECOPIN OPERATING POINT ─────────────────────────────────────────
print(f"\n{SEP}")
print("PHASE 3 — ECOPIN OPERATING POINT ANALYSIS (VALIDATION)")
print(SEP)
print("EcoPin context:")
print("  False VALID  (FP): INVALID accepted as VALID  -> poor evidence enters system")
print("  False INVALID(FN): VALID rejected as INVALID  -> useful evidence lost")
print()
for m in val_sweep:
    if m["t"] in (0.45, 0.50, 0.55, 0.60, best_t_mf1, best_t_bal):
        tag = ""
        if m["t"] == best_t_mf1: tag = " <- best mF1"
        if m["t"] == best_t_bal: tag = " <- most balanced recall"
        print(f"  t={m['t']:.2f}  mF1={m['mf1']:.4f}  "
              f"inv_rec={m['r_inv']:.3f}  val_rec={m['r_val']:.3f}  "
              f"FV={m['false_valid']:3d}  FI={m['false_invalid']:3d}{tag}")

# Recommended threshold: best val mF1
REC_T = best_t_mf1
rec_m_val = metrics_at(val_probs, val_labels, REC_T)
print(f"\nRecommended threshold: {REC_T}")
print(f"  Val  mF1={rec_m_val['mf1']:.4f}  acc={rec_m_val['acc']:.4f}  "
      f"FV={rec_m_val['false_valid']}  FI={rec_m_val['false_invalid']}")

# ── PHASE 4 — MANUAL REVIEW ZONE ─────────────────────────────────────────────
print(f"\n{SEP}")
print("PHASE 4 — CONFIDENCE DISTRIBUTION & MANUAL REVIEW ZONE")
print(SEP)

# Analyse where the probability mass sits
buckets = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
           (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
           (0.8, 0.9), (0.9, 1.001)]
print("\n  Validation probability distribution (all 180 samples):")
print(f"  {'P(valid) range':>16}  {'Count':>6}  {'% of 180':>10}  {'valid_in_bucket':>16}  {'invalid_in_bucket':>18}")
for lo, hi in buckets:
    mask   = (val_probs >= lo) & (val_probs < hi)
    n_tot  = mask.sum()
    n_v    = ((val_labels == 1) & mask).sum()
    n_i    = ((val_labels == 0) & mask).sum()
    print(f"  [{lo:.1f}, {hi:.3f})  {n_tot:>6}  {n_tot/180*100:>9.1f}%  "
          f"{n_v:>16}  {n_i:>18}")

# Find a sensible review zone: where both classes co-exist
# Look at narrow band around each threshold
print("\n  Co-existence zones (both valid & invalid samples present):")
for lo in np.arange(0.30, 0.75, 0.05):
    hi = lo + 0.15
    mask = (val_probs >= lo) & (val_probs < hi)
    n_v  = ((val_labels == 1) & mask).sum()
    n_i  = ((val_labels == 0) & mask).sum()
    if n_v > 0 and n_i > 0:
        print(f"  [{lo:.2f}, {hi:.2f}): {mask.sum():3d} images  valid={n_v}  invalid={n_i}")

# Specific recommended review zone
REVIEW_LO, REVIEW_HI = 0.35, 0.65
rev_mask  = (val_probs >= REVIEW_LO) & (val_probs <= REVIEW_HI)
auto_acc  = (val_probs >  REVIEW_HI).sum()
auto_rej  = (val_probs <  REVIEW_LO).sum()
n_review  = rev_mask.sum()
rev_valid = ((val_labels == 1) & rev_mask).sum()
rev_inv   = ((val_labels == 0) & rev_mask).sum()

print(f"\n  Proposed review zone: P(valid) in [{REVIEW_LO}, {REVIEW_HI}]  (validation set)")
print(f"    Auto-VALID  (P > {REVIEW_HI}): {auto_acc:4d}  ({auto_acc/180*100:.1f}%)")
print(f"    Manual zone            : {n_review:4d}  ({n_review/180*100:.1f}%)  "
      f"[{rev_valid} valid, {rev_inv} invalid inside zone]")
print(f"    Auto-INVALID (P < {REVIEW_LO}): {auto_rej:4d}  ({auto_rej/180*100:.1f}%)")

# Precision of auto zones
auto_acc_mask  = val_probs > REVIEW_HI
auto_rej_mask  = val_probs < REVIEW_LO
prec_auto_acc  = ((val_labels == 1) & auto_acc_mask).sum() / auto_acc_mask.sum() if auto_acc_mask.sum() else 0
prec_auto_rej  = ((val_labels == 0) & auto_rej_mask).sum() / auto_rej_mask.sum() if auto_rej_mask.sum() else 0
print(f"\n    Auto-VALID precision (correct valids above {REVIEW_HI}) : {prec_auto_acc:.3f}")
print(f"    Auto-REJECT precision (correct inv. below {REVIEW_LO}) : {prec_auto_rej:.3f}")

# ── PHASE 5 — TEST SET CONFIRMATION ──────────────────────────────────────────
print(f"\n{SEP}")
print(f"PHASE 5 — TEST SET CONFIRMATION AT SELECTED THRESHOLD t={REC_T}")
print(SEP)
rec_m_test = metrics_at(test_probs, test_labels, REC_T)
print(f"  Accuracy      : {rec_m_test['acc']:.4f}")
print(f"  Macro-F1      : {rec_m_test['mf1']:.4f}")
print(f"  INVALID  P/R/F1 : {rec_m_test['p_inv']:.4f} / {rec_m_test['r_inv']:.4f} / {rec_m_test['f1_inv']:.4f}")
print(f"  VALID    P/R/F1 : {rec_m_test['p_val']:.4f} / {rec_m_test['r_val']:.4f} / {rec_m_test['f1_val']:.4f}")
print(f"  False VALID    (FP): {rec_m_test['false_valid']}")
print(f"  False INVALID  (FN): {rec_m_test['false_invalid']}")
cm = confusion_matrix(test_labels, (test_probs >= REC_T).astype(int), labels=[0, 1])
print(f"  Confusion matrix:")
print(f"                pred_invalid  pred_valid")
print(f"    true_invalid    {cm[0][0]:5d}        {cm[0][1]:5d}")
print(f"    true_valid      {cm[1][0]:5d}        {cm[1][1]:5d}")

# Comparison: default 0.50 vs recommended
m_default = metrics_at(test_probs, test_labels, 0.50)
print(f"\n  Comparison: default 0.50 vs recommended {REC_T}")
print(f"  {'Metric':<20} {'@0.50':>8} {'@{:.2f}'.format(REC_T):>8} {'Delta':>8}")
print(f"  {'-'*46}")
for label, key in [("Accuracy","acc"),("Macro-F1","mf1"),
                   ("INVALID recall","r_inv"),("VALID recall","r_val"),
                   ("False VALID (FP)","false_valid"),("False INVALID (FN)","false_invalid")]:
    v0 = m_default[key]; vt = rec_m_test[key]
    d  = vt - v0
    print(f"  {label:<20} {v0:>8.4f} {vt:>8.4f} {d:>+8.4f}")

# ── PHASE 6 — ERROR ANALYSIS ─────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"PHASE 6 — ERROR ANALYSIS (TEST SET AT t={REC_T})")
print(SEP)

test_preds_rec = (test_probs >= REC_T).astype(int)

# False INVALID (FN): true VALID, predicted INVALID
fn_items = [(test_fns[i], test_probs[i]) for i in range(len(test_labels))
            if test_labels[i] == 1 and test_preds_rec[i] == 0]
fn_items.sort(key=lambda x: x[1])   # lowest P(valid) first

# False VALID (FP): true INVALID, predicted VALID
fp_items = [(test_fns[i], test_probs[i]) for i in range(len(test_labels))
            if test_labels[i] == 0 and test_preds_rec[i] == 1]
fp_items.sort(key=lambda x: -x[1])  # highest P(valid) first

print(f"\n  FALSE INVALID — VALID images rejected  ({len(fn_items)} images):")
print(f"  {'Filename':<25} {'P(valid)':>10}  Confidence profile")
for fn, prob in fn_items:
    if   prob < 0.20: note = "very low conf — strong misclassification"
    elif prob < 0.35: note = "low conf — model uncertain, leaning invalid"
    elif prob < REC_T: note = "borderline — near threshold"
    else:              note = ""
    print(f"    {fn:<23} {prob:>10.4f}  {note}")

print(f"\n  FALSE VALID — INVALID images accepted  ({len(fp_items)} images):")
print(f"  {'Filename':<25} {'P(valid)':>10}  Confidence profile")
for fn, prob in fp_items:
    if   prob > 0.90: note = "very high conf — strong misclassification"
    elif prob > 0.75: note = "high conf — model committed to wrong prediction"
    elif prob > REC_T: note = "borderline — near threshold"
    else:              note = ""
    print(f"    {fn:<23} {prob:>10.4f}  {note}")

# Confidence breakdown
if fn_items:
    fn_probs = [p for _, p in fn_items]
    print(f"\n  FN confidence summary:")
    print(f"    Min P(valid): {min(fn_probs):.4f}  Max: {max(fn_probs):.4f}  Mean: {np.mean(fn_probs):.4f}")
    very_low = sum(1 for p in fn_probs if p < 0.20)
    borderline = sum(1 for p in fn_probs if p >= 0.35)
    print(f"    Very low conf (P<0.20)    : {very_low}")
    print(f"    Borderline (P>=0.35)      : {borderline}")
    print(f"    [Note: visual inspection not available — confidence-based profiling only]")

if fp_items:
    fp_probs = [p for _, p in fp_items]
    print(f"\n  FP confidence summary:")
    print(f"    Min P(valid): {min(fp_probs):.4f}  Max: {max(fp_probs):.4f}  Mean: {np.mean(fp_probs):.4f}")
    high_conf = sum(1 for p in fp_probs if p > 0.75)
    borderline = sum(1 for p in fp_probs if p < 0.65)
    print(f"    High conf (P>0.75)       : {high_conf}")
    print(f"    Near-threshold (<0.65)   : {borderline}")
    print(f"    [Note: visual inspection not available — confidence-based profiling only]")

# ── PHASE 7 — COMPARE WITH EXP 3.1 ──────────────────────────────────────────
print(f"\n{SEP}")
print("PHASE 7 — COMPARISON WITH EXPERIMENT 3.1")
print(SEP)
EXP31_SWEEP = ART_31 / "reg_mlp" / "val_threshold_sweep.csv"
if EXP31_SWEEP.exists():
    import csv as _csv
    with open(EXP31_SWEEP) as f:
        rows_31 = list(_csv.DictReader(f))
    # Build comparable sweep for 3.2 val at same thresholds
    compare_ts = [0.45, 0.50, 0.55, 0.60]
    print(f"\n  RegMLP val metrics — Exp 3.1 vs 3.2  (validation set):")
    print(f"  {'Thresh':>7} {'3.1 mF1':>9} {'3.2 mF1':>9} {'Delta':>8}  "
          f"{'3.1 InvR':>10} {'3.2 InvR':>10} {'Delta':>8}")
    print(f"  {'-'*72}")
    for t in compare_ts:
        r31 = next((r for r in rows_31 if float(r["threshold"]) == t), None)
        r32 = metrics_at(val_probs, val_labels, t)
        if r31:
            mf1_31  = float(r31["macro_f1"])
            invr_31 = float(r31["recall_invalid"])
            d_mf1   = r32["mf1"] - mf1_31
            d_invr  = r32["r_inv"] - invr_31
            print(f"  {t:>7.2f} {mf1_31:>9.4f} {r32['mf1']:>9.4f} {d_mf1:>+8.4f}  "
                  f"{invr_31:>10.4f} {r32['r_inv']:>10.4f} {d_invr:>+8.4f}")
    # Test-set comparison at recommended threshold
    EXP31_TEST = ART_31 / "reg_mlp" / "test_results.json"
    if EXP31_TEST.exists():
        with open(EXP31_TEST) as f:
            t31_test = json.load(f)
        t31_probs  = np.array(t31_test["probabilities"])
        t31_labels_arr = np.array(t31_test["predictions"])
        # Load ground-truth from Exp 3.1 embeddings for fair comparison
        emb31 = Path(r"C:\dev\datasets\ecopin_dataset\embeddings\exp03")
        d31   = np.load(str(emb31 / "clip_vitb32_test.npz"), allow_pickle=True)
        t31_gt = d31["label_ints"].astype(int)
        m31_def = metrics_at(t31_probs, t31_gt, 0.50)
        m32_def = metrics_at(test_probs, test_labels, 0.50)
        m31_rec = metrics_at(t31_probs, t31_gt, 0.49)   # 3.1 best t was 0.49
        m32_rec = rec_m_test
        print(f"\n  RegMLP test metrics — Exp 3.1 vs 3.2:")
        print(f"  {'Metric':<22} {'3.1@0.50':>10} {'3.2@{:.2f}'.format(REC_T):>12} {'Delta':>8}")
        print(f"  {'-'*56}")
        for label, key in [("Macro-F1","mf1"),("INVALID recall","r_inv"),
                            ("VALID recall","r_val"),("False VALID","false_valid"),
                            ("False INVALID","false_invalid")]:
            v31 = m31_def[key]; v32 = m32_rec[key]
            d   = v32 - v31
            print(f"  {label:<22} {v31:>10.4f} {v32:>12.4f} {d:>+8.4f}")
else:
    print("  Exp 3.1 threshold sweep not found — skipping direct comparison.")

# ── SAVE SUMMARY ─────────────────────────────────────────────────────────────
summary = {
    "model": MODEL,
    "recommended_threshold": REC_T,
    "review_zone": {"lo": REVIEW_LO, "hi": REVIEW_HI},
    "val_at_recommended": rec_m_val,
    "test_at_recommended": rec_m_test,
    "test_at_0.50": m_default,
    "false_valid_test":   rec_m_test["false_valid"],
    "false_invalid_test": rec_m_test["false_invalid"],
    "val_threshold_sweep": val_sweep,
    "fn_images": [{"filename": fn, "prob_valid": round(float(p), 4)}
                  for fn, p in fn_items],
    "fp_images": [{"filename": fn, "prob_valid": round(float(p), 4)}
                  for fn, p in fp_items],
}
with open(OUT_DIR / "threshold_analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
with open(OUT_DIR / "test_threshold_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=test_sweep[0].keys()); w.writeheader(); w.writerows(test_sweep)

print(f"\nOutputs saved to {OUT_DIR}")
print("Done.")
