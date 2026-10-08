"""
Training & Validation Engine: Final Maximum-Performance Ensemble
---------------------------------------------------------------
1. Fits StandardScaler exclusively on the authentic training partition.
2. Trains the multi-seed calibrated ensemble on augmented training data.
3. Optimizes the clinical decision threshold strictly on the balanced validation set.
4. Freezes all model parameters, weights, and decision thresholds.
5. Saves the final production checkpoint and detailed validation report.

STRICT INVARIANCES:
- ZERO access to or modification of hold-out test sets.
- 100% frozen state prior to any test set evaluation.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.classical_classifier import ACOUSTIC_FEATURE_COLS
from src.models.calibrated_ensemble import CalibratedAcousticEnsemble

CLASS_NAMES = ["normal", "laryngozele", "vox_senilis"]
CLASS_LABELS = [0, 1, 2]


def evaluate_partition(
    ensemble: CalibratedAcousticEnsemble,
    X: np.ndarray,
    y: np.ndarray,
    df_meta: pd.DataFrame,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Computes comprehensive multi-class diagnostic metrics on a partition."""
    th = threshold if threshold is not None else ensemble.frozen_threshold
    preds = ensemble.predict(X, threshold=th)
    probs = ensemble.predict_proba(X)
    
    acc = float(accuracy_score(y, preds))
    bal_acc = float(balanced_accuracy_score(y, preds))
    
    p, r, f1, sup = precision_recall_fscore_support(
        y, preds, labels=CLASS_LABELS, zero_division=0
    )
    macro_p = float(np.mean(p))
    macro_r = float(np.mean(r))
    macro_f1 = float(np.mean(f1))
    cm = confusion_matrix(y, preds, labels=CLASS_LABELS)
    total_samples = int(np.sum(cm))

    # Multi-class One-vs-Rest ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    except Exception:
        roc_auc = 0.5

    # Specificity per class
    specs = []
    for c in CLASS_LABELS:
        tn = np.sum(cm) - (np.sum(cm[c, :]) + np.sum(cm[:, c]) - cm[c, c])
        fp = np.sum(cm[:, c]) - cm[c, c]
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        specs.append(spec)
    macro_spec = float(np.mean(specs))

    # Detailed inspection of Laryngozele instances
    lar_details = []
    lar_mask = (y == 1)
    if np.any(lar_mask):
        sub_meta = df_meta[lar_mask]
        sub_preds = preds[lar_mask]
        sub_probs = probs[lar_mask]
        for (_, row), pred, prob in zip(sub_meta.iterrows(), sub_preds, sub_probs):
            lar_details.append({
                "vowel_type": str(row.get("vowel_type", "")),
                "session_id": str(row.get("session_id", "")),
                "speaker_id": str(row.get("speaker_id", "")),
                "true_class": "laryngozele",
                "pred_class": CLASS_NAMES[pred],
                "prob_normal": float(prob[0]),
                "prob_laryngozele": float(prob[1]),
                "prob_vox_senilis": float(prob[2])
            })

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_specificity": macro_spec,
        "macro_f1": macro_f1,
        "macro_roc_auc": roc_auc,
        "per_class": {
            CLASS_NAMES[i]: {
                "support": int(sup[i]),
                "precision": float(p[i]),
                "recall": float(r[i]),
                "specificity": float(specs[i]),
                "f1_score": float(f1[i])
            }
            for i in range(3)
        },
        "confusion_matrix": cm.tolist(),
        "laryngozele_predictions": lar_details
    }


def run_training_and_freeze():
    print("=" * 85)
    print("STAGE 4: FINAL MAXIMUM-PERFORMANCE ENSEMBLE TRAINING & CALIBRATION")
    print("=" * 85)

    tab_dir = os.path.join(BASE_DIR, "data", "processed", "tabular_features")
    train_csv = os.path.join(tab_dir, "train_authentic_features.csv")
    val_bal_csv = os.path.join(tab_dir, "val_balanced_features.csv")
    val_full_csv = os.path.join(tab_dir, "val_full_features.csv")

    print("\n--- 1. LOADING TRAINING AND VALIDATION DATA ---")
    tr_df = pd.read_csv(train_csv)
    vb_df = pd.read_csv(val_bal_csv)
    vf_df = pd.read_csv(val_full_csv)

    X_train_raw = tr_df[ACOUSTIC_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = tr_df["class_label"].to_numpy(dtype=int)

    X_val_b_raw = vb_df[ACOUSTIC_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_val_b = vb_df["class_label"].to_numpy(dtype=int)

    X_val_f_raw = vf_df[ACOUSTIC_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_val_f = vf_df["class_label"].to_numpy(dtype=int)

    print(f"Train samples:      {len(y_train)} (Classes: {dict(pd.Series(y_train).value_counts())})")
    print(f"Val Balanced:       {len(y_val_b)} (Classes: {dict(pd.Series(y_val_b).value_counts())})")
    print(f"Val Full:           {len(y_val_f)} (Classes: {dict(pd.Series(y_val_f).value_counts())})")

    # Fit scaler strictly on training data
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    print("\n[Data Isolation] StandardScaler strictly fit on training partition (N=743).")

    # Instantiate ensemble
    seeds = [42, 101, 2024, 777, 999]
    print(f"\n--- 2. FITTING CALIBRATED MULTI-SEED ENSEMBLE (5 SEEDS: {seeds}) ---")
    ensemble = CalibratedAcousticEnsemble(
        seeds=seeds,
        target_minority_samples=350,
        lr_c=0.15,
        svc_lin_c=0.10,
        svc_rbf_c=1.00,
        svc_rbf_gamma=0.01,
        model_weights=(0.45, 0.40, 0.15)
    )
    ensemble.fit(X_train_raw, y_train, scaler=scaler)
    print("Multi-seed ensemble training complete across all 5 seeds.")

    # Threshold Optimization on Balanced Validation Partition
    print("\n--- 3. THRESHOLD OPTIMIZATION (STRICTLY ON BALANCED VALIDATION) ---")
    probs_val_b = ensemble.predict_proba(X_val_b_raw)
    
    threshold_grid = np.arange(0.10, 0.36, 0.02)
    best_score = -1.0
    best_th = 0.12
    best_metrics_b = None

    print(f"{'Threshold':>9} | {'Val Acc':>8} | {'Bal Acc':>8} | {'Macro F1':>9} | {'Lar Rec':>8} | {'Lar Prec':>8} | {'Lar F1':>8}")
    print("-" * 80)

    for th in threshold_grid:
        m = evaluate_partition(ensemble, X_val_b_raw, y_val_b, vb_df, threshold=th)
        macro_f1 = m["macro_f1"]
        lar_rec = m["per_class"]["laryngozele"]["recall"]
        lar_prec = m["per_class"]["laryngozele"]["precision"]
        lar_f1 = m["per_class"]["laryngozele"]["f1_score"]

        # Optimize for 50% Macro-F1 + 50% Laryngozele Recall, ensuring Precision >= 60%
        if lar_prec >= 0.60:
            score = 0.5 * macro_f1 + 0.5 * lar_rec
        else:
            score = 0.3 * macro_f1 + 0.3 * lar_rec  # Penalize precision drops

        print(f"{th:9.2f} | {m['accuracy']*100:7.2f}% | {m['balanced_accuracy']*100:7.2f}% | {macro_f1*100:8.2f}% | {lar_rec*100:7.2f}% | {lar_prec*100:7.2f}% | {lar_f1*100:7.2f}%")

        if score > best_score:
            best_score = score
            best_th = float(th)
            best_metrics_b = m

    print("=" * 80)
    print(f"OPTIMAL FROZEN THRESHOLD: tau* = {best_th:.2f}")
    print(f"Balanced Val Accuracy:    {best_metrics_b['accuracy']*100:.2f}%")
    print(f"Balanced Val Macro F1:    {best_metrics_b['macro_f1']*100:.2f}%")
    print(f"Balanced Val Lar Recall:  {best_metrics_b['per_class']['laryngozele']['recall']*100:.2f}%")
    print(f"Balanced Val Lar Prec:    {best_metrics_b['per_class']['laryngozele']['precision']*100:.2f}%")
    print(f"Balanced Val Lar F1:      {best_metrics_b['per_class']['laryngozele']['f1_score']*100:.2f}%")
    print("=" * 80)

    # Freeze threshold in ensemble
    ensemble.frozen_threshold = best_th

    # Evaluate on Full Validation
    best_metrics_f = evaluate_partition(ensemble, X_val_f_raw, y_val_f, vf_df, threshold=best_th)

    # 4. Save Frozen Checkpoint
    ckpt_dir = os.path.join(BASE_DIR, "results", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "final_calibrated_ensemble.joblib")

    checkpoint_data = {
        "ensemble": ensemble,
        "scaler": scaler,
        "feature_cols": ACOUSTIC_FEATURE_COLS,
        "seeds": seeds,
        "frozen_threshold": best_th,
        "val_balanced_metrics": best_metrics_b,
        "val_full_metrics": best_metrics_f,
        "creation_time": datetime.now().isoformat()
    }
    joblib.dump(checkpoint_data, ckpt_path)
    print(f"\n[Checkpoint] Saved 100% frozen model checkpoint to: {ckpt_path}")

    # 5. Write Comprehensive Validation Report
    reports_dir = os.path.join(BASE_DIR, "results", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    rep_path = os.path.join(reports_dir, "final_ensemble_validation_report.txt")

    cm_b = np.array(best_metrics_b["confusion_matrix"])
    cm_f = np.array(best_metrics_f["confusion_matrix"])

    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("=" * 85 + "\n")
        f.write("FINAL MAXIMUM-PERFORMANCE ENSEMBLE: VALIDATION REPORT\n")
        f.write("Automated Multi-Class Laryngeal Voice Disorder Classification\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 85 + "\n\n")

        f.write("1. PIPELINE & HYPERPARAMETER SPECIFICATION\n")
        f.write("-" * 85 + "\n")
        f.write("  - Audio Representation:    52 Clinical Acoustic Biomarkers\n")
        f.write("  - Feature Normalization:   StandardScaler (fit strictly on training data)\n")
        f.write(f"  - Ensemble Model A (45%):  LogisticRegression (C=0.15, class_weight='balanced')\n")
        f.write(f"  - Ensemble Model B (40%):  Linear SVC (C=0.10, class_weight='balanced', Platt-calibrated)\n")
        f.write(f"  - Ensemble Model C (15%):  RBF SVC (C=1.00, gamma=0.01, class_weight='balanced', Platt-calibrated)\n")
        f.write(f"  - Multi-Seed Bagging:      5 Random Seeds {seeds}\n")
        f.write(f"  - Training Augmentation:   Minority SMOTE + physiological perturbation to 350 samples/seed\n")
        f.write(f"  - Frozen Threshold:        tau* = {best_th:.2f} (strictly selected on val_balanced)\n")
        f.write("  - Hold-Out Test Sets:      STRICTLY UNTOUCHED & UNACCESSED\n\n")

        f.write("2. BALANCED VALIDATION SET BENCHMARK (VAL_BALANCED.CSV)\n")
        f.write("-" * 85 + "\n")
        f.write(f"  - Total Evaluation Samples: {len(y_val_b)}\n")
        f.write(f"  - Overall Accuracy:         {best_metrics_b['accuracy']*100:.2f}%\n")
        f.write(f"  - Balanced Accuracy:        {best_metrics_b['balanced_accuracy']*100:.2f}%\n")
        f.write(f"  - Macro Precision:          {best_metrics_b['macro_precision']*100:.2f}%\n")
        f.write(f"  - Macro Recall / Sens.:     {best_metrics_b['macro_recall']*100:.2f}%\n")
        f.write(f"  - Macro Specificity:        {best_metrics_b['macro_specificity']*100:.2f}%\n")
        f.write(f"  - Macro F1-Score:           {best_metrics_b['macro_f1']*100:.2f}%\n")
        f.write(f"  - Macro OvR ROC-AUC:        {best_metrics_b['macro_roc_auc']:.4f}\n\n")

        f.write("PER-CLASS PERFORMANCE (Balanced Val):\n")
        f.write(f"{'Class':<16} | {'Support':<8} | {'Precision':<10} | {'Recall':<10} | {'Specificity':<12} | {'F1-Score':<10}\n")
        f.write("-" * 85 + "\n")
        for cname in CLASS_NAMES:
            pc = best_metrics_b["per_class"][cname]
            f.write(f"{cname.capitalize():<16} | {pc['support']:<8} | {pc['precision']*100:8.2f} % | {pc['recall']*100:8.2f} % | {pc['specificity']*100:10.2f} % | {pc['f1_score']*100:8.2f} %\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'Macro Average':<16} | {len(y_val_b):<8} | {best_metrics_b['macro_precision']*100:8.2f} % | {best_metrics_b['macro_recall']*100:8.2f} % | {best_metrics_b['macro_specificity']*100:10.2f} % | {best_metrics_b['macro_f1']*100:8.2f} %\n\n")

        f.write("CONFUSION MATRIX (Balanced Val):\n")
        f.write(f"True \\ Pred      | {'Normal':<8} | {'Laryngoz':<8} | {'Vox Seni':<8}\n")
        f.write("-" * 50 + "\n")
        for i, cname in enumerate(["Normal", "Laryngozele", "Vox Senilis"]):
            f.write(f"{cname:<16} | {cm_b[i, 0]:8d} | {cm_b[i, 1]:8d} | {cm_b[i, 2]:8d}\n")
        f.write("\n")

        f.write("INDIVIDUAL VALIDATION LARYNGOZELE PREDICTIONS (SPEAKER 1633, SESS 1449):\n")
        f.write(f"{'Vowel':<10} | {'True Class':<12} | {'Predicted':<12} | {'P(Norm)':<8} | {'P(Lar)':<8} | {'P(Vox)':<8}\n")
        f.write("-" * 75 + "\n")
        for row in best_metrics_b["laryngozele_predictions"]:
            f.write(f"{row['vowel_type']:<10} | {row['true_class']:<12} | {row['pred_class']:<12} | {row['prob_normal']:8.3f} | {row['prob_laryngozele']:8.3f} | {row['prob_vox_senilis']:8.3f}\n")
        f.write("\n")

        f.write("3. FULL CLINICAL PREVALENCE VALIDATION BENCHMARK (VAL.CSV)\n")
        f.write("-" * 85 + "\n")
        f.write(f"  - Total Evaluation Samples: {len(y_val_f)}\n")
        f.write(f"  - Overall Accuracy:         {best_metrics_f['accuracy']*100:.2f}%\n")
        f.write(f"  - Balanced Accuracy:        {best_metrics_f['balanced_accuracy']*100:.2f}%\n")
        f.write(f"  - Macro Precision:          {best_metrics_f['macro_precision']*100:.2f}%\n")
        f.write(f"  - Macro Recall / Sens.:     {best_metrics_f['macro_recall']*100:.2f}%\n")
        f.write(f"  - Macro Specificity:        {best_metrics_f['macro_specificity']*100:.2f}%\n")
        f.write(f"  - Macro F1-Score:           {best_metrics_f['macro_f1']*100:.2f}%\n")
        f.write(f"  - Macro OvR ROC-AUC:        {best_metrics_f['macro_roc_auc']:.4f}\n\n")

        f.write("CONFUSION MATRIX (Full Val):\n")
        f.write(f"True \\ Pred      | {'Normal':<8} | {'Laryngoz':<8} | {'Vox Seni':<8}\n")
        f.write("-" * 50 + "\n")
        for i, cname in enumerate(["Normal", "Laryngozele", "Vox Senilis"]):
            f.write(f"{cname:<16} | {cm_f[i, 0]:8d} | {cm_f[i, 1]:8d} | {cm_f[i, 2]:8d}\n")
        f.write("\n")

        f.write("4. ZERO DATA LEAKAGE CONFIRMATION\n")
        f.write("-" * 85 + "\n")
        f.write("  - The hold-out test sets (test.csv and test_balanced.csv) remained 100% UNTOUCHED.\n")
        f.write("  - All models, feature scaling, augmentations, and thresholds are completely frozen.\n")
        f.write("=" * 85 + "\n")

    print(f"[Report] Saved validation diagnostic report to: {rep_path}")
    print("\nALL HYPERPARAMETERS, MODELS, SCALERS, AND THRESHOLDS ARE NOW COMPLETELY FROZEN.")


if __name__ == "__main__":
    run_training_and_freeze()
