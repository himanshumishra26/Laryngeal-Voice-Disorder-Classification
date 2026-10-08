"""
Training and Validation Engine: Classical SVM Acoustic Baseline
---------------------------------------------------------------
1. Extracts tabular acoustic biomarkers for training and validation splits.
2. Fits an RBF-kernel SVM with balanced class weighting on the training set.
3. Tunes hyperparameters (C, gamma) against the validation partition.
4. Evaluates comprehensive multi-class metrics (Accuracy, Macro F1, Recall, Confusion Matrix)
   strictly on validation data.
5. Saves the best model checkpoint and generates a diagnostic report.

STRICT INVARIANCES:
- ZERO access to or modification of hold-out test sets (test.csv / test_balanced.csv).
- Data scaling (StandardScaler) is strictly fit on the training partition only.
"""

import os
import sys
import json
import joblib
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.features.acoustic_biomarkers import extract_tabular_dataset
from src.models.classical_classifier import create_svm_pipeline, ACOUSTIC_FEATURE_COLS


CLASS_NAMES = ["normal", "laryngozele", "vox_senilis"]
CLASS_LABELS = [0, 1, 2]


def build_authentic_train_manifest(base_dir: str = BASE_DIR) -> str:
    """
    Builds a clean, authentic demographic-matched training manifest:
    - 26 Normal speakers (13 male, 13 female: ~363 samples)
    - 26 Vox Senilis speakers (366 samples)
    - 1 Laryngozele speaker (14 authentic samples, ZERO artificial pitch-shift duplicates)
    """
    out_manifest = os.path.join(base_dir, "data", "splits", "train_authentic_matched.csv")
    if os.path.exists(out_manifest) and os.path.getsize(out_manifest) > 1000:
        return out_manifest

    train_full = pd.read_csv(os.path.join(base_dir, "data", "splits", "train.csv"))
    
    # 1. 14 authentic Laryngozele samples
    lar_train = train_full[train_full["class_name"] == "laryngozele"].copy()
    
    # 2. 26 Vox Senilis speakers
    vox_train = train_full[train_full["class_name"] == "vox_senilis"].copy()
    
    # 3. 26 Normal speakers (13 male, 13 female)
    norm_train = train_full[train_full["class_name"] == "normal"].copy()
    male_spks = sorted(norm_train[norm_train["gender"] == "m"]["speaker_id"].unique())
    female_spks = sorted(norm_train[norm_train["gender"] == "w"]["speaker_id"].unique())
    
    np.random.seed(42)
    sel_male = np.random.choice(male_spks, size=min(13, len(male_spks)), replace=False)
    sel_female = np.random.choice(female_spks, size=min(13, len(female_spks)), replace=False)
    sel_norm_spks = list(sel_male) + list(sel_female)
    
    norm_matched = norm_train[norm_train["speaker_id"].isin(sel_norm_spks)].copy()
    
    combined = pd.concat([vox_train, lar_train, norm_matched], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    combined.to_csv(out_manifest, index=False)
    print(f"[TrainEngine] Created authentic training manifest: {out_manifest} ({len(combined)} samples)")
    return out_manifest


def evaluate_model(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    df_meta: pd.DataFrame
) -> Dict[str, Any]:
    """
    Computes complete multi-class diagnostic metrics on a validation set.
    """
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)
    acc = float(accuracy_score(y, preds))

    p, r, f1, sup = precision_recall_fscore_support(y, preds, labels=CLASS_LABELS, zero_division=0)
    macro_p = float(np.mean(p))
    macro_r = float(np.mean(r))
    macro_f1 = float(np.mean(f1))
    cm = confusion_matrix(y, preds, labels=CLASS_LABELS)

    # Multi-class ROC-AUC (OvR)
    try:
        roc_auc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    except Exception:
        roc_auc = 0.5

    # Specificity per class: TN / (TN + FP)
    specs = []
    for c in CLASS_LABELS:
        tn = np.sum(cm) - (np.sum(cm[c, :]) + np.sum(cm[:, c]) - cm[c, c])
        fp = np.sum(cm[:, c]) - cm[c, c]
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        specs.append(spec)
    macro_spec = float(np.mean(specs))

    # Detailed inspection of Laryngozele samples
    lar_details = []
    lar_mask = (y == 1)
    if np.any(lar_mask):
        sub_meta = df_meta[lar_mask]
        sub_preds = preds[lar_mask]
        sub_probs = probs[lar_mask]
        for (_, row), pred, prob in zip(sub_meta.iterrows(), sub_preds, sub_probs):
            lar_details.append({
                "vowel_type": row.get("vowel_type", ""),
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


def run_classical_training_and_tuning():
    print("=" * 80)
    print("STAGE 4/5 OPTION B: CLASSICAL ACOUSTIC SVM BASELINE PIPELINE")
    print("=" * 80)

    # 1. Prepare Manifests (HOLD-OUT TEST SET IS STRICTLY EXCLUDED)
    train_manifest = build_authentic_train_manifest(BASE_DIR)
    val_bal_manifest = os.path.join(BASE_DIR, "data", "splits", "val_balanced.csv")
    val_full_manifest = os.path.join(BASE_DIR, "data", "splits", "val.csv")

    # 2. Extract / Load Tabular Biomarkers
    tab_dir = os.path.join(BASE_DIR, "data", "processed", "tabular_features")
    os.makedirs(tab_dir, exist_ok=True)
    
    train_feat_csv = os.path.join(tab_dir, "train_authentic_features.csv")
    val_bal_feat_csv = os.path.join(tab_dir, "val_balanced_features.csv")
    val_full_feat_csv = os.path.join(tab_dir, "val_full_features.csv")

    print("\n--- 1. FEATURE EXTRACTION / CACHE CHECK ---")
    train_df = extract_tabular_dataset(train_manifest, train_feat_csv, base_dir=BASE_DIR, max_workers=6)
    val_bal_df = extract_tabular_dataset(val_bal_manifest, val_bal_feat_csv, base_dir=BASE_DIR, max_workers=6)
    val_full_df = extract_tabular_dataset(val_full_manifest, val_full_feat_csv, base_dir=BASE_DIR, max_workers=6)

    # Prepare numpy arrays
    X_train = train_df[ACOUSTIC_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df["class_label"].to_numpy(dtype=int)

    X_val_bal = val_bal_df[ACOUSTIC_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_val_bal = val_bal_df["class_label"].to_numpy(dtype=int)

    X_val_full = val_full_df[ACOUSTIC_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_val_full = val_full_df["class_label"].to_numpy(dtype=int)

    print(f"\nFeature Arrays Summary:")
    print(f"  X_train:    {X_train.shape} (Classes: {dict(pd.Series(y_train).value_counts())})")
    print(f"  X_val_bal:  {X_val_bal.shape} (Classes: {dict(pd.Series(y_val_bal).value_counts())})")
    print(f"  X_val_full: {X_val_full.shape} (Classes: {dict(pd.Series(y_val_full).value_counts())})")

    # 3. Hyperparameter Tuning Grid
    print("\n--- 2. HYPERPARAMETER TUNING ON VALIDATION DATA ---")
    param_grid = [
        {"C": 0.1, "gamma": "scale"},
        {"C": 0.5, "gamma": "scale"},
        {"C": 1.0, "gamma": "scale"},
        {"C": 2.0, "gamma": "scale"},
        {"C": 5.0, "gamma": "scale"},
        {"C": 10.0, "gamma": "scale"},
        {"C": 0.5, "gamma": "auto"},
        {"C": 1.0, "gamma": "auto"},
        {"C": 2.0, "gamma": "auto"},
        {"C": 1.0, "gamma": 0.01},
        {"C": 2.0, "gamma": 0.01},
        {"C": 5.0, "gamma": 0.01},
    ]

    best_pipeline = None
    best_params = None
    best_score = -1.0
    best_metrics_bal = None

    print(f"{'C':>6} | {'gamma':>8} | {'Val Acc':>9} | {'Macro P':>9} | {'Macro R':>9} | {'Macro F1':>9} | {'Lary Rec':>9}")
    print("-" * 75)

    for params in param_grid:
        c_val = params["C"]
        gamma_val = params["gamma"]

        pipeline = create_svm_pipeline(C=c_val, gamma=gamma_val, class_weight="balanced")
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_val_bal, y_val_bal, val_bal_df)
        macro_f1 = metrics["macro_f1"]
        lary_recall = metrics["per_class"]["laryngozele"]["recall"]

        # Selection score balances overall macro F1 and minority recall
        selection_score = 0.5 * macro_f1 + 0.5 * lary_recall

        print(f"{c_val:6.1f} | {str(gamma_val):>8} | {metrics['accuracy']*100:8.2f}% | {metrics['macro_precision']*100:8.2f}% | {metrics['macro_recall']*100:8.2f}% | {macro_f1*100:8.2f}% | {lary_recall*100:8.2f}%")

        if selection_score > best_score:
            best_score = selection_score
            best_pipeline = pipeline
            best_params = params
            best_metrics_bal = metrics

    print("\n" + "=" * 80)
    print(f"BEST HYPERPARAMETERS: C={best_params['C']}, gamma={best_params['gamma']}")
    print(f"Best Balanced Val Macro F1: {best_metrics_bal['macro_f1']*100:.2f}%")
    print(f"Best Laryngozele Recall:     {best_metrics_bal['per_class']['laryngozele']['recall']*100:.2f}%")
    print("=" * 80)

    # 4. Evaluate Best Pipeline on Full Validation as well
    best_metrics_full = evaluate_model(best_pipeline, X_val_full, y_val_full, val_full_df)

    # 5. Save Model Checkpoint
    ckpt_dir = os.path.join(BASE_DIR, "results", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best_svm_baseline.joblib")
    joblib.dump({
        "pipeline": best_pipeline,
        "best_params": best_params,
        "feature_cols": ACOUSTIC_FEATURE_COLS,
        "val_balanced_metrics": best_metrics_bal,
        "val_full_metrics": best_metrics_full
    }, ckpt_path)
    print(f"\n[Checkpoint] Saved best SVM pipeline to {ckpt_path}")

    # 6. Generate Comprehensive Validation Diagnostic Report
    reports_dir = os.path.join(BASE_DIR, "results", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "classical_baseline_val_report.txt")

    cm_bal = np.array(best_metrics_bal["confusion_matrix"])
    cm_full = np.array(best_metrics_full["confusion_matrix"])

    report_lines = [
        "=" * 85,
        "STAGE 4/5 OPTION B: CLASSICAL ACOUSTIC BIOMARKER SVM VALIDATION REPORT",
        "Automated Multi-Class Laryngeal Voice Disorder Classification",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 85,
        "",
        "1. MODEL ARCHITECTURE & PIPELINE CONFIGURATION",
        "-" * 85,
        "  - Feature Extractor:       AcousticBiomarkerExtractor (52 clinical features)",
        "  - Preprocessing / Scaling: StandardScaler (strictly fit on train partition)",
        "  - Classification Model:    Support Vector Classifier (SVC, RBF Kernel, One-vs-One)",
        f"  - Best Hyperparameters:    C={best_params['C']}, gamma={best_params['gamma']}",
        "  - Cost-Sensitive Weight:   class_weight='balanced' (inversely proportional to frequency)",
        "  - Probability Calibration: Platt scaling (probability=True)",
        "  - Hold-Out Test Sets:      STRICTLY UNTOUCHED & UNACCESSED",
        "",
        "2. PRIMARY BENCHMARK: BALANCED VALIDATION SET (VAL_BALANCED.CSV)",
        "-" * 85,
        f"  - Total Evaluation Samples: {len(y_val_bal)}",
        f"  - Overall Accuracy:         {best_metrics_bal['accuracy']*100:.2f}%",
        f"  - Macro Precision:          {best_metrics_bal['macro_precision']*100:.2f}%",
        f"  - Macro Recall / Sens.:     {best_metrics_bal['macro_recall']*100:.2f}%",
        f"  - Macro Specificity:        {best_metrics_bal['macro_specificity']*100:.2f}%",
        f"  - Macro F1-Score:           {best_metrics_bal['macro_f1']*100:.2f}%",
        f"  - Macro One-vs-Rest ROC-AUC:{best_metrics_bal['macro_roc_auc']:.4f}",
        "",
        "PER-CLASS VALIDATION PERFORMANCE (Balanced Val):",
        "Class            | Support  | Precision   | Recall      | Specificity  | F1-Score   ",
        "-" * 85,
    ]

    for cname in CLASS_NAMES:
        m = best_metrics_bal["per_class"][cname]
        report_lines.append(
            f"{cname.capitalize():16s} | {m['support']:8d} | {m['precision']*100:9.2f} % | {m['recall']*100:9.2f} % | {m['specificity']*100:10.2f} % | {m['f1_score']*100:8.2f} %"
        )

    report_lines.extend([
        "-" * 85,
        f"Macro Average    | {len(y_val_bal):8d} | {best_metrics_bal['macro_precision']*100:9.2f} % | {best_metrics_bal['macro_recall']*100:9.2f} % | {best_metrics_bal['macro_specificity']*100:10.2f} % | {best_metrics_bal['macro_f1']*100:8.2f} %",
        "",
        "CONFUSION MATRIX (Balanced Val - Raw Counts):",
        "True \\ Pred      | Normal   | Laryngoz | Vox Seni",
        "-" * 85,
        f"Normal           | {cm_bal[0, 0]:8d} | {cm_bal[0, 1]:8d} | {cm_bal[0, 2]:8d}",
        f"Laryngozele      | {cm_bal[1, 0]:8d} | {cm_bal[1, 1]:8d} | {cm_bal[1, 2]:8d}",
        f"Vox Senilis      | {cm_bal[2, 0]:8d} | {cm_bal[2, 1]:8d} | {cm_bal[2, 2]:8d}",
        "",
        "3. INDIVIDUAL PREDICTIONS FOR VALIDATION LARYNGOZELE (SPEAKER 1633, SESS 1449)",
        "-" * 85,
        "Vowel    | True Class   | Predicted Class | P(Normal) | P(Laryngozele) | P(Vox Senilis)",
        "-" * 85,
    ])

    for row in best_metrics_bal["laryngozele_predictions"]:
        report_lines.append(
            f"{row['vowel_type']:8s} | {row['true_class']:12s} | {row['pred_class']:15s} | {row['prob_normal']:9.3f} | {row['prob_laryngozele']:14.3f} | {row['prob_vox_senilis']:13.3f}"
        )

    report_lines.extend([
        "",
        "4. SECONDARY BENCHMARK: FULL VALIDATION SET (VAL.CSV)",
        "-" * 85,
        f"  - Total Evaluation Samples: {len(y_val_full)}",
        f"  - Overall Accuracy:         {best_metrics_full['accuracy']*100:.2f}%",
        f"  - Macro Precision:          {best_metrics_full['macro_precision']*100:.2f}%",
        f"  - Macro Recall / Sens.:     {best_metrics_full['macro_recall']*100:.2f}%",
        f"  - Macro Specificity:        {best_metrics_full['macro_specificity']*100:.2f}%",
        f"  - Macro F1-Score:           {best_metrics_full['macro_f1']*100:.2f}%",
        f"  - Macro One-vs-Rest ROC-AUC:{best_metrics_full['macro_roc_auc']:.4f}",
        "",
        "CONFUSION MATRIX (Full Val - Raw Counts):",
        "True \\ Pred      | Normal   | Laryngoz | Vox Seni",
        "-" * 85,
        f"Normal           | {cm_full[0, 0]:8d} | {cm_full[0, 1]:8d} | {cm_full[0, 2]:8d}",
        f"Laryngozele      | {cm_full[1, 0]:8d} | {cm_full[1, 1]:8d} | {cm_full[1, 2]:8d}",
        f"Vox Senilis      | {cm_full[2, 0]:8d} | {cm_full[2, 1]:8d} | {cm_full[2, 2]:8d}",
        "",
        "5. SCIENTIFIC INTEGRITY CONFIRMATION",
        "-" * 85,
        "  - The hold-out test sets (test.csv and test_balanced.csv) remained 100% UNTOUCHED.",
        "  - Zero data leakage: Feature scaling and OvO decision boundaries fit only on train.",
        "  - Zero synthetic pitch-shift sample replication used.",
        "=" * 85
    ])

    report_text = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[Report] Wrote comprehensive validation report to {report_path}")
    print("\n" + report_text)


if __name__ == "__main__":
    run_classical_training_and_tuning()
