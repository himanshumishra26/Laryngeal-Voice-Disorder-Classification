"""
Evaluation Module: Classical Acoustic SVM Baseline
---------------------------------------------------
Evaluates the trained SVM pipeline (best_svm_baseline.joblib) strictly against
the hold-out test sets (test_balanced.csv and test.csv).
NO RETRAINING OR TUNING IS PERFORMED.

Computes:
- Overall Accuracy
- Per-class and Macro Precision, Recall, Specificity, F1-Score, ROC-AUC
- Confusion matrices (raw counts & normalized)
- Individual predictions on the hold-out Laryngozele patient (Speaker 2191)
- Generates publication-quality figures:
  * results/figures/confusion_matrix_svm_balanced.png
  * results/figures/confusion_matrix_svm_full.png
  * results/figures/roc_curves_svm_balanced.png
  * results/figures/roc_curves_svm_full.png
- Saves full scientific ASCII report to results/reports/classical_baseline_test_report.txt
"""

import os
import sys
import joblib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.features.acoustic_biomarkers import extract_tabular_dataset
from src.models.classical_classifier import ACOUSTIC_FEATURE_COLS

CLASS_NAMES = ["normal", "laryngozele", "vox_senilis"]
CLASS_LABELS = [0, 1, 2]


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    title: str,
    class_names: List[str] = CLASS_NAMES
) -> None:
    """
    Plots a publication-quality confusion matrix with raw counts and normalized percentages.
    """
    cm_norm = cm.astype(float) / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized Recall Ratio", rotation=-90, va="bottom", fontsize=10)

    clean_names = [c.replace("_", " ").title() for c in class_names]
    tick_marks = np.arange(len(clean_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(clean_names, fontsize=10, fontweight="bold")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(clean_names, fontsize=10, fontweight="bold")

    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_norm = cm_norm[i, j]
            val_raw = cm[i, j]
            color = "white" if val_norm > thresh else "black"
            ax.text(
                j, i,
                f"{val_raw}\n({val_norm * 100:.1f}%)",
                ha="center", va="center",
                color=color,
                fontsize=11,
                fontweight="bold"
            )

    ax.set_ylabel("True Clinical Class", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_xlabel("Predicted Diagnosis (SVM RBF)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved confusion matrix: {save_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
    title: str,
    class_names: List[str] = CLASS_NAMES
) -> None:
    """
    Plots multi-class One-vs-Rest ROC curves.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    aucs = []
    for idx, c_name in enumerate(class_names):
        y_binary = (y_true == idx).astype(int)
        if len(np.unique(y_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_binary, y_prob[:, idx])
            roc_auc_val = auc(fpr, tpr)
            aucs.append(roc_auc_val)
            clean_name = c_name.replace("_", " ").title()
            ax.plot(fpr, tpr, label=f"{clean_name} (AUC = {roc_auc_val:.3f})", color=colors[idx], linewidth=2.2)

    macro_auc = np.mean(aucs) if len(aucs) > 0 else 0.5
    ax.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.500)", linewidth=1.5, alpha=0.7)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10, fontweight="bold")
    ax.set_title(f"{title}\nMacro ROC-AUC = {macro_auc:.3f}", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved ROC curves: {save_path}")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    df_meta: pd.DataFrame
) -> Dict[str, Any]:
    """
    Computes all diagnostic performance metrics on a test set.
    """
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_LABELS, zero_division=0)
    macro_p = float(np.mean(p))
    macro_r = float(np.mean(r))
    macro_f1 = float(np.mean(f1))
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    # Specificities
    specs = []
    for c in CLASS_LABELS:
        tn = np.sum(cm) - (np.sum(cm[c, :]) + np.sum(cm[:, c]) - cm[c, c])
        fp = np.sum(cm[:, c]) - cm[c, c]
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        specs.append(spec)
    macro_spec = float(np.mean(specs))

    # Per-class and Macro ROC-AUC
    aucs = {}
    for idx, cname in enumerate(CLASS_NAMES):
        y_bin = (y_true == idx).astype(int)
        if len(np.unique(y_bin)) > 1:
            try:
                aucs[cname] = float(roc_auc_score(y_bin, y_prob[:, idx]))
            except Exception:
                aucs[cname] = 0.5
        else:
            aucs[cname] = 0.5
    macro_auc = float(np.mean(list(aucs.values())))

    # Individual predictions for Laryngozele samples
    lar_details = []
    lar_mask = (y_true == 1)
    if np.any(lar_mask):
        sub_meta = df_meta[lar_mask]
        sub_preds = y_pred[lar_mask]
        sub_probs = y_prob[lar_mask]
        for (_, row), pred, prob in zip(sub_meta.iterrows(), sub_preds, sub_probs):
            lar_details.append({
                "vowel_type": row.get("vowel_type", ""),
                "session_id": str(row.get("session_id", "")),
                "speaker_id": str(row.get("speaker_id", "")),
                "gender": str(row.get("gender", "")),
                "age": float(row.get("age", -1.0)),
                "true_class": "laryngozele",
                "pred_class": CLASS_NAMES[pred],
                "prob_normal": float(prob[0]),
                "prob_laryngozele": float(prob[1]),
                "prob_vox_senilis": float(prob[2])
            })

    return {
        "n_samples": len(y_true),
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_specificity": macro_spec,
        "macro_f1": macro_f1,
        "macro_roc_auc": macro_auc,
        "per_class": {
            CLASS_NAMES[i]: {
                "support": int(sup[i]),
                "precision": float(p[i]),
                "recall": float(r[i]),
                "specificity": float(specs[i]),
                "f1_score": float(f1[i]),
                "roc_auc": float(aucs[CLASS_NAMES[i]])
            }
            for i in range(3)
        },
        "confusion_matrix": cm.tolist(),
        "laryngozele_predictions": lar_details
    }


def evaluate_test_sets():
    print("=" * 85)
    print("STAGE 5: CLASSICAL ACOUSTIC SVM EVALUATION ON HOLD-OUT TEST SETS")
    print("=" * 85)

    # 1. Load Trained SVM Checkpoint
    ckpt_path = os.path.join(BASE_DIR, "results", "checkpoints", "best_svm_baseline.joblib")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\n[Load] Loading trained SVM pipeline from {ckpt_path}...")
    saved_obj = joblib.load(ckpt_path)
    pipeline = saved_obj["pipeline"]
    best_params = saved_obj["best_params"]
    print(f"[Model] Loaded pipeline: SVC(kernel='rbf', C={best_params['C']}, gamma={best_params['gamma']})")

    # 2. Extract / Load Features for Test Sets (NO RETRAINING)
    tab_dir = os.path.join(BASE_DIR, "data", "processed", "tabular_features")
    os.makedirs(tab_dir, exist_ok=True)

    test_bal_csv = os.path.join(BASE_DIR, "data", "splits", "test_balanced.csv")
    test_full_csv = os.path.join(BASE_DIR, "data", "splits", "test.csv")

    test_bal_feat_csv = os.path.join(tab_dir, "test_balanced_features.csv")
    test_full_feat_csv = os.path.join(tab_dir, "test_full_features.csv")

    print("\n--- 1. EXTRACTING / CACHING TEST TABULAR FEATURES ---")
    test_bal_df = extract_tabular_dataset(test_bal_csv, test_bal_feat_csv, base_dir=BASE_DIR, max_workers=6)
    test_full_df = extract_tabular_dataset(test_full_csv, test_full_feat_csv, base_dir=BASE_DIR, max_workers=6)

    # Arrays
    X_test_bal = test_bal_df[ACOUSTIC_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test_bal = test_bal_df["class_label"].to_numpy(dtype=int)

    X_test_full = test_full_df[ACOUSTIC_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test_full = test_full_df["class_label"].to_numpy(dtype=int)

    # 3. Model Inference (Strictly predict, zero fitting)
    print("\n--- 2. RUNNING TEST SET INFERENCE (NO TUNING / NO FITTING) ---")
    preds_bal = pipeline.predict(X_test_bal)
    probs_bal = pipeline.predict_proba(X_test_bal)

    preds_full = pipeline.predict(X_test_full)
    probs_full = pipeline.predict_proba(X_test_full)

    # 4. Compute Comprehensive Metrics
    metrics_bal = compute_metrics(y_test_bal, preds_bal, probs_bal, test_bal_df)
    metrics_full = compute_metrics(y_test_full, preds_full, probs_full, test_full_df)

    # 5. Generate Figures
    figures_dir = os.path.join(BASE_DIR, "results", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    cm_bal_path = os.path.join(figures_dir, "confusion_matrix_svm_balanced.png")
    plot_confusion_matrix(
        cm=np.array(metrics_bal["confusion_matrix"]),
        save_path=cm_bal_path,
        title="SVM RBF Multi-Class Confusion Matrix\n(Balanced Test Set, N=181)"
    )

    cm_full_path = os.path.join(figures_dir, "confusion_matrix_svm_full.png")
    plot_confusion_matrix(
        cm=np.array(metrics_full["confusion_matrix"]),
        save_path=cm_full_path,
        title="SVM RBF Multi-Class Confusion Matrix\n(Full Prevalence Test Set, N=1503)"
    )

    roc_bal_path = os.path.join(figures_dir, "roc_curves_svm_balanced.png")
    plot_roc_curves(
        y_true=y_test_bal,
        y_prob=probs_bal,
        save_path=roc_bal_path,
        title="SVM RBF One-vs-Rest ROC Curves (Balanced Test Set)"
    )

    roc_full_path = os.path.join(figures_dir, "roc_curves_svm_full.png")
    plot_roc_curves(
        y_true=y_test_full,
        y_prob=probs_full,
        save_path=roc_full_path,
        title="SVM RBF One-vs-Rest ROC Curves (Full Prevalence Test Set)"
    )

    # 6. Format ASCII Report
    cm_bal = np.array(metrics_bal["confusion_matrix"])
    cm_full = np.array(metrics_full["confusion_matrix"])

    report_lines = [
        "=" * 85,
        "STAGE 5: CLASSICAL ACOUSTIC BIOMARKER SVM FINAL TEST EVALUATION REPORT",
        "Automated Multi-Class Laryngeal Voice Disorder Classification",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 85,
        "",
        "1. EVALUATED MODEL CONFIGURATION",
        "-" * 85,
        "  - Model Architecture:      Support Vector Classifier (SVC, RBF Kernel, One-vs-One)",
        f"  - Checkpoint Path:         {ckpt_path}",
        f"  - Hyperparameters:         C={best_params['C']}, gamma={best_params['gamma']}",
        "  - Feature Space:           52 Clinical Acoustic Biomarkers (F0, Jitter, Shimmer, HNR, CPP, Formants)",
        "  - Feature Preprocessor:    StandardScaler (strictly fit on train partition)",
        "  - Class Weighting:         class_weight='balanced' (cost-sensitive)",
        "  - Training Data:           Authentic samples only (0 artificial pitch-shifted replicas)",
        "",
        "2. PRIMARY BENCHMARK: BALANCED TEST SET (DEMOGRAPHIC-MATCHED, TEST_BALANCED.CSV)",
        "-" * 85,
        f"  - Total Evaluation Samples: {metrics_bal['n_samples']}",
        f"  - Overall Accuracy:         {metrics_bal['accuracy'] * 100:.2f}%",
        f"  - Macro Precision:          {metrics_bal['macro_precision'] * 100:.2f}%",
        f"  - Macro Recall / Sens.:     {metrics_bal['macro_recall'] * 100:.2f}%",
        f"  - Macro Specificity:        {metrics_bal['macro_specificity'] * 100:.2f}%",
        f"  - Macro F1-Score:           {metrics_bal['macro_f1'] * 100:.2f}%",
        f"  - Macro One-vs-Rest ROC-AUC:{metrics_bal['macro_roc_auc']:.4f}",
        "",
        "PER-CLASS DIAGNOSTIC PERFORMANCE (Balanced Test Set):",
        "Class            | Support  | Precision   | Recall      | Specificity  | F1-Score   | ROC-AUC ",
        "-" * 85,
    ]

    for cname in CLASS_NAMES:
        m = metrics_bal["per_class"][cname]
        report_lines.append(
            f"{cname.capitalize():16s} | {m['support']:8d} | {m['precision']*100:9.2f} % | {m['recall']*100:9.2f} % | {m['specificity']*100:10.2f} % | {m['f1_score']*100:8.2f} % | {m['roc_auc']:.4f}"
        )

    report_lines.extend([
        "-" * 85,
        f"{'Macro Average':<16} | {metrics_bal['n_samples']:8d} | {metrics_bal['macro_precision']*100:9.2f} % | {metrics_bal['macro_recall']*100:9.2f} % | {metrics_bal['macro_specificity']*100:10.2f} % | {metrics_bal['macro_f1']*100:8.2f} % | {metrics_bal['macro_roc_auc']:.4f}",
        "",
        "CONFUSION MATRIX (Balanced Test Set - Raw Counts):",
        "True \\ Pred      | Normal   | Laryngoz | Vox Seni",
        "-" * 85,
        f"Normal           | {cm_bal[0, 0]:8d} | {cm_bal[0, 1]:8d} | {cm_bal[0, 2]:8d}",
        f"Laryngozele      | {cm_bal[1, 0]:8d} | {cm_bal[1, 1]:8d} | {cm_bal[1, 2]:8d}",
        f"Vox Senilis      | {cm_bal[2, 0]:8d} | {cm_bal[2, 1]:8d} | {cm_bal[2, 2]:8d}",
        "",
        "3. INDIVIDUAL PREDICTIONS FOR TEST LARYNGOZELE PATIENT (SPEAKER 2191, SESS 1981)",
        "-" * 85,
        "Vowel    | True Class   | Predicted Class | P(Normal) | P(Laryngozele) | P(Vox Senilis)",
        "-" * 85,
    ])

    for row in metrics_bal["laryngozele_predictions"]:
        report_lines.append(
            f"{row['vowel_type']:8s} | {row['true_class']:12s} | {row['pred_class']:15s} | {row['prob_normal']:9.3f} | {row['prob_laryngozele']:14.3f} | {row['prob_vox_senilis']:13.3f}"
        )

    report_lines.extend([
        "",
        "4. SECONDARY BENCHMARK: FULL CLINICAL PREVALENCE TEST SET (TEST.CSV)",
        "-" * 85,
        f"  - Total Evaluation Samples: {metrics_full['n_samples']}",
        f"  - Overall Accuracy:         {metrics_full['accuracy'] * 100:.2f}%",
        f"  - Macro Precision:          {metrics_full['macro_precision'] * 100:.2f}%",
        f"  - Macro Recall / Sens.:     {metrics_full['macro_recall'] * 100:.2f}%",
        f"  - Macro Specificity:        {metrics_full['macro_specificity'] * 100:.2f}%",
        f"  - Macro F1-Score:           {metrics_full['macro_f1'] * 100:.2f}%",
        f"  - Macro One-vs-Rest ROC-AUC:{metrics_full['macro_roc_auc']:.4f}",
        "",
        "PER-CLASS DIAGNOSTIC PERFORMANCE (Full Prevalence Test Set):",
        "Class            | Support  | Precision   | Recall      | Specificity  | F1-Score   | ROC-AUC ",
        "-" * 85,
    ])

    for cname in CLASS_NAMES:
        m = metrics_full["per_class"][cname]
        report_lines.append(
            f"{cname.capitalize():16s} | {m['support']:8d} | {m['precision']*100:9.2f} % | {m['recall']*100:9.2f} % | {m['specificity']*100:10.2f} % | {m['f1_score']*100:8.2f} % | {m['roc_auc']:.4f}"
        )

    report_lines.extend([
        "-" * 85,
        f"{'Macro Average':<16} | {metrics_full['n_samples']:8d} | {metrics_full['macro_precision']*100:9.2f} % | {metrics_full['macro_recall']*100:9.2f} % | {metrics_full['macro_specificity']*100:10.2f} % | {metrics_full['macro_f1']*100:8.2f} % | {metrics_full['macro_roc_auc']:.4f}",
        "",
        "CONFUSION MATRIX (Full Prevalence Test Set - Raw Counts):",
        "True \\ Pred      | Normal   | Laryngoz | Vox Seni",
        "-" * 85,
        f"Normal           | {cm_full[0, 0]:8d} | {cm_full[0, 1]:8d} | {cm_full[0, 2]:8d}",
        f"Laryngozele      | {cm_full[1, 0]:8d} | {cm_full[1, 1]:8d} | {cm_full[1, 2]:8d}",
        f"Vox Senilis      | {cm_full[2, 0]:8d} | {cm_full[2, 1]:8d} | {cm_full[2, 2]:8d}",
        "",
        "5. ZERO DATA LEAKAGE & SCIENTIFIC RIGOR CONFIRMATION",
        "-" * 85,
        "  - The evaluated test sets were NEVER accessed or utilized during training or tuning.",
        "  - Feature standardization (StandardScaler) was fit strictly on the training partition.",
        "  - No retraining, parameter tuning, or post-hoc threshold adjustment was performed.",
        "=" * 85
    ])

    report_text = "\n".join(report_lines)
    reports_dir = os.path.join(BASE_DIR, "results", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_file = os.path.join(reports_dir, "classical_baseline_test_report.txt")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n[Report] Saved complete test evaluation report to {report_file}")
    print("\n" + report_text)


if __name__ == "__main__":
    evaluate_test_sets()
