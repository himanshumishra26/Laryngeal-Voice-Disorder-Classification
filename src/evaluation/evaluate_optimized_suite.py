"""
Final Test Evaluation Engine: Complete Benchmark Suite
-------------------------------------------------------
Evaluates ALL five models on the UNTOUCHED held-out test sets:
  1. GMM Baseline (best_gmm_model.joblib, permanent reference benchmark)
  2. DNN Baseline (best_dnn_model.pt)
  3. Optimized GMM (optimized_gmm_model.joblib)
  4. Optimized DNN (optimized_dnn_model.pt, ResNet-MLP + Focal Loss)
  5. Optimized GMM + DNN Ensemble (optimized_ensemble_model.joblib)

Test Benchmarks:
  - Primary Benchmark: Demographic-Matched Balanced Test Set (test_balanced_features.csv, N=181)
  - Secondary Benchmark: Full Clinical Prevalence Test Set (test_full_features.csv, N=1503)

Reports:
  - Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, Macro Specificity, Macro F1, ROC-AUC
  - Confusion Matrices
  - Per-Class Laryngozele Precision, Recall, Specificity, F1-Score
  - Per-sample predictions on Test Laryngozele Patient (Speaker 2191)

Methodological Reference:
    Frontiers in Digital Health (2026) - Voice disorders classification using ML:
    a scoping review (doi:10.3389/fdgth.2026.1800132).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.gmm_classifier import GaussianMixtureVoiceClassifier
from src.models.dnn_classifier import DeepNeuralNetworkVoiceClassifier
from src.models.gmm_dnn_ensemble import GMMDNNEnsembleClassifier

CLASS_NAMES = ["normal", "laryngozele", "vox_senilis"]
CLASS_LABELS = [0, 1, 2]


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASS_NAMES
) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    # Specificities
    tot = len(y_true)
    specificities = []
    for idx in range(len(class_names)):
        tp = cm[idx, idx]
        fn = np.sum(cm[idx, :]) - tp
        fp = np.sum(cm[:, idx]) - tp
        tn = tot - tp - fn - fp
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        specificities.append(spec)

    # ROC-AUC (OvR)
    try:
        if len(np.unique(y_true)) > 1:
            auc_scores = []
            for idx in range(len(class_names)):
                y_binary = (y_true == idx).astype(int)
                if len(np.unique(y_binary)) > 1:
                    auc_scores.append(roc_auc_score(y_binary, y_prob[:, idx]))
            macro_auc = float(np.mean(auc_scores)) if auc_scores else 0.0
        else:
            macro_auc = 0.0
    except Exception:
        macro_auc = 0.0

    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "support": int(sup[idx]),
            "precision": float(p[idx]),
            "recall": float(r[idx]),
            "specificity": float(specificities[idx]),
            "f1_score": float(f1[idx])
        }

    return {
        "accuracy": acc,
        "balanced_accuracy": float(np.mean(r)),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_specificity": float(np.mean(specificities)),
        "macro_f1": float(np.mean(f1)),
        "macro_roc_auc": macro_auc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


def run_full_suite_evaluation():
    print("=" * 85)
    print("STAGE 5: FINAL HELD-OUT TEST EVALUATION - FULL BENCHMARK SUITE")
    print("=" * 85)

    test_bal_path = os.path.join(BASE_DIR, "data", "processed", "tabular_features", "test_balanced_features.csv")
    test_full_path = os.path.join(BASE_DIR, "data", "processed", "tabular_features", "test_full_features.csv")

    df_test_bal = pd.read_csv(test_bal_path)
    df_test_full = pd.read_csv(test_full_path)

    meta_cols = ["audio_path", "class_name", "class_label", "session_id", "speaker_id", "gender", "age", "vowel_type"]
    feat_cols = [c for c in df_test_bal.columns if c not in meta_cols]

    X_bal = df_test_bal[feat_cols].values
    y_bal = df_test_bal["class_label"].values

    X_full = df_test_full[feat_cols].values
    y_full = df_test_full["class_label"].values

    print(f"[Evaluation] Balanced Test Set Samples: {len(y_bal)} (Normal={sum(y_bal==0)}, LZ={sum(y_bal==1)}, Vox={sum(y_bal==2)})")
    print(f"[Evaluation] Full Prevalence Samples:   {len(y_full)} (Normal={sum(y_full==0)}, LZ={sum(y_full==1)}, Vox={sum(y_full==2)})")

    # Models definition
    models_to_evaluate = [
        ("GMM (Baseline Reference)", "results/checkpoints/best_gmm_model.joblib", "gmm"),
        ("DNN (Baseline)", "results/checkpoints/best_dnn_model.pt", "dnn"),
        ("Optimized GMM", "results/checkpoints/optimized_gmm_model.joblib", "gmm"),
        ("Optimized DNN (ResNet-MLP)", "results/checkpoints/optimized_dnn_model.pt", "dnn"),
        ("Optimized GMM + DNN Ensemble", "results/checkpoints/optimized_ensemble_model.joblib", "ensemble"),
    ]

    results_bal = {}
    results_full = {}
    predictions_bal = {}
    probabilities_bal = {}

    for name, rel_path, mtype in models_to_evaluate:
        ckpt_path = os.path.join(BASE_DIR, rel_path)
        print(f"\n[Evaluation] Loading and evaluating: {name}...")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        if mtype == "gmm":
            model = GaussianMixtureVoiceClassifier.load(ckpt_path)
        elif mtype == "dnn":
            model = DeepNeuralNetworkVoiceClassifier.load(ckpt_path)
        elif mtype == "ensemble":
            model = GMMDNNEnsembleClassifier.load(ckpt_path)
        else:
            raise ValueError(f"Unknown model type: {mtype}")

        # Primary Benchmark: Balanced Test
        preds_b = model.predict(X_bal)
        probs_b = model.predict_proba(X_bal)
        m_b = compute_comprehensive_metrics(y_bal, preds_b, probs_b)
        results_bal[name] = m_b
        predictions_bal[name] = preds_b
        probabilities_bal[name] = probs_b

        # Secondary Benchmark: Full Prevalence
        preds_f = model.predict(X_full)
        probs_f = model.predict_proba(X_full)
        m_f = compute_comprehensive_metrics(y_full, preds_f, probs_f)
        results_full[name] = m_f

        print(f"  -> Balanced Test Acc: {m_b['accuracy']*100:.2f}%, Bal Acc: {m_b['balanced_accuracy']*100:.2f}%, Macro F1: {m_b['macro_f1']*100:.2f}%, LZ Rec: {m_b['per_class']['laryngozele']['recall']*100:.2f}%, LZ Prec: {m_b['per_class']['laryngozele']['precision']*100:.2f}%")

    # =========================================================================
    # BUILD FORMATTED COMPARATIVE REPORTS
    # =========================================================================
    lines = []
    lines.append("=" * 105)
    lines.append("FINAL HELD-OUT TEST EVALUATION: OPTIMIZATION & BENCHMARK SUITE")
    lines.append("Automated Multi-Class Laryngeal Voice Disorder Classification")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Methodological Reference: Frontiers in Digital Health (2026) - doi:10.3389/fdgth.2026.1800132")
    lines.append("=" * 105 + "\n")

    lines.append("PRIMARY BENCHMARK: BALANCED TEST SET (DEMOGRAPHIC-MATCHED, TEST_BALANCED_FEATURES.CSV, N=181)")
    lines.append("-" * 105)
    header = f"{'Model':<30} | {'Acc':<8} | {'Bal Acc':<8} | {'Macro F1':<8} | {'Macro Spec':<10} | {'ROC-AUC':<8} | {'LZ Rec':<8} | {'LZ Prec':<8} | {'LZ F1':<8}"
    lines.append(header)
    lines.append("-" * 105)

    for name, _, _ in models_to_evaluate:
        mb = results_bal[name]
        lz = mb["per_class"]["laryngozele"]
        row = (
            f"{name:<30} | {mb['accuracy']*100:6.2f}% | {mb['balanced_accuracy']*100:6.2f}% | "
            f"{mb['macro_f1']*100:6.2f}% | {mb['macro_specificity']*100:6.2f}%   | {mb['macro_roc_auc']:6.4f}   | "
            f"{lz['recall']*100:6.2f}% | {lz['precision']*100:6.2f}% | {lz['f1_score']*100:6.2f}%"
        )
        lines.append(row)

    lines.append("-" * 105 + "\n")

    lines.append("SECONDARY BENCHMARK: FULL CLINICAL PREVALENCE TEST SET (TEST_FULL_FEATURES.CSV, N=1503)")
    lines.append("-" * 105)
    header_f = f"{'Model':<30} | {'Acc':<8} | {'Bal Acc':<8} | {'Macro F1':<8} | {'Macro Spec':<10} | {'ROC-AUC':<8} | {'LZ Rec':<8} | {'LZ Prec':<8} | {'LZ F1':<8}"
    lines.append(header_f)
    lines.append("-" * 105)

    for name, _, _ in models_to_evaluate:
        mf = results_full[name]
        lz = mf["per_class"]["laryngozele"]
        row = (
            f"{name:<30} | {mf['accuracy']*100:6.2f}% | {mf['balanced_accuracy']*100:6.2f}% | "
            f"{mf['macro_f1']*100:6.2f}% | {mf['macro_specificity']*100:6.2f}%   | {mf['macro_roc_auc']:6.4f}   | "
            f"{lz['recall']*100:6.2f}% | {lz['precision']*100:6.2f}% | {lz['f1_score']*100:6.2f}%"
        )
        lines.append(row)

    lines.append("-" * 105 + "\n")

    # Detailed Confusion Matrices on Balanced Test
    lines.append("CONFUSION MATRICES ON BALANCED TEST SET (N=181):")
    lines.append("-" * 105)
    for name, _, _ in models_to_evaluate:
        cm = np.array(results_bal[name]["confusion_matrix"])
        lines.append(f"[{name}]")
        lines.append(f"                 Pred Normal   Pred Laryngoz   Pred Vox Seni")
        lines.append(f"  True Normal:        {cm[0,0]:<14} {cm[0,1]:<15} {cm[0,2]}")
        lines.append(f"  True Laryngozele:   {cm[1,0]:<14} {cm[1,1]:<15} {cm[1,2]}")
        lines.append(f"  True Vox Senilis:   {cm[2,0]:<14} {cm[2,1]:<15} {cm[2,2]}\n")

    # Test Laryngozele patient breakdown
    lines.append("TEST LARYNGOZELE PATIENT (SPEAKER 2191, SESS 1981) PREDICTIONS BREAKDOWN:")
    lines.append("-" * 105)
    lz_idx = np.where(df_test_bal["class_name"].values == "laryngozele")[0]
    lines.append(f"{'Vowel / Task':<14} | {'GMM Baseline':<14} | {'DNN Baseline':<14} | {'Optimized GMM':<14} | {'Opt DNN':<14} | {'Opt Ensemble':<14}")
    lines.append("-" * 105)
    for idx in lz_idx:
        vtask = df_test_bal.iloc[idx].get("vowel_type", "unknown")
        preds_for_vtask = [CLASS_NAMES[predictions_bal[name][idx]] for name, _, _ in models_to_evaluate]
        lines.append(f"{vtask:<14} | {preds_for_vtask[0]:<14} | {preds_for_vtask[1]:<14} | {preds_for_vtask[2]:<14} | {preds_for_vtask[3]:<14} | {preds_for_vtask[4]:<14}")
    lines.append("\n" + "=" * 105)

    report_str = "\n".join(lines)
    print("\n" + report_str)

    # Save outputs
    out_rep_path = os.path.join(BASE_DIR, "results", "reports", "final_complete_benchmark_comparison.txt")
    with open(out_rep_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"\n[Evaluation] Final comparison report written to {out_rep_path}")

    # JSON metrics
    metrics_json = {
        "timestamp": datetime.now().isoformat(),
        "balanced_test_benchmark": results_bal,
        "full_prevalence_test_benchmark": results_full
    }
    json_path = os.path.join(BASE_DIR, "results", "reports", "optimized_suite_test_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"[Evaluation] Metrics JSON saved to {json_path}")

    return metrics_json


if __name__ == "__main__":
    run_full_suite_evaluation()
