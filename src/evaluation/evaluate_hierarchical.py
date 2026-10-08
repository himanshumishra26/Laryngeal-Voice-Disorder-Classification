"""
Two-Stage Hierarchical Evaluation Entrypoint
--------------------------------------------
Evaluates the trained Two-Stage Hierarchical Voice Classifier against
the completely untouched hold-out test sets:
- Primary Benchmark: test_balanced.csv (N=181)
- Secondary Benchmark: test.csv (N=1503)

Computes:
- Overall Accuracy
- Per-class Precision, Recall (Sensitivity), Specificity, F1-Score, One-vs-Rest ROC-AUC
- Confusion Matrices (counts and percentages)
- Multi-class ROC curves
- Publication-quality figures & structured report serialization
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.hierarchical_classifier import HierarchicalVoiceClassifier
from src.preprocessing.dataset import SVDVoiceDataset
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.evaluate import ModelEvaluator


def run_hierarchical_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hierarchical Evaluation] Execution Device: {device}")

    ckpt_dir = os.path.join(BASE_DIR, "results", "checkpoints")
    s1_ckpt = os.path.join(ckpt_dir, "hierarchical_stage1_best.pt")
    s2_ckpt = os.path.join(ckpt_dir, "hierarchical_stage2_best.pt")

    print(f"[Hierarchical Evaluation] Loading Stage 1 Checkpoint: {s1_ckpt}")
    print(f"[Hierarchical Evaluation] Loading Stage 2 Checkpoint: {s2_ckpt}")

    hierarchical_model = HierarchicalVoiceClassifier.load_from_checkpoints(
        stage1_checkpoint_path=s1_ckpt,
        stage2_checkpoint_path=s2_ckpt,
        device=device,
        n_mels=64
    )

    class_names = ["normal", "laryngozele", "vox_senilis"]
    splits_dir = os.path.join(BASE_DIR, "data", "splits")
    figures_dir = os.path.join(BASE_DIR, "results", "figures")
    reports_dir = os.path.join(BASE_DIR, "results", "reports")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    def evaluate_test_csv(csv_path: str, split_name: str) -> Dict[str, Any]:
        ds = SVDVoiceDataset(csv_path)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        all_true = []
        all_probs_3class = []
        all_probs_s1 = []
        all_probs_s2 = []

        with torch.no_grad():
            for batch_x, batch_y, _ in loader:
                batch_x = batch_x.to(device)
                p3, p1, p2 = hierarchical_model(batch_x)
                all_probs_3class.append(p3.cpu().numpy())
                all_probs_s1.append(p1.cpu().numpy())
                all_probs_s2.append(p2.cpu().numpy())
                all_true.append(batch_y.numpy())

        y_true = np.concatenate(all_true, axis=0)
        y_prob = np.concatenate(all_probs_3class, axis=0)
        y_pred = np.argmax(y_prob, axis=1)

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=class_names
        )
        metrics["split_name"] = split_name
        return metrics, y_true, y_pred, y_prob

    results = {}

    # Dummy evaluator for plot methods
    evaluator_dummy = ModelEvaluator(
        model=hierarchical_model.stage1_model,
        device=device,
        class_names=class_names
    )

    # 1. Evaluate Primary Benchmark: Balanced Test Set
    balanced_csv = os.path.join(splits_dir, "test_balanced.csv")
    print(f"[Hierarchical Evaluation] Evaluating Primary Benchmark: {balanced_csv} (N=181)...")
    metrics_b, y_true_b, y_pred_b, y_prob_b = evaluate_test_csv(balanced_csv, "test_balanced")
    results["balanced"] = metrics_b

    # Plots for Balanced
    cm_path_b = os.path.join(figures_dir, "confusion_matrix_hierarchical_balanced.png")
    evaluator_dummy.plot_confusion_matrix(
        cm=np.array(metrics_b["confusion_matrix"]),
        save_path=cm_path_b,
        title="Two-Stage Hierarchical Confusion Matrix (Balanced Test Set)",
        normalize=True
    )
    print(f"[Hierarchical Evaluation] Saved: {cm_path_b}")

    if metrics_b.get("roc_curves"):
        roc_path_b = os.path.join(figures_dir, "roc_curves_hierarchical_balanced.png")
        evaluator_dummy.plot_roc_curves(
            roc_curves_data=metrics_b["roc_curves"],
            macro_auc=metrics_b["macro_avg"].get("roc_auc"),
            save_path=roc_path_b,
            title="Two-Stage Hierarchical ROC Curves (Balanced Test Set)"
        )
        print(f"[Hierarchical Evaluation] Saved: {roc_path_b}")

    # 2. Evaluate Secondary Benchmark: Full Clinical Prevalence Test Set
    full_csv = os.path.join(splits_dir, "test.csv")
    print(f"[Hierarchical Evaluation] Evaluating Full Prevalence Benchmark: {full_csv} (N=1503)...")
    metrics_f, y_true_f, y_pred_f, y_prob_f = evaluate_test_csv(full_csv, "test_full")
    results["full"] = metrics_f

    # Plots for Full
    cm_path_f = os.path.join(figures_dir, "confusion_matrix_hierarchical_full.png")
    evaluator_dummy.plot_confusion_matrix(
        cm=np.array(metrics_f["confusion_matrix"]),
        save_path=cm_path_f,
        title="Two-Stage Hierarchical Confusion Matrix (Full Prevalence Test Set)",
        normalize=True
    )
    print(f"[Hierarchical Evaluation] Saved: {cm_path_f}")

    if metrics_f.get("roc_curves"):
        roc_path_f = os.path.join(figures_dir, "roc_curves_hierarchical_full.png")
        evaluator_dummy.plot_roc_curves(
            roc_curves_data=metrics_f["roc_curves"],
            macro_auc=metrics_f["macro_avg"].get("roc_auc"),
            save_path=roc_path_f,
            title="Two-Stage Hierarchical ROC Curves (Full Prevalence Test Set)"
        )
        print(f"[Hierarchical Evaluation] Saved: {roc_path_f}")

    # Format text report
    ckpt_meta = {
        "path": f"Stage 1: {s1_ckpt} | Stage 2: {s2_ckpt}",
        "epoch": "Stage 1: Ep 33 | Stage 2: Ep 2",
        "val_loss": "Stage 1: 0.6392 | Stage 2: 0.4501",
        "val_accuracy": "Stage 1: 64.84% | Stage 2: 89.80%"
    }

    report_text = evaluator_dummy.format_text_report(
        metrics_balanced=metrics_b,
        metrics_full=metrics_f,
        checkpoint_meta=ckpt_meta
    )
    # Replace header
    report_text = report_text.replace("STAGE 5: FINAL MODEL EVALUATION REPORT", "STAGE 5: TWO-STAGE HIERARCHICAL MODEL EVALUATION REPORT")

    report_path = os.path.join(reports_dir, "hierarchical_evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[Hierarchical Evaluation] Report saved to: {report_path}")

    # Save JSON metrics
    json_path = os.path.join(reports_dir, "hierarchical_test_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Hierarchical Evaluation] Metrics JSON saved to: {json_path}")

    return results


if __name__ == "__main__":
    run_hierarchical_evaluation()
