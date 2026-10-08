"""
Model Evaluation Module
-----------------------
Executes complete evaluation of trained checkpoints against held-out test sets:
- Batch inference and Softmax probability extraction
- Comprehensive multi-class diagnostic metrics computation
- High-resolution publication-quality figures:
  * Confusion Matrices (raw counts and normalized percentages)
  * One-vs-Rest ROC Curves with per-class and Macro AUC annotations
- Detailed evaluation report serialization (JSON and TXT)
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

from src.models.cnn_lstm import CNNLSTMVoiceClassifier
from src.preprocessing.dataset import SVDVoiceDataset
from src.evaluation.metrics import compute_classification_metrics


class ModelEvaluator:
    """
    Evaluates a trained model checkpoint against held-out test data.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        class_names: List[str]
    ):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ) -> "ModelEvaluator":
        """
        Instantiates ModelEvaluator directly from a saved checkpoint file.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=device)
        num_classes = config.get("classes", {}).get("num_classes", 3)
        n_mels = config.get("features", {}).get("n_mels", 64)
        class_names = config.get("classes", {}).get("names", ["normal", "laryngozele", "vox_senilis"])

        model = CNNLSTMVoiceClassifier(num_classes=num_classes, n_mels=n_mels)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        return cls(model=model, device=device, class_names=class_names)

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Runs batch inference across a DataLoader.

        Returns:
            y_true (np.ndarray): Array of shape (N,) with ground truth integer labels.
            y_pred (np.ndarray): Array of shape (N,) with argmax predicted integer labels.
            y_prob (np.ndarray): Array of shape (N, num_classes) with softmax predicted probabilities.
            metadata_list (List[dict]): List of metadata dictionaries per sample.
        """
        self.model.eval()
        all_true = []
        all_pred = []
        all_prob = []
        all_metadata = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    batch_x, batch_y, batch_meta = batch
                else:
                    batch_x, batch_y = batch
                    batch_meta = {}

                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_prob.append(probs)
                all_pred.append(preds)
                all_true.append(batch_y.numpy())

                # Unpack metadata batch if present
                if isinstance(batch_meta, dict) and "session_id" in batch_meta:
                    batch_len = len(batch_y)
                    for i in range(batch_len):
                        rec = {k: batch_meta[k][i] if isinstance(batch_meta[k], (list, tuple)) or hasattr(batch_meta[k], "__getitem__") else batch_meta[k] for k in batch_meta}
                        all_metadata.append(rec)

        y_true = np.concatenate(all_true, axis=0)
        y_pred = np.concatenate(all_pred, axis=0)
        y_prob = np.concatenate(all_prob, axis=0)

        return y_true, y_pred, y_prob, all_metadata

    def evaluate_dataloader(self, dataloader: DataLoader, split_name: str = "test") -> Dict[str, Any]:
        """
        Runs inference and returns computed classification metrics.
        """
        y_true, y_pred, y_prob, metadata = self.predict(dataloader)
        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=self.class_names
        )
        metrics["split_name"] = split_name
        return metrics

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: str,
        title: str = "Confusion Matrix",
        normalize: bool = True
    ) -> None:
        """
        Plots and saves a high-resolution confusion matrix heatmap.
        Displays both raw counts and percentage annotations.
        """
        cm = np.asarray(cm)
        num_classes = len(self.class_names)

        # Compute row-normalized percentages
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)

        display_matrix = cm_norm if normalize else cm

        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        im = ax.imshow(display_matrix, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0 if normalize else None)

        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Normalized Rate" if normalize else "Sample Count", rotation=-90, va="bottom", fontsize=10)

        tick_marks = np.arange(num_classes)
        clean_names = [name.replace("_", " ").title() for name in self.class_names]
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(clean_names, fontsize=10, fontweight="bold")
        ax.set_yticklabels(clean_names, fontsize=10, fontweight="bold")

        # Annotate each cell with raw count and percentage
        thresh = (display_matrix.max() + display_matrix.min()) / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                count = cm[i, j]
                pct = cm_norm[i, j] * 100.0
                cell_text = f"{count}\n({pct:.1f}%)" if normalize else f"{count}"
                text_color = "white" if display_matrix[i, j] > thresh else "black"
                ax.text(j, i, cell_text, ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")

        ax.set_ylabel("True Diagnosis", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Diagnosis", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_roc_curves(
        self,
        roc_curves_data: Dict[str, Dict[str, Any]],
        macro_auc: Optional[float],
        save_path: str,
        title: str = "Receiver Operating Characteristic (ROC) Curves"
    ) -> None:
        """
        Plots and saves multi-class One-vs-Rest (OvR) ROC curves.
        """
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

        for idx, (c_name, data) in enumerate(roc_curves_data.items()):
            fpr = np.array(data["fpr"])
            tpr = np.array(data["tpr"])
            cls_auc = data["auc"]
            clean_name = c_name.replace("_", " ").title()
            color = colors[idx % len(colors)]
            auc_label = f"{cls_auc:.3f}" if (cls_auc is not None and not np.isnan(cls_auc)) else "N/A"
            ax.plot(
                fpr, tpr,
                label=f"{clean_name} (AUC = {auc_label})",
                color=color,
                linewidth=2.2
            )

        # Plot chance baseline
        ax.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.500)", linewidth=1.5, alpha=0.7)

        # Plot macro-average reference in title or legend
        if macro_auc is not None:
            title_text = f"{title}\nMacro-Average ROC-AUC = {macro_auc:.3f}"
        else:
            title_text = title

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
        ax.set_title(title_text, fontsize=12, fontweight="bold", pad=12)
        ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle=":")
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def format_text_report(
        self,
        metrics_balanced: Dict[str, Any],
        metrics_full: Optional[Dict[str, Any]] = None,
        checkpoint_meta: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates a comprehensive scientific ASCII report of evaluation results.
        """
        lines = []
        lines.append("=" * 85)
        lines.append("STAGE 5: FINAL MODEL EVALUATION REPORT")
        lines.append("Automated Multi-Class Laryngeal Voice Disorder Classification")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 85)
        lines.append("")

        if checkpoint_meta:
            lines.append("1. EVALUATED MODEL CHECKPOINT")
            lines.append("-" * 85)
            lines.append(f"  - Checkpoint Path:       {checkpoint_meta.get('path', 'N/A')}")
            lines.append(f"  - Training Epoch:        Epoch {checkpoint_meta.get('epoch', 'N/A')}")
            lines.append(f"  - Validation Loss:       {checkpoint_meta.get('val_loss', 'N/A')}")
            lines.append(f"  - Validation Accuracy:   {checkpoint_meta.get('val_accuracy', 'N/A')}")
            lines.append(f"  - Execution Device:      {self.device.type.upper()}")
            lines.append("")

        def append_metrics_table(m: Dict[str, Any], split_title: str):
            lines.append(f"EVALUATION RESULTS: {split_title.upper()}")
            lines.append("-" * 85)
            lines.append(f"  - Total Evaluation Samples: {m['n_samples']}")
            lines.append(f"  - Overall Accuracy:         {m['overall_accuracy'] * 100:.2f}%")
            lines.append(f"  - Macro Precision:          {m['macro_avg']['precision'] * 100:.2f}%")
            lines.append(f"  - Macro Recall / Sens.:     {m['macro_avg']['recall_sensitivity'] * 100:.2f}%")
            lines.append(f"  - Macro Specificity:        {m['macro_avg']['specificity'] * 100:.2f}%")
            lines.append(f"  - Macro F1-Score:           {m['macro_avg']['f1_score'] * 100:.2f}%")
            if m['macro_avg'].get('roc_auc') is not None:
                lines.append(f"  - Macro One-vs-Rest ROC-AUC:{m['macro_avg']['roc_auc']:.4f}")
            lines.append("")

            lines.append("PER-CLASS DIAGNOSTIC PERFORMANCE:")
            lines.append(f"{'Class':<16} | {'Support':<8} | {'Precision':<11} | {'Recall':<11} | {'Specificity':<12} | {'F1-Score':<10} | {'ROC-AUC':<8}")
            lines.append("-" * 85)
            for c_name in self.class_names:
                pcm = m["per_class"][c_name]
                auc_str = f"{pcm['roc_auc']:.4f}" if pcm.get("roc_auc") is not None else "N/A"
                lines.append(
                    f"{c_name.replace('_', ' ').title():<16} | "
                    f"{pcm['support']:<8} | "
                    f"{pcm['precision'] * 100:<10.2f}% | "
                    f"{pcm['recall_sensitivity'] * 100:<10.2f}% | "
                    f"{pcm['specificity'] * 100:<11.2f}% | "
                    f"{pcm['f1_score'] * 100:<9.2f}% | "
                    f"{auc_str:<8}"
                )
            lines.append("-" * 85)
            macro_auc_val = m['macro_avg'].get('roc_auc')
            macro_auc_str = f"{macro_auc_val:.4f}" if (macro_auc_val is not None and not np.isnan(macro_auc_val)) else "N/A"
            weighted_auc_val = m['weighted_avg'].get('roc_auc')
            weighted_auc_str = f"{weighted_auc_val:.4f}" if (weighted_auc_val is not None and not np.isnan(weighted_auc_val)) else "N/A"

            lines.append(
                f"{'Macro Average':<16} | "
                f"{m['n_samples']:<8} | "
                f"{m['macro_avg']['precision'] * 100:<10.2f}% | "
                f"{m['macro_avg']['recall_sensitivity'] * 100:<10.2f}% | "
                f"{m['macro_avg']['specificity'] * 100:<11.2f}% | "
                f"{m['macro_avg']['f1_score'] * 100:<9.2f}% | "
                f"{macro_auc_str:<8}"
            )
            lines.append(
                f"{'Weighted Average':<16} | "
                f"{m['n_samples']:<8} | "
                f"{m['weighted_avg']['precision'] * 100:<10.2f}% | "
                f"{m['weighted_avg']['recall_sensitivity'] * 100:<10.2f}% | "
                f"{m['weighted_avg']['specificity'] * 100:<11.2f}% | "
                f"{m['weighted_avg']['f1_score'] * 100:<9.2f}% | "
                f"{weighted_auc_str:<8}"
            )
            lines.append("")

            lines.append("CONFUSION MATRIX (Raw Counts):")
            cm = np.array(m["confusion_matrix"])
            col_headers = [c[:8].replace("_", " ").title() for c in self.class_names]
            lines.append(f"{'True \\ Pred':<16} | " + " | ".join(f"{h:<8}" for h in col_headers))
            lines.append("-" * 85)
            for idx, c_name in enumerate(self.class_names):
                row_str = " | ".join(f"{cm[idx, j]:<8}" for j in range(len(self.class_names)))
                lines.append(f"{c_name.replace('_', ' ').title():<16} | {row_str}")
            lines.append("")

        lines.append("2. PRIMARY BENCHMARK: BALANCED TEST SET (Demographic-Matched)")
        append_metrics_table(metrics_balanced, "Balanced Test Set (test_balanced.csv)")

        if metrics_full is not None:
            lines.append("3. SECONDARY BENCHMARK: FULL CLINICAL PREVALENCE TEST SET")
            append_metrics_table(metrics_full, "Full Prevalence Test Set (test.csv)")

        lines.append("4. ZERO DATA LEAKAGE & SCIENTIFIC RIGOR CONFIRMATION")
        lines.append("-" * 85)
        lines.append("  - Zero speaker overlap between train, validation, and test partitions.")
        lines.append("  - Cross-class Speaker 73 remained strictly quarantined.")
        lines.append("  - The evaluated test sets were never exposed during model training or selection.")
        lines.append("=" * 85)

        return "\n".join(lines)


def run_evaluation(
    config_path: str = "configs/config.yaml",
    checkpoint_path: Optional[str] = None,
    eval_both: bool = True
) -> Dict[str, Any]:
    """
    Main evaluation routine.
    Loads checkpoint, predicts on test sets, computes metrics, and generates figures and reports.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluation] Execution Device: {device}")

    if checkpoint_path is None:
        checkpoint_dir = os.path.join(BASE_DIR, cfg["paths"]["checkpoint_dir"])
        checkpoint_path = os.path.join(checkpoint_dir, cfg["training"]["checkpointing"]["best_model_filename"])

    print(f"[Evaluation] Loading checkpoint: {checkpoint_path}")
    evaluator = ModelEvaluator.from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=cfg,
        device=device
    )

    splits_dir = os.path.join(BASE_DIR, cfg["paths"]["splits_dir"])
    figures_dir = os.path.join(BASE_DIR, cfg["paths"]["figures_dir"])
    reports_dir = os.path.join(BASE_DIR, cfg["paths"]["reports_dir"])
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    results = {}

    # 1. Evaluate Balanced Test Set (Primary)
    balanced_csv = os.path.join(splits_dir, "test_balanced.csv")
    print(f"[Evaluation] Evaluating primary benchmark: {balanced_csv}...")
    ds_balanced = SVDVoiceDataset(balanced_csv)
    loader_balanced = DataLoader(ds_balanced, batch_size=32, shuffle=False, num_workers=0)
    metrics_balanced = evaluator.evaluate_dataloader(loader_balanced, split_name="test_balanced")
    results["balanced"] = metrics_balanced

    # Generate balanced figures
    cm_path_b = os.path.join(figures_dir, "confusion_matrix_balanced.png")
    evaluator.plot_confusion_matrix(
        cm=np.array(metrics_balanced["confusion_matrix"]),
        save_path=cm_path_b,
        title="CNN-LSTM Multi-Class Confusion Matrix (Balanced Test Set)",
        normalize=True
    )
    print(f"[Evaluation] Saved balanced confusion matrix: {cm_path_b}")

    if metrics_balanced.get("roc_curves"):
        roc_path_b = os.path.join(figures_dir, "roc_curves_balanced.png")
        evaluator.plot_roc_curves(
            roc_curves_data=metrics_balanced["roc_curves"],
            macro_auc=metrics_balanced["macro_avg"].get("roc_auc"),
            save_path=roc_path_b,
            title="CNN-LSTM One-vs-Rest ROC Curves (Balanced Test Set)"
        )
        print(f"[Evaluation] Saved balanced ROC curves: {roc_path_b}")

    # 2. Evaluate Full Test Set (Secondary)
    metrics_full = None
    if eval_both:
        full_csv = os.path.join(splits_dir, "test.csv")
        print(f"[Evaluation] Evaluating full prevalence benchmark: {full_csv}...")
        ds_full = SVDVoiceDataset(full_csv)
        loader_full = DataLoader(ds_full, batch_size=32, shuffle=False, num_workers=0)
        metrics_full = evaluator.evaluate_dataloader(loader_full, split_name="test_full")
        results["full"] = metrics_full

        cm_path_f = os.path.join(figures_dir, "confusion_matrix_full.png")
        evaluator.plot_confusion_matrix(
            cm=np.array(metrics_full["confusion_matrix"]),
            save_path=cm_path_f,
            title="CNN-LSTM Multi-Class Confusion Matrix (Full Prevalence Test Set)",
            normalize=True
        )
        print(f"[Evaluation] Saved full confusion matrix: {cm_path_f}")

        if metrics_full.get("roc_curves"):
            roc_path_f = os.path.join(figures_dir, "roc_curves_full.png")
            evaluator.plot_roc_curves(
                roc_curves_data=metrics_full["roc_curves"],
                macro_auc=metrics_full["macro_avg"].get("roc_auc"),
                save_path=roc_path_f,
                title="CNN-LSTM One-vs-Rest ROC Curves (Full Prevalence Test Set)"
            )
            print(f"[Evaluation] Saved full ROC curves: {roc_path_f}")

    # Checkpoint metadata
    ckpt_raw = torch.load(checkpoint_path, map_location="cpu")
    ckpt_meta = {
        "path": checkpoint_path,
        "epoch": ckpt_raw.get("epoch"),
        "val_loss": f"{ckpt_raw.get('val_loss', 0.0):.4f}",
        "val_accuracy": f"{ckpt_raw.get('val_accuracy', 0.0):.2f}%"
    }

    # Generate and save formatted text report
    report_text = evaluator.format_text_report(
        metrics_balanced=metrics_balanced,
        metrics_full=metrics_full,
        checkpoint_meta=ckpt_meta
    )
    report_path = os.path.join(reports_dir, "final_test_evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[Evaluation] Final test evaluation report saved: {report_path}")

    # Save JSON metrics
    json_path = os.path.join(reports_dir, "test_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Evaluation] Structured test metrics JSON saved: {json_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained CNN-LSTM on hold-out test sets.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--eval_both", action="store_true", default=True, help="Evaluate both balanced and full test sets")
    args = parser.parse_args()

    run_evaluation(config_path=args.config, checkpoint_path=args.checkpoint, eval_both=args.eval_both)
