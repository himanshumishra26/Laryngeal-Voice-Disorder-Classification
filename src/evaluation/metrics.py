"""
Evaluation Metrics Module
-------------------------
Calculates comprehensive diagnostic evaluation metrics for multi-class classification:
- Overall Accuracy
- Per-class and Macro/Weighted Precision
- Per-class and Macro/Weighted Recall / Sensitivity
- Per-class and Macro/Weighted Specificity (True Negative Rate)
- Per-class and Macro/Weighted F1-Score
- Confusion Matrix (raw counts and row-normalized)
- Multi-class One-vs-Rest (OvR) ROC-AUC and ROC Curve coordinates
"""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes rigorous multi-class diagnostic evaluation metrics.

    Parameters:
        y_true (np.ndarray): Ground truth integer class labels of shape (N,).
        y_pred (np.ndarray): Predicted integer class labels of shape (N,).
        y_prob (np.ndarray, optional): Predicted class probabilities of shape (N, K).
        class_names (List[str], optional): List of class string names ordered by label index.

    Returns:
        Dict[str, Any]: Fully structured, JSON-serializable dictionary of evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n_samples = len(y_true)

    # Determine unique classes
    if class_names is not None:
        num_classes = len(class_names)
        labels = list(range(num_classes))
    else:
        labels = np.unique(np.concatenate([y_true, y_pred])).tolist()
        num_classes = len(labels)
        class_names = [f"class_{i}" for i in labels]

    # Overall Accuracy
    overall_acc = float(accuracy_score(y_true, y_pred))

    # Confusion Matrix: shape (K, K) where C[i, j] is true class i predicted as class j
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total_samples = int(np.sum(cm))

    # Normalized Confusion Matrix (by true row support)
    cm_norm = np.zeros_like(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    # Per-class diagnostic metrics
    per_class_metrics = {}
    precisions = []
    recalls = []
    specificities = []
    f1_scores = []
    supports = []

    for idx, c_name in enumerate(class_names):
        tp = int(cm[idx, idx])
        fn = int(np.sum(cm[idx, :]) - tp)
        fp = int(np.sum(cm[:, idx]) - tp)
        tn = int(total_samples - tp - fn - fp)
        support = int(tp + fn)

        # Precision (PPV)
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        # Recall / Sensitivity (TPR)
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        # Specificity / Selectivity (TNR)
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # F1-Score
        f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        specificities.append(spec)
        f1_scores.append(f1)
        supports.append(support)

        per_class_metrics[c_name] = {
            "class_index": idx,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": prec,
            "recall_sensitivity": rec,
            "specificity": spec,
            "f1_score": f1
        }

    # Macro Averages (unweighted mean across classes)
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_specificity = float(np.mean(specificities))
    macro_f1 = float(np.mean(f1_scores))

    # Weighted Averages (weighted by support of each class)
    total_support = float(sum(supports))
    if total_support > 0:
        weighted_precision = float(np.average(precisions, weights=supports))
        weighted_recall = float(np.average(recalls, weights=supports))
        weighted_specificity = float(np.average(specificities, weights=supports))
        weighted_f1 = float(np.average(f1_scores, weights=supports))
    else:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_specificity = 0.0
        weighted_f1 = 0.0

    # ROC-AUC calculations (if probability outputs provided)
    roc_results = {}
    roc_curves_data = {}
    macro_roc_auc = None
    weighted_roc_auc = None

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        # Check if probabilities have appropriate shape
        if y_prob.ndim == 2 and y_prob.shape[1] == num_classes:
            per_class_aucs = []
            for idx, c_name in enumerate(class_names):
                # Binary one-vs-rest ground truth
                y_binary = (y_true == idx).astype(int)
                # Check that both classes exist in y_binary
                if len(np.unique(y_binary)) > 1:
                    cls_auc = float(roc_auc_score(y_binary, y_prob[:, idx]))
                    fpr, tpr, thresholds = roc_curve(y_binary, y_prob[:, idx])
                    roc_results[c_name] = cls_auc
                    per_class_aucs.append(cls_auc)
                    roc_curves_data[c_name] = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "auc": cls_auc
                    }
                    per_class_metrics[c_name]["roc_auc"] = cls_auc
                else:
                    roc_results[c_name] = None
                    per_class_metrics[c_name]["roc_auc"] = None

            # Multi-class OvR Macro and Weighted ROC-AUC via scikit-learn
            try:
                # Check if all classes present in y_true
                unique_true = np.unique(y_true)
                if len(unique_true) == num_classes:
                    macro_roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
                    weighted_roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
                elif len(per_class_aucs) > 0:
                    macro_roc_auc = float(np.mean(per_class_aucs))
                    valid_supports = [supports[idx] for idx, c in enumerate(class_names) if per_class_metrics[c].get("roc_auc") is not None]
                    weighted_roc_auc = float(np.average(per_class_aucs, weights=valid_supports))
            except Exception:
                if len(per_class_aucs) > 0:
                    macro_roc_auc = float(np.mean(per_class_aucs))

    return {
        "n_samples": n_samples,
        "overall_accuracy": overall_acc,
        "per_class": per_class_metrics,
        "macro_avg": {
            "precision": macro_precision,
            "recall_sensitivity": macro_recall,
            "specificity": macro_specificity,
            "f1_score": macro_f1,
            "roc_auc": macro_roc_auc
        },
        "weighted_avg": {
            "precision": weighted_precision,
            "recall_sensitivity": weighted_recall,
            "specificity": weighted_specificity,
            "f1_score": weighted_f1,
            "roc_auc": weighted_roc_auc
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "roc_curves": roc_curves_data
    }
