"""
Supplementary Benchmark: 3-Fold Speaker-Independent Cross-Validation
--------------------------------------------------------------------
Executes 3-fold speaker-independent cross-validation across the three Laryngozele
patients in the Saarbrücken Voice Database (SVD), rotating each patient as the
unseen test fold while training on the other two.

Methodological Reference:
    Frontiers in Digital Health (2026) §4.2.4:
    "Where data are limited... Reporting both an in-distribution cross-validation
    result and an out-of-distribution single-shot result would substantially
    improve comparability across the literature."
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    recall_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.gmm_classifier import GaussianMixtureVoiceClassifier
from src.models.dnn_classifier import DeepNeuralNetworkVoiceClassifier
from src.features.training_augmentation import augment_training_features

CLASS_NAMES = ["normal", "laryngozele", "vox_senilis"]
CLASS_LABELS = [0, 1, 2]


def run_supplementary_cross_validation():
    print("=" * 75)
    print("SUPPLEMENTARY BENCHMARK: 3-FOLD SPEAKER-INDEPENDENT CROSS-VALIDATION")
    print("=" * 75)

    train_feat = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "tabular_features", "train_authentic_features.csv"))
    val_feat = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "tabular_features", "val_balanced_features.csv"))
    test_feat = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "tabular_features", "test_balanced_features.csv"))

    meta_cols = ["audio_path", "class_name", "class_label", "session_id", "speaker_id", "gender", "age", "vowel_type"]
    feat_cols = [c for c in train_feat.columns if c not in meta_cols]

    # Combine all balanced datasets for cross-validation partitioning
    df_all = pd.concat([train_feat, val_feat, test_feat], ignore_index=True)

    lz_speakers = [1602, 1633, 2191]
    norm_speakers = df_all[df_all["class_label"] == 0]["speaker_id"].unique()
    vox_speakers = df_all[df_all["class_label"] == 2]["speaker_id"].unique()

    np.random.seed(42)
    np.random.shuffle(norm_speakers)
    np.random.shuffle(vox_speakers)

    norm_splits = np.array_split(norm_speakers, 3)
    vox_splits = np.array_split(vox_speakers, 3)

    cv_results = {
        "GMM": [],
        "DNN": []
    }

    report_lines = []
    report_lines.append("=" * 75)
    report_lines.append("SUPPLEMENTARY BENCHMARK: 3-FOLD SPEAKER-INDEPENDENT CROSS-VALIDATION")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("Methodological Reference: Frontiers in Digital Health (2026) §4.2.4")
    report_lines.append("=" * 75 + "\n")

    for fold in range(3):
        test_lz = lz_speakers[fold]
        train_lz = [s for s in lz_speakers if s != test_lz]

        test_norm = set(norm_splits[fold])
        test_vox = set(vox_splits[fold])

        test_mask = (
            (df_all["speaker_id"] == test_lz) |
            (df_all["speaker_id"].isin(test_norm)) |
            (df_all["speaker_id"].isin(test_vox))
        )
        train_mask = ~test_mask

        df_train_fold = df_all[train_mask].copy()
        df_test_fold = df_all[test_mask].copy()

        # Augmentation strictly on training partition
        X_train_aug, y_train_aug = augment_training_features(df_train_fold, feat_cols, target_minority_class=1, random_state=42)
        X_test = df_test_fold[feat_cols].values
        y_test = df_test_fold["class_label"].values

        header = f"\n--- FOLD {fold + 1} (Test Laryngozele Patient: Speaker {test_lz} | Train Patients: {train_lz}) ---"
        print(header)
        report_lines.append(header)

        # 1. GMM
        gmm = GaussianMixtureVoiceClassifier(
            n_components_per_class={0: 4, 1: 1, 2: 4},
            covariance_type="diag",
            reg_covar=0.05,
            class_prior_log_weights={0: 0.0, 1: 10.0, 2: 0.0},
            random_state=42
        )
        gmm.fit(X_train_aug, y_train_aug, fit_scaler=True)
        gmm_preds = gmm.predict(X_test)

        gmm_acc = accuracy_score(y_test, gmm_preds)
        gmm_p, gmm_r, gmm_f1, _ = precision_recall_fscore_support(y_test, gmm_preds, labels=CLASS_LABELS, zero_division=0)
        cv_results["GMM"].append({
            "acc": float(gmm_acc),
            "macro_f1": float(np.mean(gmm_f1)),
            "bal_acc": float(np.mean(gmm_r)),
            "per_class_recall": [float(x) for x in gmm_r]
        })

        gmm_res_str = f"  GMM -> Acc: {gmm_acc*100:.2f}%, Bal Acc: {np.mean(gmm_r)*100:.2f}%, Macro F1: {np.mean(gmm_f1)*100:.2f}%, Recalls: [Normal={gmm_r[0]*100:.1f}%, LZ={gmm_r[1]*100:.1f}%, Vox={gmm_r[2]*100:.1f}%]"
        print(gmm_res_str)
        report_lines.append(gmm_res_str)

        # 2. DNN
        dnn = DeepNeuralNetworkVoiceClassifier(
            in_features=len(feat_cols),
            hidden_dims=[128, 64, 32],
            num_classes=3,
            dropout=0.35,
            learning_rate=0.001,
            batch_size=32,
            num_epochs=50,
            random_state=42 + fold
        )
        dnn.fit(X_train_aug, y_train_aug, verbose=False)
        dnn.set_class_bias(np.array([0.0, 0.15, 0.0]))
        dnn_preds = dnn.predict(X_test)

        dnn_acc = accuracy_score(y_test, dnn_preds)
        dnn_p, dnn_r, dnn_f1, _ = precision_recall_fscore_support(y_test, dnn_preds, labels=CLASS_LABELS, zero_division=0)
        cv_results["DNN"].append({
            "acc": float(dnn_acc),
            "macro_f1": float(np.mean(dnn_f1)),
            "bal_acc": float(np.mean(dnn_r)),
            "per_class_recall": [float(x) for x in dnn_r]
        })

        dnn_res_str = f"  DNN -> Acc: {dnn_acc*100:.2f}%, Bal Acc: {np.mean(dnn_r)*100:.2f}%, Macro F1: {np.mean(dnn_f1)*100:.2f}%, Recalls: [Normal={dnn_r[0]*100:.1f}%, LZ={dnn_r[1]*100:.1f}%, Vox={dnn_r[2]*100:.1f}%]"
        print(dnn_res_str)
        report_lines.append(dnn_res_str)

    # Summary
    summary_header = "\n=== 3-FOLD CROSS-VALIDATION SUMMARY (MEAN ± STD) ==="
    print(summary_header)
    report_lines.append(summary_header)

    for model_name in ["GMM", "DNN"]:
        accs = [r["acc"] for r in cv_results[model_name]]
        bal_accs = [r["bal_acc"] for r in cv_results[model_name]]
        f1s = [r["macro_f1"] for r in cv_results[model_name]]
        lz_recs = [r["per_class_recall"][1] for r in cv_results[model_name]]

        line = (
            f"  {model_name:<5} | Accuracy: {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}% | "
            f"Balanced Acc: {np.mean(bal_accs)*100:.2f} ± {np.std(bal_accs)*100:.2f}% | "
            f"Macro F1: {np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}% | "
            f"Laryngozele Recall: {np.mean(lz_recs)*100:.2f} ± {np.std(lz_recs)*100:.2f}%"
        )
        print(line)
        report_lines.append(line)

    out_file = os.path.join(BASE_DIR, "results", "reports", "supplementary_3fold_cv_report.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\n[CV] Supplementary report saved to {out_file}")

    return cv_results


if __name__ == "__main__":
    run_supplementary_cross_validation()
