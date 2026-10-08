"""
Production Training Engine: Optimized GMM, DNN, and Ensemble Suite
-------------------------------------------------------------------
1. Loads authentic training and validation tabular acoustic features.
2. Applies training-only pitch augmentation to the training Laryngozele patient (Speaker 1602).
3. Fits and calibrates the Optimized Gaussian Mixture Model (GMM).
4. Fits and calibrates the Optimized Deep Neural Network (ResNet-MLP with Focal Loss).
5. Fits and calibrates the Optimized GMM + DNN Probability Blending Ensemble.
6. Evaluates and reports validation performance for all three models.
7. Freezes all model weights, scalers, selectors, likelihood gates, and thresholds.
8. Saves checkpoints:
     - results/checkpoints/optimized_gmm_model.joblib
     - results/checkpoints/optimized_dnn_model.pt
     - results/checkpoints/optimized_ensemble_model.joblib

STRICT INVARIANCES:
- Held-out test sets (test_balanced_features.csv, test_full_features.csv) are NEVER accessed.
- Baseline checkpoints (best_gmm_model.joblib, best_dnn_model.pt) remain untouched.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
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
from src.models.gmm_dnn_ensemble import GMMDNNEnsembleClassifier
from src.features.training_augmentation import augment_training_features

CLASS_NAMES = ["normal", "laryngozele", "vox_senilis"]
CLASS_LABELS = [0, 1, 2]


def train_and_freeze_optimization_suite():
    print("=" * 80)
    print("STAGE 4: OPTIMIZED GMM, DNN & ENSEMBLE TRAINING & CALIBRATION")
    print("=" * 80)

    train_path = os.path.join(BASE_DIR, "data", "processed", "tabular_features", "train_authentic_features.csv")
    val_path = os.path.join(BASE_DIR, "data", "processed", "tabular_features", "val_balanced_features.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    meta_cols = ["audio_path", "class_name", "class_label", "session_id", "speaker_id", "gender", "age", "vowel_type"]
    feat_cols = [c for c in df_train.columns if c not in meta_cols]

    print(f"[Train Suite] Acoustic features: {len(feat_cols)}")
    print(f"[Train Suite] Train samples: {len(df_train)} (Normal={sum(df_train['class_label']==0)}, LZ={sum(df_train['class_label']==1)}, Vox={sum(df_train['class_label']==2)})")
    print(f"[Train Suite] Val samples:   {len(df_val)} (Normal={sum(df_val['class_label']==0)}, LZ={sum(df_val['class_label']==1)}, Vox={sum(df_val['class_label']==2)})")

    # Training-only pitch augmentation for minority class
    X_train_aug, y_train_aug = augment_training_features(df_train, feat_cols, target_minority_class=1, random_state=42)
    print(f"[Train Suite] Augmented train samples: {len(y_train_aug)} (Normal={sum(y_train_aug==0)}, LZ={sum(y_train_aug==1)}, Vox={sum(y_train_aug==2)})")

    X_val = df_val[feat_cols].values
    y_val = df_val["class_label"].values

    # =========================================================================
    # 1. OPTIMIZED GAUSSIAN MIXTURE MODEL (GMM)
    # =========================================================================
    print("\n" + "-" * 80)
    print("1. TRAINING OPTIMIZED GAUSSIAN MIXTURE MODEL (GMM)...")
    print("-" * 80)

    opt_gmm = GaussianMixtureVoiceClassifier(
        n_components_per_class={0: 6, 1: 1, 2: 6},
        covariance_type="diag",
        reg_covar=0.30,
        class_prior_log_weights={0: 0.0, 1: 7.0, 2: 0.0},
        k_best_features=35,
        random_state=42,
        max_iter=200
    )
    opt_gmm.fit(X_train_aug, y_train_aug, fit_scaler=True)

    # Compute Likelihood Support Gate strictly from training partition
    X_train_proc = opt_gmm._transform(X_train_aug)
    lz_train_scores = opt_gmm.models[1].score_samples(X_train_proc[y_train_aug == 1])
    min_lz_train_score = float(np.min(lz_train_scores))
    opt_gmm.set_likelihood_gate(1, min_lz_train_score)

    gmm_val_preds = opt_gmm.predict(X_val)
    gmm_val_probs = opt_gmm.predict_proba(X_val)
    gmm_acc = float(accuracy_score(y_val, gmm_val_preds))
    p_gmm, r_gmm, f1_gmm, _ = precision_recall_fscore_support(y_val, gmm_val_preds, labels=CLASS_LABELS, zero_division=0)
    cm_gmm = confusion_matrix(y_val, gmm_val_preds, labels=CLASS_LABELS)

    print(f"[Optimized GMM Validation Results]")
    print(f"  Overall Accuracy:    {gmm_acc*100:.2f}%")
    print(f"  Balanced Accuracy:   {np.mean(r_gmm)*100:.2f}%")
    print(f"  Macro F1-Score:      {np.mean(f1_gmm)*100:.2f}%")
    print(f"  Laryngozele Recall:  {r_gmm[1]*100:.2f}%")
    print(f"  Laryngozele Prec:    {p_gmm[1]*100:.2f}%")
    print(f"  Confusion Matrix:\n{cm_gmm}")

    gmm_ckpt = os.path.join(BASE_DIR, "results", "checkpoints", "optimized_gmm_model.joblib")
    opt_gmm.save(gmm_ckpt)

    # =========================================================================
    # 2. OPTIMIZED DEEP NEURAL NETWORK (RESNET-MLP + FOCAL LOSS)
    # =========================================================================
    print("\n" + "-" * 80)
    print("2. TRAINING OPTIMIZED DEEP NEURAL NETWORK (RESNET-MLP + FOCAL LOSS)...")
    print("-" * 80)

    opt_dnn = DeepNeuralNetworkVoiceClassifier(
        in_features=len(feat_cols),
        arch="resnet",
        hidden_dims=[128],
        num_classes=3,
        dropout=0.25,
        learning_rate=0.001,
        weight_decay=1e-4,
        loss_type="focal",
        focal_gamma=1.5,
        scheduler_type="cosine",
        k_best_features=None,
        batch_size=64,
        num_epochs=50,
        random_state=42
    )

    opt_dnn.fit(
        X_train_aug,
        y_train_aug,
        X_val=X_val,
        y_val=y_val,
        seeds=[42, 123, 777],
        verbose=False
    )

    # Decision threshold calibration strictly on validation set
    dnn_raw_probs = opt_dnn.predict_proba(X_val)
    best_dnn_bias = 0.0
    best_dnn_bal_acc = -1.0
    best_dnn_preds = None
    best_p_dnn, best_r_dnn, best_f1_dnn = None, None, None

    for bias_val in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        preds = np.argmax(dnn_raw_probs + np.array([0.0, bias_val, 0.0]), axis=1)
        p, r, f1, _ = precision_recall_fscore_support(y_val, preds, labels=CLASS_LABELS, zero_division=0)
        bal_acc = float(np.mean(r))
        if bal_acc > best_dnn_bal_acc:
            best_dnn_bal_acc = bal_acc
            best_dnn_bias = bias_val
            best_dnn_preds = preds
            best_p_dnn, best_r_dnn, best_f1_dnn = p, r, f1

    opt_dnn.set_class_bias(np.array([0.0, best_dnn_bias, 0.0]))
    dnn_acc = float(accuracy_score(y_val, best_dnn_preds))
    cm_dnn = confusion_matrix(y_val, best_dnn_preds, labels=CLASS_LABELS)

    print(f"[Optimized DNN Validation Results]")
    print(f"  Selected LZ Bias:    +{best_dnn_bias:.2f}")
    print(f"  Overall Accuracy:    {dnn_acc*100:.2f}%")
    print(f"  Balanced Accuracy:   {np.mean(best_r_dnn)*100:.2f}%")
    print(f"  Macro F1-Score:      {np.mean(best_f1_dnn)*100:.2f}%")
    print(f"  Laryngozele Recall:  {best_r_dnn[1]*100:.2f}%")
    print(f"  Laryngozele Prec:    {best_p_dnn[1]*100:.2f}%")
    print(f"  Confusion Matrix:\n{cm_dnn}")

    dnn_ckpt = os.path.join(BASE_DIR, "results", "checkpoints", "optimized_dnn_model.pt")
    opt_dnn.save(dnn_ckpt)

    # =========================================================================
    # 3. OPTIMIZED GMM + DNN PROBABILITY BLENDING ENSEMBLE
    # =========================================================================
    print("\n" + "-" * 80)
    print("3. BUILDING OPTIMIZED GMM + DNN PROBABILITY BLENDING ENSEMBLE...")
    print("-" * 80)

    best_ens_score = -1.0
    best_ens_alpha = 0.25
    best_ens_bias = 0.10
    best_ens_preds = None
    best_p_ens, best_r_ens, best_f1_ens = None, None, None

    for alpha in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
        blend_probs = alpha * gmm_val_probs + (1.0 - alpha) * dnn_raw_probs
        for bias_val in [0.0, 0.05, 0.08, 0.10, 0.15]:
            preds = np.argmax(blend_probs + np.array([0.0, bias_val, 0.0]), axis=1)
            p, r, f1, _ = precision_recall_fscore_support(y_val, preds, labels=CLASS_LABELS, zero_division=0)
            bal_acc = float(np.mean(r))
            if bal_acc > best_ens_score:
                best_ens_score = bal_acc
                best_ens_alpha = alpha
                best_ens_bias = bias_val
                best_ens_preds = preds
                best_p_ens, best_r_ens, best_f1_ens = p, r, f1

    opt_ens = GMMDNNEnsembleClassifier(
        gmm_model=opt_gmm,
        dnn_model=opt_dnn,
        alpha=best_ens_alpha,
        class_bias=np.array([0.0, best_ens_bias, 0.0])
    )

    ens_acc = float(accuracy_score(y_val, best_ens_preds))
    cm_ens = confusion_matrix(y_val, best_ens_preds, labels=CLASS_LABELS)

    print(f"[Optimized Ensemble Validation Results]")
    print(f"  Optimal Alpha (GMM): {best_ens_alpha:.2f} (DNN weight: {1.0 - best_ens_alpha:.2f})")
    print(f"  Selected LZ Bias:    +{best_ens_bias:.2f}")
    print(f"  Overall Accuracy:    {ens_acc*100:.2f}%")
    print(f"  Balanced Accuracy:   {np.mean(best_r_ens)*100:.2f}%")
    print(f"  Macro F1-Score:      {np.mean(best_f1_ens)*100:.2f}%")
    print(f"  Laryngozele Recall:  {best_r_ens[1]*100:.2f}%")
    print(f"  Laryngozele Prec:    {best_p_ens[1]*100:.2f}%")
    print(f"  Confusion Matrix:\n{cm_ens}")

    ens_ckpt = os.path.join(BASE_DIR, "results", "checkpoints", "optimized_ensemble_model.joblib")
    opt_ens.save(ens_ckpt)

    # =========================================================================
    # 4. WRITE VALIDATION COMPARISON REPORT
    # =========================================================================
    rep_lines = []
    rep_lines.append("=" * 85)
    rep_lines.append("VALIDATION BENCHMARK REPORT: OPTIMIZATION SUITE")
    rep_lines.append("Automated Multi-Class Laryngeal Voice Disorder Classification")
    rep_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rep_lines.append("=" * 85 + "\n")

    rep_lines.append("1. VALIDATION PERFORMANCE SUMMARY (DEMOGRAPHIC-MATCHED VALIDATION PARTITION, N=182)")
    rep_lines.append("-" * 85)
    rep_lines.append(f"{'Metric':<25} | {'Optimized GMM':<16} | {'Optimized DNN':<16} | {'Optimized Ensemble':<18}")
    rep_lines.append("-" * 85)
    rep_lines.append(f"{'Overall Accuracy':<25} | {gmm_acc*100:6.2f}%          | {dnn_acc*100:6.2f}%          | {ens_acc*100:6.2f}%")
    rep_lines.append(f"{'Balanced Accuracy':<25} | {np.mean(r_gmm)*100:6.2f}%          | {np.mean(best_r_dnn)*100:6.2f}%          | {np.mean(best_r_ens)*100:6.2f}%")
    rep_lines.append(f"{'Macro F1-Score':<25} | {np.mean(f1_gmm)*100:6.2f}%          | {np.mean(best_f1_dnn)*100:6.2f}%          | {np.mean(best_f1_ens)*100:6.2f}%")
    rep_lines.append(f"{'Laryngozele Recall':<25} | {r_gmm[1]*100:6.2f}%          | {best_r_dnn[1]*100:6.2f}%          | {best_r_ens[1]*100:6.2f}%")
    rep_lines.append(f"{'Laryngozele Precision':<25} | {p_gmm[1]*100:6.2f}%          | {best_p_dnn[1]*100:6.2f}%          | {best_p_ens[1]*100:6.2f}%\n")

    rep_lines.append("2. CONFUSION MATRICES (Validation Partition):")
    rep_lines.append("Optimized GMM:\n" + str(cm_gmm) + "\n")
    rep_lines.append("Optimized DNN (ResNet-MLP):\n" + str(cm_dnn) + "\n")
    rep_lines.append("Optimized Ensemble:\n" + str(cm_ens) + "\n")

    rep_lines.append("3. SAVED FROZEN CHECKPOINTS:")
    rep_lines.append(f"  - GMM:      {gmm_ckpt}")
    rep_lines.append(f"  - DNN:      {dnn_ckpt}")
    rep_lines.append(f"  - Ensemble: {ens_ckpt}\n")
    rep_lines.append("=" * 85)

    rep_path = os.path.join(BASE_DIR, "results", "reports", "optimization_validation_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rep_lines))
    print(f"\n[Train Suite] Validation report written to {rep_path}")
    print("\n" + "\n".join(rep_lines))


if __name__ == "__main__":
    train_and_freeze_optimization_suite()
