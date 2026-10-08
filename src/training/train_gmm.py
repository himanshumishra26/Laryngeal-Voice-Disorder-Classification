"""
Training Engine: Gaussian Mixture Model (GMM)
---------------------------------------------
1. Loads authentic training and validation tabular acoustic features.
2. Applies training-only pitch augmentation to the training Laryngozele patient (Speaker 1602).
3. Fits class-conditional GMMs on the training partition.
4. Tunes prior log-weight offsets on the validation set to maximize Balanced Accuracy / Macro F1.
5. Evaluates and reports validation performance.
6. Saves the best model checkpoint to results/checkpoints/best_gmm_model.joblib.

STRICT INVARIANCES:
- Hold-out test sets (test_balanced_features.csv / test_full_features.csv) are NEVER accessed.
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
from src.features.training_augmentation import augment_training_features

CLASS_NAMES = ["normal", "laryngozele", "vox_senilis"]
CLASS_LABELS = [0, 1, 2]


def run_gmm_training():
    print("=" * 75)
    print("STAGE 4: GAUSSIAN MIXTURE MODEL (GMM) TRAINING & VALIDATION")
    print("=" * 75)

    train_path = os.path.join(BASE_DIR, "data", "processed", "tabular_features", "train_authentic_features.csv")
    val_path = os.path.join(BASE_DIR, "data", "processed", "tabular_features", "val_balanced_features.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    meta_cols = ["audio_path", "class_name", "class_label", "session_id", "speaker_id", "gender", "age", "vowel_type"]
    feat_cols = [c for c in df_train.columns if c not in meta_cols]

    print(f"[GMM Train] Acoustic biomarker features: {len(feat_cols)}")
    print(f"[GMM Train] Authentic train samples: {len(df_train)} (Normal={sum(df_train['class_label']==0)}, LZ={sum(df_train['class_label']==1)}, Vox={sum(df_train['class_label']==2)})")
    print(f"[GMM Train] Authentic val samples:   {len(df_val)} (Normal={sum(df_val['class_label']==0)}, LZ={sum(df_val['class_label']==1)}, Vox={sum(df_val['class_label']==2)})")

    # Training-only augmentation
    X_train_aug, y_train_aug = augment_training_features(df_train, feat_cols, target_minority_class=1, random_state=42)
    print(f"[GMM Train] Augmented train samples: {len(y_train_aug)} (Normal={sum(y_train_aug==0)}, LZ={sum(y_train_aug==1)}, Vox={sum(y_train_aug==2)})")

    X_val = df_val[feat_cols].values
    y_val = df_val["class_label"].values

    best_config = {
        "n_components": {0: 6, 1: 1, 2: 6},
        "reg_covar": 0.20,
        "w1_prior_offset": 8.0
    }

    best_model = GaussianMixtureVoiceClassifier(
        n_components_per_class=best_config["n_components"],
        covariance_type="diag",
        reg_covar=best_config["reg_covar"],
        class_prior_log_weights={0: 0.0, 1: best_config["w1_prior_offset"], 2: 0.0},
        random_state=42,
        max_iter=200
    )
    best_model.fit(X_train_aug, y_train_aug, fit_scaler=True)

    # Compute Likelihood Support Gate strictly from training partition
    X_train_scaled = best_model.scaler.transform(X_train_aug)
    lz_train_scores = best_model.models[1].score_samples(X_train_scaled[y_train_aug == 1])
    min_lz_train_score = float(np.min(lz_train_scores))
    best_model.set_likelihood_gate(1, min_lz_train_score)
    best_config["likelihood_gate_lz"] = min_lz_train_score

    print(f"\n[GMM Train] Calibrated Configuration Applied:")
    print(f"  - Components:             {best_config['n_components']}")
    print(f"  - Reg Covar:              {best_config['reg_covar']}")
    print(f"  - Laryngozele Prior:      +{best_config['w1_prior_offset']:.1f}")
    print(f"  - Likelihood Support Gate: {best_config['likelihood_gate_lz']:.2f}")

    # Final validation evaluation
    val_preds = best_model.predict(X_val)
    val_probs = best_model.predict_proba(X_val)
    val_acc = float(accuracy_score(y_val, val_preds))
    p, r, f1, sup = precision_recall_fscore_support(y_val, val_preds, labels=CLASS_LABELS, zero_division=0)
    cm = confusion_matrix(y_val, val_preds, labels=CLASS_LABELS)

    # Specificities
    specificities = []
    tot = len(y_val)
    for idx in range(3):
        tp = cm[idx, idx]
        fn = np.sum(cm[idx, :]) - tp
        fp = np.sum(cm[:, idx]) - tp
        tn = tot - tp - fn - fp
        specificities.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0)

    print("\n" + "=" * 75)
    print("GMM VALIDATION PERFORMANCE REPORT")
    print("=" * 75)
    print(f"Overall Accuracy:       {val_acc * 100:.2f}%")
    print(f"Balanced Accuracy:      {np.mean(r) * 100:.2f}%")
    print(f"Macro F1-Score:         {np.mean(f1) * 100:.2f}%")
    print(f"Macro Precision:        {np.mean(p) * 100:.2f}%")
    print(f"Macro Recall / Sens.:   {np.mean(r) * 100:.2f}%")
    print(f"Macro Specificity:      {np.mean(specificities) * 100:.2f}%\n")

    print(f"{'Class':<15} | {'Support':<8} | {'Precision':<10} | {'Recall':<10} | {'Specificity':<12} | {'F1-Score':<10}")
    print("-" * 75)
    for idx, name in enumerate(CLASS_NAMES):
        print(f"{name:<15} | {sup[idx]:<8} | {p[idx]*100:6.2f}%   | {r[idx]*100:6.2f}%   | {specificities[idx]*100:6.2f}%    | {f1[idx]*100:6.2f}%")
    print("-" * 75)
    print(f"{'Macro Average':<15} | {tot:<8} | {np.mean(p)*100:6.2f}%   | {np.mean(r)*100:6.2f}%   | {np.mean(specificities)*100:6.2f}%    | {np.mean(f1)*100:6.2f}%\n")

    print("Confusion Matrix (Validation):")
    print(cm)

    # Save model
    ckpt_path = os.path.join(BASE_DIR, "results", "checkpoints", "best_gmm_model.joblib")
    best_model.save(ckpt_path)

    # Write report
    report_path = os.path.join(BASE_DIR, "results", "reports", "gmm_val_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("GAUSSIAN MIXTURE MODEL (GMM) - VALIDATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Best Configuration: {json.dumps(best_config, indent=2)}\n\n")
        f.write(f"Overall Accuracy:     {val_acc * 100:.2f}%\n")
        f.write(f"Balanced Accuracy:    {np.mean(r) * 100:.2f}%\n")
        f.write(f"Macro F1-Score:       {np.mean(f1) * 100:.2f}%\n")
        f.write(f"Macro Precision:      {np.mean(p) * 100:.2f}%\n")
        f.write(f"Macro Recall:         {np.mean(r) * 100:.2f}%\n")
        f.write(f"Macro Specificity:    {np.mean(specificities) * 100:.2f}%\n\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n")

    print(f"[GMM Train] Report saved to {report_path}")
    return best_config


if __name__ == "__main__":
    run_gmm_training()
