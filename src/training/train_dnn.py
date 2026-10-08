"""
Training Engine: Deep Neural Network (DNN)
------------------------------------------
1. Loads authentic training and validation tabular acoustic features.
2. Applies training-only pitch augmentation to the training Laryngozele patient (Speaker 1602).
3. Fits the multi-layer perceptron (VoiceDNNModule) with cost-sensitive loss.
4. Tunes validation decision bias / thresholds to maximize Balanced Accuracy / Macro F1.
5. Evaluates and reports validation performance.
6. Saves the best model checkpoint to results/checkpoints/best_dnn_model.pt.

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

from src.models.dnn_classifier import DeepNeuralNetworkVoiceClassifier
from src.features.training_augmentation import augment_training_features

CLASS_NAMES = ["normal", "laryngozele", "vox_senilis"]
CLASS_LABELS = [0, 1, 2]


def run_dnn_training():
    print("=" * 75)
    print("STAGE 4: DEEP NEURAL NETWORK (DNN) TRAINING & VALIDATION")
    print("=" * 75)

    train_path = os.path.join(BASE_DIR, "data", "processed", "tabular_features", "train_authentic_features.csv")
    val_path = os.path.join(BASE_DIR, "data", "processed", "tabular_features", "val_balanced_features.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    meta_cols = ["audio_path", "class_name", "class_label", "session_id", "speaker_id", "gender", "age", "vowel_type"]
    feat_cols = [c for c in df_train.columns if c not in meta_cols]

    print(f"[DNN Train] Acoustic biomarker features: {len(feat_cols)}")
    print(f"[DNN Train] Authentic train samples: {len(df_train)}")
    print(f"[DNN Train] Authentic val samples:   {len(df_val)}")

    # Training-only augmentation
    X_train_aug, y_train_aug = augment_training_features(df_train, feat_cols, target_minority_class=1, random_state=42)
    print(f"[DNN Train] Augmented train samples: {len(y_train_aug)} (Normal={sum(y_train_aug==0)}, LZ={sum(y_train_aug==1)}, Vox={sum(y_train_aug==2)})")

    X_val = df_val[feat_cols].values
    y_val = df_val["class_label"].values

    # Train DNN
    dnn = DeepNeuralNetworkVoiceClassifier(
        in_features=len(feat_cols),
        hidden_dims=[128, 64, 32],
        num_classes=3,
        dropout=0.35,
        learning_rate=0.001,
        weight_decay=1e-4,
        batch_size=32,
        num_epochs=70,
        random_state=42
    )

    dnn.fit(X_train_aug, y_train_aug, X_val=X_val, y_val=y_val, verbose=True)

    # Threshold calibration on validation set
    val_probs = dnn.predict_proba(X_val)

    best_bias = 0.0
    best_macro_f1 = -1.0
    best_preds = None

    for bias_val in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        bias_vec = np.array([0.0, bias_val, 0.0])
        preds = np.argmax(val_probs + bias_vec, axis=1)
        p, r, f1, _ = precision_recall_fscore_support(y_val, preds, labels=CLASS_LABELS, zero_division=0)
        macro_f1 = float(np.mean(f1))
        lz_rec = float(r[1])

        if macro_f1 > best_macro_f1 and lz_rec >= 0.25:
            best_macro_f1 = macro_f1
            best_bias = bias_val
            best_preds = preds

    if best_preds is None:
        best_bias = 0.0
        best_preds = np.argmax(val_probs, axis=1)

    dnn.set_class_bias(np.array([0.0, best_bias, 0.0]))
    print(f"\n[DNN Train] Selected Laryngozele Probability Bias: +{best_bias:.2f}")

    val_acc = float(accuracy_score(y_val, best_preds))
    p, r, f1, sup = precision_recall_fscore_support(y_val, best_preds, labels=CLASS_LABELS, zero_division=0)
    cm = confusion_matrix(y_val, best_preds, labels=CLASS_LABELS)

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
    print("DNN VALIDATION PERFORMANCE REPORT")
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
    ckpt_path = os.path.join(BASE_DIR, "results", "checkpoints", "best_dnn_model.pt")
    dnn.save(ckpt_path)

    # Write report
    report_path = os.path.join(BASE_DIR, "results", "reports", "dnn_val_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("DEEP NEURAL NETWORK (DNN) - VALIDATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Selected Laryngozele Bias: +{best_bias:.2f}\n\n")
        f.write(f"Overall Accuracy:     {val_acc * 100:.2f}%\n")
        f.write(f"Balanced Accuracy:    {np.mean(r) * 100:.2f}%\n")
        f.write(f"Macro F1-Score:       {np.mean(f1) * 100:.2f}%\n")
        f.write(f"Macro Precision:      {np.mean(p) * 100:.2f}%\n")
        f.write(f"Macro Recall:         {np.mean(r) * 100:.2f}%\n")
        f.write(f"Macro Specificity:    {np.mean(specificities) * 100:.2f}%\n\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n")

    print(f"[DNN Train] Report saved to {report_path}")


if __name__ == "__main__":
    run_dnn_training()
