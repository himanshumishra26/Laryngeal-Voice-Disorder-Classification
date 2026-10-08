"""
Two-Stage Hierarchical Training Entrypoint
------------------------------------------
Trains both specialized stages independently:
- Stage 1: Healthy (0) vs. Pathological (1)
- Stage 2: Laryngozele (0) vs. Vox Senilis (1)

Saves checkpoints:
- results/checkpoints/hierarchical_stage1_best.pt
- results/checkpoints/hierarchical_stage2_best.pt

STRICT INVARIANCE:
- Test sets remain 100% UNTOUCHED and UNSEEN.
- Only uses train_stage1/2.csv and val_stage1/2.csv.
"""

import os
import sys
import json
import yaml
import torch
import torch.nn as nn
from datetime import datetime
from typing import Tuple, Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.cnn_lstm import CNNLSTMVoiceClassifier
from src.training.trainer import ModelTrainer
from src.preprocessing.dataset import SVDVoiceDataset, SpecAugment


def train_single_stage(
    stage_num: int,
    stage_name: str,
    train_csv: str,
    val_csv: str,
    num_classes: int,
    epochs: int,
    batch_size: int,
    monitor_metric: str,
    monitor_mode: str,
    checkpoint_filename: str,
    device: torch.device
) -> Tuple[dict, str]:
    print(f"\n{'='*80}")
    print(f"TRAINING HIERARCHICAL STAGE {stage_num}: {stage_name.upper()}")
    print(f"{'='*80}")
    print(f"  Train CSV: {train_csv}")
    print(f"  Val CSV:   {val_csv}")

    train_ds = SVDVoiceDataset(train_csv, transform=SpecAugment())
    val_ds = SVDVoiceDataset(val_csv)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"  Train samples: {len(train_ds)} ({len(train_loader)} batches) | Val samples: {len(val_ds)} ({len(val_loader)} batches)")

    model = CNNLSTMVoiceClassifier(num_classes=num_classes, n_mels=64)

    config = {
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "label_smoothing": 0.05,
            "gradient_clip_norm": 5.0,
            "scheduler": {
                "mode": "min",
                "factor": 0.5,
                "patience": 5,
                "min_lr": 0.00001
            },
            "early_stopping": {
                "enabled": True,
                "patience": 12,
                "min_delta": 0.0001,
                "monitor": monitor_metric,
                "mode": monitor_mode
            },
            "checkpointing": {
                "enabled": True,
                "best_model_filename": checkpoint_filename,
                "last_model_filename": f"last_{checkpoint_filename}"
            }
        },
        "paths": {
            "checkpoint_dir": "results/checkpoints"
        }
    }

    trainer = ModelTrainer(model=model, device=device, config=config)
    history = trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=epochs, verbose=True)

    return history, trainer.best_model_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hierarchical Training] Execution Device: {device}")

    splits_dir = os.path.join(BASE_DIR, "data", "splits", "hierarchical")

    # 1. Train Stage 1: Healthy (0) vs Pathological (1)
    train_s1 = os.path.join(splits_dir, "train_stage1.csv")
    val_s1 = os.path.join(splits_dir, "val_stage1.csv")
    history_s1, ckpt_s1 = train_single_stage(
        stage_num=1,
        stage_name="Healthy vs. Pathological",
        train_csv=train_s1,
        val_csv=val_s1,
        num_classes=2,
        epochs=35,
        batch_size=32,
        monitor_metric="val_macro_f1",
        monitor_mode="max",
        checkpoint_filename="hierarchical_stage1_best.pt",
        device=device
    )

    # 2. Train Stage 2: Laryngozele (0) vs Vox Senilis (1)
    train_s2 = os.path.join(splits_dir, "train_stage2.csv")
    val_s2 = os.path.join(splits_dir, "val_stage2.csv")
    history_s2, ckpt_s2 = train_single_stage(
        stage_num=2,
        stage_name="Laryngozele vs. Vox Senilis",
        train_csv=train_s2,
        val_csv=val_s2,
        num_classes=2,
        epochs=35,
        batch_size=32,
        monitor_metric="val_macro_f1",
        monitor_mode="max",
        checkpoint_filename="hierarchical_stage2_best.pt",
        device=device
    )

    # Save histories
    hist_path = os.path.join(BASE_DIR, "results", "reports", "hierarchical_training_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({"stage1": history_s1, "stage2": history_s2}, f, indent=2)
    print(f"\n[Hierarchical Training] Histories saved to: {hist_path}")

    # Plot progression
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    # Stage 1 Loss & Acc
    ep1 = range(1, len(history_s1["train_loss"]) + 1)
    axes[0, 0].plot(ep1, history_s1["train_loss"], label="Train Loss")
    axes[0, 0].plot(ep1, history_s1["val_loss"], label="Val Loss", linestyle="--")
    axes[0, 0].set_title("Stage 1: Healthy vs Pathological - Loss", fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ep1, history_s1["val_macro_f1"], label="Val Macro F1", color="#2ca02c")
    axes[0, 1].set_title("Stage 1: Healthy vs Pathological - Macro F1", fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Stage 2 Loss & Acc
    ep2 = range(1, len(history_s2["train_loss"]) + 1)
    axes[1, 0].plot(ep2, history_s2["train_loss"], label="Train Loss")
    axes[1, 0].plot(ep2, history_s2["val_loss"], label="Val Loss", linestyle="--")
    axes[1, 0].set_title("Stage 2: Laryngozele vs Vox Senilis - Loss", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ep2, history_s2["val_macro_f1"], label="Val Macro F1", color="#d62728")
    axes[1, 1].set_title("Stage 2: Laryngozele vs Vox Senilis - Macro F1", fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(BASE_DIR, "results", "figures", "hierarchical_training_curves.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Hierarchical Training] Curves saved to: {fig_path}")
    print("\n[Hierarchical Training] BOTH STAGES TRAINED SUCCESSFULLY!")
    print(f"  Stage 1 Best Checkpoint: {ckpt_s1}")
    print(f"  Stage 2 Best Checkpoint: {ckpt_s2}")


if __name__ == "__main__":
    main()
