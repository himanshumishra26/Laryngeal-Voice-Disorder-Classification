"""
Training Pipeline Module
------------------------
Manages model optimization, backpropagation, validation monitoring,
learning rate scheduling, early stopping, and checkpoint serialization.
Supports label smoothing and validation macro-F1 checkpoint selection.
"""

import os
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score


class ModelTrainer:
    """
    Orchestrates training and validation passes for CNN-LSTM voice classification.
    Supports learning rate scheduling, early stopping, and model checkpointing.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config

        train_cfg = config.get("training", {})
        paths_cfg = config.get("paths", {})

        # Label smoothing
        label_smoothing = float(train_cfg.get("label_smoothing", 0.1))

        # Loss function with optional class weighting and label smoothing
        if class_weights is not None:
            weights_tensor = class_weights.to(device).float()
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.lr = train_cfg.get("learning_rate", 0.001)
        self.weight_decay = train_cfg.get("weight_decay", 0.0001)
        self.grad_clip = train_cfg.get("gradient_clip_norm", 5.0)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        scheduler_cfg = train_cfg.get("scheduler", {})
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=scheduler_cfg.get("mode", "min"),
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 5),
            min_lr=scheduler_cfg.get("min_lr", 0.00001)
        )

        # Early Stopping configuration
        early_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping_enabled = early_cfg.get("enabled", True)
        self.early_stopping_patience = early_cfg.get("patience", 10)
        self.early_stopping_min_delta = early_cfg.get("min_delta", 0.0001)
        self.monitor_metric = early_cfg.get("monitor", "val_macro_f1")
        self.monitor_mode = early_cfg.get("mode", "max" if "f1" in self.monitor_metric or "acc" in self.monitor_metric else "min")

        # Checkpointing configuration
        ckpt_cfg = train_cfg.get("checkpointing", {})
        self.checkpointing_enabled = ckpt_cfg.get("enabled", True)
        self.checkpoint_dir = paths_cfg.get("checkpoint_dir", "results/checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_path = os.path.join(
            self.checkpoint_dir,
            ckpt_cfg.get("best_model_filename", "best_cnn_lstm.pt")
        )
        self.last_model_path = os.path.join(
            self.checkpoint_dir,
            ckpt_cfg.get("last_model_filename", "last_cnn_lstm.pt")
        )

        if self.monitor_mode == "max":
            self.best_val_score = -float("inf")
        else:
            self.best_val_score = float("inf")
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """
        Runs one forward/backward training epoch.
        Returns:
            metrics (dict): {"loss": float, "accuracy": float}
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for batch_x, batch_y, _ in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            loss.backward()

            # Gradient clipping for LSTM stability
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

            self.optimizer.step()

            batch_size = batch_y.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = (correct / max(total_samples, 1)) * 100.0
        return {"loss": avg_loss, "accuracy": avg_acc}

    def validate(self, dataloader) -> Dict[str, float]:
        """
        Runs one evaluation pass over the validation dataset.
        Returns:
            metrics (dict): {"loss": float, "accuracy": float, "macro_f1": float}
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        all_true = []
        all_pred = []

        with torch.no_grad():
            for batch_x, batch_y, _ in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)

                batch_size = batch_y.size(0)
                total_loss += loss.item() * batch_size
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total_samples += batch_size

                all_true.extend(batch_y.cpu().numpy().tolist())
                all_pred.extend(preds.cpu().numpy().tolist())

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = (correct / max(total_samples, 1)) * 100.0
        macro_f1 = float(f1_score(all_true, all_pred, average="macro", zero_division=0)) * 100.0

        return {"loss": avg_loss, "accuracy": avg_acc, "macro_f1": macro_f1}

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        val_loss: float,
        val_acc: float,
        val_macro_f1: float,
        train_loss: float,
        train_acc: float
    ) -> None:
        """Serializes model weights and training metadata."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "config": self.config
        }
        torch.save(checkpoint, filepath)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 50,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop with learning rate scheduling, early stopping, and checkpointing.
        """
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_f1": [],
            "learning_rate": []
        }

        if verbose:
            print(f"[Training] Starting training on {self.device} for up to {epochs} epochs...")
            print(f"[Training] Monitoring: {self.monitor_metric} ({self.monitor_mode}) | Early stopping patience: {self.early_stopping_patience} epochs")

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step(val_metrics["loss"])

            current_lr = self.optimizer.param_groups[0]["lr"]
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])
            history["learning_rate"].append(current_lr)

            # Determine score for best model
            if self.monitor_metric == "val_macro_f1":
                current_score = val_metrics["macro_f1"]
            elif self.monitor_metric == "val_accuracy":
                current_score = val_metrics["accuracy"]
            else:
                current_score = val_metrics["loss"]

            if self.monitor_mode == "max":
                is_best = current_score > (self.best_val_score + self.early_stopping_min_delta)
            else:
                is_best = current_score < (self.best_val_score - self.early_stopping_min_delta)

            if is_best:
                self.best_val_score = current_score
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                if self.checkpointing_enabled:
                    self.save_checkpoint(
                        self.best_model_path,
                        epoch=epoch,
                        val_loss=val_metrics["loss"],
                        val_acc=val_metrics["accuracy"],
                        val_macro_f1=val_metrics["macro_f1"],
                        train_loss=train_metrics["loss"],
                        train_acc=train_metrics["accuracy"]
                    )
            else:
                self.patience_counter += 1

            # Save latest checkpoint
            if self.checkpointing_enabled:
                self.save_checkpoint(
                    self.last_model_path,
                    epoch=epoch,
                    val_loss=val_metrics["loss"],
                    val_acc=val_metrics["accuracy"],
                    val_macro_f1=val_metrics["macro_f1"],
                    train_loss=train_metrics["loss"],
                    train_acc=train_metrics["accuracy"]
                )

            if verbose:
                best_marker = " [*Best Checkpoint Saved*]" if is_best else ""
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] | "
                    f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}% | "
                    f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%, Val Macro-F1: {val_metrics['macro_f1']:.2f}% | "
                    f"LR: {current_lr:.6f}{best_marker}"
                )

            # Early stopping check
            if self.early_stopping_enabled and self.patience_counter >= self.early_stopping_patience:
                if verbose:
                    print(
                        f"\n[Early Stopping] Triggered at epoch {epoch}. "
                        f"{self.monitor_metric} did not improve for {self.early_stopping_patience} consecutive epochs. "
                        f"Best {self.monitor_metric}: {self.best_val_score:.4f}."
                    )
                break

        return history
