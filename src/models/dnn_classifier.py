"""
Deep Neural Network (DNN) Classifier for Laryngeal Voice Disorder Classification
--------------------------------------------------------------------------------
Implements both Multi-Layer Perceptron (MLP) and Residual Multi-Layer Perceptron (ResNet-MLP)
with Batch Normalization, GELU/LeakyReLU activations, Dropout regularization,
Focal Loss / Cost-Sensitive Cross Entropy, and Cosine Annealing learning rate scheduling
for multi-class classification (Normal, Laryngozele, Vox Senilis) using acoustic biomarkers.

Methodological Reference:
    Frontiers in Digital Health (2026) - Voice disorders classification using ML:
    a scoping review (doi:10.3389/fdgth.2026.1800132).
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional, Tuple


class VoiceDNNModule(nn.Module):
    """
    Standard Feedforward Multi-Layer Perceptron (MLP) for acoustic biomarkers.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int] = None,
        num_classes: int = 3,
        dropout: float = 0.35
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResBlock(nn.Module):
    """
    Residual dense block with Batch Normalization, GELU activations, and Dropout.
    """
    def __init__(self, dim: int, dropout: float = 0.25):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop(self.act(self.bn1(self.fc1(x))))
        out = self.bn2(self.fc2(out))
        return self.act(out + residual)


class VoiceResNetModule(nn.Module):
    """
    Residual Multi-Layer Perceptron (ResNet-MLP) for acoustic biomarkers.
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        num_classes: int = 3,
        dropout: float = 0.25
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss with class-frequency weighting:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class DeepNeuralNetworkVoiceClassifier:
    """
    Scikit-learn compatible wrapper for the VoiceDNNModule / VoiceResNetModule.
    Includes feature scaling, feature selection, training loop, validation tracking,
    checkpointing, multi-seed ensembling, and inference.
    """

    def __init__(
        self,
        in_features: int = 53,
        arch: str = "resnet",
        hidden_dims: List[int] = None,
        num_classes: int = 3,
        dropout: float = 0.25,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        loss_type: str = "focal",
        focal_gamma: float = 1.5,
        scheduler_type: str = "cosine",
        k_best_features: Optional[int] = None,
        batch_size: int = 64,
        num_epochs: int = 50,
        random_state: int = 42,
        device: Optional[str] = None
    ):
        self.in_features = in_features
        self.arch = arch
        self.hidden_dims = hidden_dims or ([128] if arch == "resnet" else [128, 64, 32])
        self.num_classes = num_classes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.scheduler_type = scheduler_type
        self.k_best_features = k_best_features
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.random_state = random_state

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.scaler = StandardScaler()
        self.feature_selector = None
        self.model: Optional[nn.Module] = None
        self.ensemble_models: Optional[List[nn.Module]] = None
        self.class_bias: Optional[np.ndarray] = None
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_macro_f1": []
        }

    def _init_model(self, seed: Optional[int] = None) -> nn.Module:
        s = seed if seed is not None else self.random_state
        torch.manual_seed(s)
        np.random.seed(s)

        effective_in = self.k_best_features if self.k_best_features is not None else self.in_features

        if self.arch == "resnet":
            hidden_dim = self.hidden_dims[0] if isinstance(self.hidden_dims, list) else self.hidden_dims
            return VoiceResNetModule(
                in_features=effective_in,
                hidden_dim=hidden_dim,
                num_blocks=2,
                num_classes=self.num_classes,
                dropout=self.dropout
            ).to(self.device)
        else:
            return VoiceDNNModule(
                in_features=effective_in,
                hidden_dims=self.hidden_dims,
                num_classes=self.num_classes,
                dropout=self.dropout
            ).to(self.device)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if getattr(self, "feature_selector", None) is not None:
            X = self.feature_selector.transform(X)
        return X

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        seeds: Optional[List[int]] = None,
        verbose: bool = True
    ) -> "DeepNeuralNetworkVoiceClassifier":
        """
        Trains the DNN on X_train, y_train with optional validation monitoring.
        If seeds list is provided (e.g. [42, 123, 777]), trains a multi-seed ensemble.
        """
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=int)
        self.in_features = X_train.shape[1]

        # Scaler
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Feature selection
        if self.k_best_features is not None:
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            self.feature_selector = SelectKBest(mutual_info_classif, k=self.k_best_features)
            X_train_proc = self.feature_selector.fit_transform(X_train_scaled, y_train)
        else:
            self.feature_selector = None
            X_train_proc = X_train_scaled

        if X_val is not None and y_val is not None:
            X_val_proc = self._transform(X_val)
            X_val_t = torch.tensor(X_val_proc, dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_val, dtype=torch.long).to(self.device)
        else:
            X_val_t, y_val_t = None, None

        # Class weights
        counts = np.bincount(y_train, minlength=self.num_classes)
        weights = len(y_train) / (self.num_classes * np.maximum(counts, 1).astype(float))
        weights_t = torch.tensor(weights, dtype=torch.float32).to(self.device)

        seeds_to_run = seeds if seeds is not None else [self.random_state]
        trained_models = []

        for s in seeds_to_run:
            m = self._init_model(seed=s)
            if self.loss_type == "focal":
                criterion = FocalLoss(alpha=weights_t, gamma=self.focal_gamma)
            else:
                criterion = nn.CrossEntropyLoss(weight=weights_t)

            optimizer = optim.AdamW(m.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            if self.scheduler_type == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-5)
            else:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)

            train_ds = TensorDataset(
                torch.tensor(X_train_proc, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            )
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

            best_val_score = -1.0
            best_state = None

            for epoch in range(1, self.num_epochs + 1):
                m.train()
                running_loss = 0.0
                for bx, by in train_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    out = m(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * len(by)

                epoch_train_loss = running_loss / len(train_ds)

                if X_val_t is not None:
                    m.eval()
                    with torch.no_grad():
                        val_logits = m(X_val_t)
                        val_loss = criterion(val_logits, y_val_t).item()
                        val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
                        val_preds = np.argmax(val_probs, axis=1)

                        from sklearn.metrics import precision_recall_fscore_support
                        _, r, _, _ = precision_recall_fscore_support(y_val, val_preds, labels=list(range(self.num_classes)), zero_division=0)
                        val_bal_acc = float(np.mean(r))

                    if self.scheduler_type == "cosine":
                        scheduler.step()
                    else:
                        scheduler.step(val_loss)

                    if val_bal_acc > best_val_score:
                        best_val_score = val_bal_acc
                        best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}

            if best_state is not None:
                m.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

            trained_models.append(m)

        self.ensemble_models = trained_models
        self.model = trained_models[0]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Computes softmax posterior probabilities on input features, averaging across ensemble seeds if present.
        """
        X_proc = self._transform(X)
        X_t = torch.tensor(X_proc, dtype=torch.float32).to(self.device)

        models = self.ensemble_models if getattr(self, "ensemble_models", None) is not None else [self.model]
        probs_all = []

        for m in models:
            m.eval()
            with torch.no_grad():
                logits = m(X_t)
                p = torch.softmax(logits, dim=1).cpu().numpy()
                probs_all.append(p)

        return np.mean(probs_all, axis=0)

    def predict(self, X: np.ndarray, class_bias: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predicts class labels via argmax over calibrated class probabilities.
        """
        probs = self.predict_proba(X)
        if class_bias is not None:
            probs = probs + class_bias
        elif self.class_bias is not None:
            probs = probs + self.class_bias
        return np.argmax(probs, axis=1)

    def set_class_bias(self, bias: np.ndarray) -> None:
        self.class_bias = np.asarray(bias, dtype=float)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        ensemble_states = [m.state_dict() for m in self.ensemble_models] if self.ensemble_models else [self.model.state_dict()]
        checkpoint = {
            "in_features": self.in_features,
            "arch": self.arch,
            "hidden_dims": self.hidden_dims,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "loss_type": self.loss_type,
            "k_best_features": self.k_best_features,
            "state_dict": self.model.state_dict(),
            "ensemble_state_dicts": ensemble_states,
            "scaler": self.scaler,
            "feature_selector": self.feature_selector,
            "class_bias": self.class_bias,
            "training_history": self.training_history
        }
        torch.save(checkpoint, filepath)
        print(f"[DNNClassifier] Saved model checkpoint to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "DeepNeuralNetworkVoiceClassifier":
        checkpoint = torch.load(filepath, map_location=device or "cpu", weights_only=False)
        model = cls(
            in_features=checkpoint.get("in_features", 53),
            arch=checkpoint.get("arch", "mlp"),
            hidden_dims=checkpoint.get("hidden_dims", [128, 64, 32]),
            num_classes=checkpoint.get("num_classes", 3),
            dropout=checkpoint.get("dropout", 0.35),
            loss_type=checkpoint.get("loss_type", "ce"),
            k_best_features=checkpoint.get("k_best_features", None),
            device=device
        )
        model.scaler = checkpoint["scaler"]
        model.feature_selector = checkpoint.get("feature_selector", None)
        model.class_bias = checkpoint.get("class_bias", None)
        model.training_history = checkpoint.get("training_history", {})

        # Load models
        if "ensemble_state_dicts" in checkpoint and len(checkpoint["ensemble_state_dicts"]) > 1:
            models = []
            for s_dict in checkpoint["ensemble_state_dicts"]:
                m = model._init_model()
                m.load_state_dict(s_dict)
                models.append(m)
            model.ensemble_models = models
            model.model = models[0]
        else:
            model.model = model._init_model()
            model.model.load_state_dict(checkpoint["state_dict"])
            model.ensemble_models = [model.model]

        print(f"[DNNClassifier] Loaded model checkpoint from {filepath}")
        return model
