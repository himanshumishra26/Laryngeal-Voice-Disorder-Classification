"""
GMM + DNN Calibrated Probability Blending Ensemble Classifier
--------------------------------------------------------------
Combines the class-conditional Gaussian Mixture Model (GMM) with the
Residual Deep Neural Network (ResNet-DNN) through calibrated probability
blending and decision threshold optimization for multi-class laryngeal
voice disorder classification (Normal, Laryngozele, Vox Senilis).

Mathematical Formulation:
    P_ens(c | x) = alpha * P_GMM(c | x) + (1 - alpha) * P_DNN(c | x)
    y_pred = argmax_c [ P_ens(c | x) + b_c ]

Methodological Reference:
    Frontiers in Digital Health (2026) - Voice disorders classification using ML:
    a scoping review (doi:10.3389/fdgth.2026.1800132).
"""

import os
import joblib
import numpy as np
import torch
from typing import Dict, Any, List, Optional
from src.models.gmm_classifier import GaussianMixtureVoiceClassifier
from src.models.dnn_classifier import DeepNeuralNetworkVoiceClassifier


class GMMDNNEnsembleClassifier:
    """
    Ensemble Classifier combining GMM and DNN predictions via calibrated probability fusion.
    """

    def __init__(
        self,
        gmm_model: GaussianMixtureVoiceClassifier,
        dnn_model: DeepNeuralNetworkVoiceClassifier,
        alpha: float = 0.25,
        class_bias: Optional[np.ndarray] = None
    ):
        """
        Parameters:
            gmm_model: Trained GaussianMixtureVoiceClassifier instance.
            dnn_model: Trained DeepNeuralNetworkVoiceClassifier instance.
            alpha: Blending weight for GMM probabilities (1 - alpha for DNN).
            class_bias: Decision threshold probability offset vector [b_0, b_1, b_2].
        """
        self.gmm_model = gmm_model
        self.dnn_model = dnn_model
        self.alpha = float(alpha)
        self.class_bias = np.asarray(class_bias, dtype=float) if class_bias is not None else np.zeros(3)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Computes convex combination of GMM and DNN posterior probabilities.
        """
        X = np.asarray(X, dtype=float)
        gmm_probs = self.gmm_model.predict_proba(X)
        dnn_probs = self.dnn_model.predict_proba(X)
        ens_probs = self.alpha * gmm_probs + (1.0 - self.alpha) * dnn_probs
        return ens_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts discrete class labels via argmax over calibrated ensemble probabilities.
        """
        probs = self.predict_proba(X)
        calibrated_probs = probs + self.class_bias
        return np.argmax(calibrated_probs, axis=1)

    def set_blend_weight(self, alpha: float) -> None:
        """
        Sets the probability blending weight for GMM.
        """
        self.alpha = float(alpha)

    def set_class_bias(self, bias: np.ndarray) -> None:
        """
        Sets the class probability bias vector for calibration.
        """
        self.class_bias = np.asarray(bias, dtype=float)

    def save(self, filepath: str) -> None:
        """
        Saves the ensemble model, including component models and fusion parameters.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "gmm_model": self.gmm_model,
            "dnn_checkpoint": {
                "in_features": self.dnn_model.in_features,
                "arch": self.dnn_model.arch,
                "hidden_dims": self.dnn_model.hidden_dims,
                "num_classes": self.dnn_model.num_classes,
                "dropout": self.dnn_model.dropout,
                "loss_type": self.dnn_model.loss_type,
                "k_best_features": self.dnn_model.k_best_features,
                "state_dict": self.dnn_model.model.state_dict(),
                "ensemble_state_dicts": [m.state_dict() for m in self.dnn_model.ensemble_models] if self.dnn_model.ensemble_models else [self.dnn_model.model.state_dict()],
                "scaler": self.dnn_model.scaler,
                "feature_selector": self.dnn_model.feature_selector,
                "class_bias": self.dnn_model.class_bias,
            },
            "alpha": self.alpha,
            "class_bias": self.class_bias
        }
        joblib.dump(checkpoint, filepath)
        print(f"[GMMDNNEnsemble] Saved ensemble checkpoint to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "GMMDNNEnsembleClassifier":
        """
        Loads the ensemble model from disk.
        """
        checkpoint = joblib.load(filepath)
        gmm_model = checkpoint["gmm_model"]

        dnn_dict = checkpoint["dnn_checkpoint"]
        dnn_model = DeepNeuralNetworkVoiceClassifier(
            in_features=dnn_dict["in_features"],
            arch=dnn_dict["arch"],
            hidden_dims=dnn_dict["hidden_dims"],
            num_classes=dnn_dict["num_classes"],
            dropout=dnn_dict["dropout"],
            loss_type=dnn_dict["loss_type"],
            k_best_features=dnn_dict["k_best_features"],
            device=device
        )
        dnn_model.scaler = dnn_dict["scaler"]
        dnn_model.feature_selector = dnn_dict.get("feature_selector", None)
        dnn_model.class_bias = dnn_dict.get("class_bias", None)

        if "ensemble_state_dicts" in dnn_dict and len(dnn_dict["ensemble_state_dicts"]) > 1:
            models = []
            for s_dict in dnn_dict["ensemble_state_dicts"]:
                m = dnn_model._init_model()
                m.load_state_dict(s_dict)
                models.append(m)
            dnn_model.ensemble_models = models
            dnn_model.model = models[0]
        else:
            dnn_model.model = dnn_model._init_model()
            dnn_model.model.load_state_dict(dnn_dict["state_dict"])
            dnn_model.ensemble_models = [dnn_model.model]

        ensemble = cls(
            gmm_model=gmm_model,
            dnn_model=dnn_model,
            alpha=checkpoint["alpha"],
            class_bias=checkpoint["class_bias"]
        )
        print(f"[GMMDNNEnsemble] Loaded ensemble checkpoint from {filepath}")
        return ensemble
