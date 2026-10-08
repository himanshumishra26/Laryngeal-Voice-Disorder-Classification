"""
Gaussian Mixture Model (GMM) Classifier for Laryngeal Voice Disorder Classification
-----------------------------------------------------------------------------------
Implements class-conditional Gaussian Mixture Models with regularized covariance
matrices and calibrated class prior weighting for multi-class classification
(Normal, Laryngozele, Vox Senilis) using clinical acoustic biomarkers.

Methodological Reference:
    Frontiers in Digital Health (2026) - Voice disorders classification using ML:
    a scoping review (doi:10.3389/fdgth.2026.1800132).
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class GaussianMixtureVoiceClassifier:
    """
    Multi-Class Gaussian Mixture Model Classifier.
    Fits a separate Gaussian Mixture Model for each voice pathology class.
    """

    def __init__(
        self,
        n_components_per_class: Dict[int, int] = None,
        covariance_type: str = "diag",
        reg_covar: float = 0.05,
        class_prior_log_weights: Optional[Dict[int, float]] = None,
        likelihood_gates: Optional[Dict[int, float]] = None,
        k_best_features: Optional[int] = None,
        random_state: int = 42,
        max_iter: int = 200
    ):
        """
        Parameters:
            n_components_per_class: Dict mapping class label to number of Gaussian components.
                                   Defaults to {0: 4, 1: 1, 2: 4}.
            covariance_type: 'diag', 'full', 'tied', or 'spherical'.
            reg_covar: Non-negative regularization added to the diagonal of covariance.
            class_prior_log_weights: Optional log prior weight offsets for decision calibration.
            likelihood_gates: Optional dict mapping class label to minimum raw log-likelihood support gate.
            k_best_features: Optional number of top acoustic features to select via mutual information.
            random_state: Random seed for reproducibility.
            max_iter: Maximum number of EM iterations.
        """
        if n_components_per_class is None:
            n_components_per_class = {0: 4, 1: 1, 2: 4}
        self.n_components_per_class = n_components_per_class
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.class_prior_log_weights = class_prior_log_weights or {0: 0.0, 1: 0.0, 2: 0.0}
        self.likelihood_gates = likelihood_gates or {}
        self.k_best_features = k_best_features
        self.random_state = random_state
        self.max_iter = max_iter

        self.models: Dict[int, GaussianMixture] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_selector: Optional[Any] = None
        self.classes_: Optional[np.ndarray] = None
        self.num_classes_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, fit_scaler: bool = True) -> "GaussianMixtureVoiceClassifier":
        """
        Fits class-conditional Gaussian Mixture Models on the feature matrix.

        Parameters:
            X: Feature matrix of shape (N, D).
            y: Integer class labels of shape (N,).
            fit_scaler: Whether to fit a StandardScaler on X.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        self.num_classes_ = len(self.classes_)

        if fit_scaler:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        if self.k_best_features is not None:
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            self.feature_selector = SelectKBest(mutual_info_classif, k=self.k_best_features)
            X_proc = self.feature_selector.fit_transform(X_scaled, y)
        else:
            self.feature_selector = None
            X_proc = X_scaled

        self.models = {}
        for c in self.classes_:
            Xc = X_proc[y == c]
            n_comp = self.n_components_per_class.get(c, 2)
            n_comp = min(n_comp, len(Xc))
            
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
                max_iter=self.max_iter
            )
            gmm.fit(Xc)
            self.models[c] = gmm

        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if getattr(self, "feature_selector", None) is not None:
            X = self.feature_selector.transform(X)
        return X

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Computes calibrated class log-probabilities for each sample.
        """
        X_scaled = self._transform(X)
        n_samples = len(X_scaled)
        log_probs = np.zeros((n_samples, self.num_classes_))

        for idx, c in enumerate(sorted(self.classes_)):
            # score_samples returns per-sample log-likelihood: log P(x | c)
            log_lik = self.models[c].score_samples(X_scaled)
            # Add prior log weight offset
            prior_offset = self.class_prior_log_weights.get(c, 0.0)
            log_probs[:, idx] = log_lik + prior_offset

        return log_probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Computes normalized posterior probabilities via softmax over calibrated class log-probabilities,
        applying likelihood support gates if defined.
        """
        log_probs = self.predict_log_proba(X).copy()

        if self.likelihood_gates:
            X_scaled = self._transform(X)
            for c, gate_val in self.likelihood_gates.items():
                if c in self.models:
                    c_idx = np.where(self.classes_ == c)[0][0]
                    raw_ll = self.models[c].score_samples(X_scaled)
                    gated_out = raw_ll < gate_val
                    log_probs[gated_out, c_idx] = -1e9

        # Numerically stable softmax
        max_log = np.max(log_probs, axis=1, keepdims=True)
        exp_log = np.exp(log_probs - max_log)
        probs = exp_log / np.sum(exp_log, axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels via argmax over gated posterior probabilities.
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def set_class_prior_log_weights(self, weights: Dict[int, float]) -> None:
        """
        Updates the log-prior offsets for decision calibration.
        """
        self.class_prior_log_weights.update(weights)

    def set_likelihood_gate(self, class_label: int, min_log_likelihood: float) -> None:
        """
        Sets a minimum raw log-likelihood support threshold for the given class.
        """
        self.likelihood_gates[class_label] = float(min_log_likelihood)

    def save(self, filepath: str) -> None:
        """
        Serializes the model, GMM components, and scaler to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[GMMClassifier] Saved model checkpoint to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "GaussianMixtureVoiceClassifier":
        """
        Loads a saved model checkpoint from disk.
        """
        model = joblib.load(filepath)
        print(f"[GMMClassifier] Loaded model checkpoint from {filepath}")
        return model
