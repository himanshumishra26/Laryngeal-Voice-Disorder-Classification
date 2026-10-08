"""
Calibrated Multi-Seed Acoustic Biomarker Ensemble Model
-------------------------------------------------------
Ensemble architecture combining regularized linear-margin models and
calibrated support vector classifiers operating on 52 clinical acoustic biomarkers.

Key Design Elements:
1. Training-only Minority Augmentation: Expands the single training patient's
   authentic samples into 350 pitch- and resonance-perturbed synthetic variants per seed.
2. Multi-Model Complementarity:
   - Model A (45%): L2-Regularized Logistic Regression (C=0.15, balanced)
   - Model B (40%): Calibrated Linear SVC (C=0.10, balanced, 3-fold Platt scaling)
   - Model C (15%): Calibrated RBF SVC (C=1.00, gamma=0.01, balanced, 3-fold Platt scaling)
3. Multi-Seed Bagging: Fits models across 5 distinct random seeds (42, 101, 2024, 777, 999)
   and averages posterior probabilities to eliminate stochastic sampling noise.
4. Frozen Decision Threshold: Applies a calibrated detection threshold for the minority
   pathology optimized exclusively on validation data.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


class CalibratedAcousticEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier over 52 acoustic biomarkers combining multiple calibrated
    linear-margin models across multiple random seeds.
    """

    def __init__(
        self,
        seeds: Optional[List[int]] = None,
        target_minority_samples: int = 350,
        lr_c: float = 0.15,
        svc_lin_c: float = 0.10,
        svc_rbf_c: float = 1.00,
        svc_rbf_gamma: float = 0.01,
        model_weights: Optional[Tuple[float, float, float]] = None,
        frozen_threshold: float = 0.12
    ):
        self.seeds = seeds if seeds is not None else [42, 101, 2024, 777, 999]
        self.target_minority_samples = target_minority_samples
        self.lr_c = lr_c
        self.svc_lin_c = svc_lin_c
        self.svc_rbf_c = svc_rbf_c
        self.svc_rbf_gamma = svc_rbf_gamma
        self.model_weights = model_weights if model_weights is not None else (0.45, 0.40, 0.15)
        self.frozen_threshold = frozen_threshold
        
        # Internal state
        self.scaler: Optional[StandardScaler] = None
        self.fitted_models_: List[Dict[str, Any]] = []
        self.classes_ = np.array([0, 1, 2])

    def _augment_minority_training(
        self,
        X_lar: np.ndarray,
        seed: int
    ) -> np.ndarray:
        """
        Generates synthetic training samples for Laryngozele in standardized feature space.
        Strictly executed only on the training partition.
        """
        np.random.seed(seed)
        num_syn = max(0, self.target_minority_samples - len(X_lar))
        if num_syn == 0:
            return X_lar.copy()

        syn_samples = []
        for _ in range(num_syn):
            # Pick two distinct samples from X_lar
            idx1, idx2 = np.random.choice(len(X_lar), size=2, replace=True)
            lam = np.random.uniform(0.1, 0.9)
            # Convex combination
            syn = lam * X_lar[idx1] + (1 - lam) * X_lar[idx2]
            # Add subtle physiological acoustic perturbation (sigma = 0.05)
            jitter = np.random.normal(0, 0.05, size=syn.shape)
            syn_samples.append(syn + jitter)

        syn_arr = np.array(syn_samples, dtype=np.float32)
        return np.vstack([X_lar, syn_arr])

    def fit(self, X: np.ndarray, y: np.ndarray, scaler: Optional[StandardScaler] = None):
        """
        Fits the multi-seed calibrated ensemble on the training set.
        
        Args:
            X (np.ndarray): Unscaled raw training features of shape (N, 52).
            y (np.ndarray): Training integer labels of shape (N,).
            scaler (StandardScaler, optional): Pre-fitted scaler on training data.
        """
        if scaler is not None:
            self.scaler = scaler
            X_sc = self.scaler.transform(X)
        else:
            self.scaler = StandardScaler()
            X_sc = self.scaler.fit_transform(X)

        X_lar = X_sc[y == 1]
        X_norm = X_sc[y == 0]
        X_vox = X_sc[y == 2]

        self.fitted_models_ = []
        w_lr, w_svc_lin, w_svc_rbf = self.model_weights

        for seed in self.seeds:
            # 1. Training-only minority class augmentation
            X_lar_aug = self._augment_minority_training(X_lar, seed)
            
            # Form balanced training matrix for this seed
            X_aug = np.vstack([X_norm, X_lar_aug, X_vox])
            y_aug = np.concatenate([
                np.zeros(len(X_norm), dtype=int),
                np.ones(len(X_lar_aug), dtype=int),
                np.full(len(X_vox), 2, dtype=int)
            ])

            # 2. Fit Model A: Regularized Logistic Regression
            lr = LogisticRegression(
                C=self.lr_c,
                class_weight="balanced",
                max_iter=1500,
                random_state=seed
            )
            lr.fit(X_aug, y_aug)

            # 3. Fit Model B: Calibrated Linear SVC
            base_svc_lin = SVC(
                C=self.svc_lin_c,
                kernel="linear",
                class_weight="balanced",
                random_state=seed
            )
            svc_lin = CalibratedClassifierCV(base_svc_lin, cv=3)
            svc_lin.fit(X_aug, y_aug)

            # 4. Fit Model C: Calibrated RBF SVC
            base_svc_rbf = SVC(
                C=self.svc_rbf_c,
                kernel="rbf",
                gamma=self.svc_rbf_gamma,
                class_weight="balanced",
                random_state=seed
            )
            svc_rbf = CalibratedClassifierCV(base_svc_rbf, cv=3)
            svc_rbf.fit(X_aug, y_aug)

            self.fitted_models_.append({
                "seed": seed,
                "lr": lr,
                "svc_lin": svc_lin,
                "svc_rbf": svc_rbf,
                "weights": (w_lr, w_svc_lin, w_svc_rbf)
            })

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Computes ensemble calibrated posterior probabilities across all models and seeds.
        
        Args:
            X (np.ndarray): Raw feature array of shape (N, 52).
            
        Returns:
            probs (np.ndarray): Posterior probability array of shape (N, 3).
        """
        if self.scaler is None or len(self.fitted_models_) == 0:
            raise RuntimeError("Model must be fitted before predict_proba can be called.")

        X_sc = self.scaler.transform(X)
        all_seed_probs = []

        for model_dict in self.fitted_models_:
            lr = model_dict["lr"]
            svc_lin = model_dict["svc_lin"]
            svc_rbf = model_dict["svc_rbf"]
            w_lr, w_svc_lin, w_svc_rbf = model_dict["weights"]

            p_lr = lr.predict_proba(X_sc)
            p_svc_lin = svc_lin.predict_proba(X_sc)
            p_svc_rbf = svc_rbf.predict_proba(X_sc)

            p_seed = w_lr * p_lr + w_svc_lin * p_svc_lin + w_svc_rbf * p_svc_rbf
            all_seed_probs.append(p_seed)

        # Average probabilities across all random seeds
        avg_probs = np.mean(all_seed_probs, axis=0)
        
        # Ensure exact row normalization
        row_sums = avg_probs.sum(axis=1, keepdims=True)
        avg_probs = avg_probs / np.clip(row_sums, 1e-12, None)
        return avg_probs

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Applies the frozen clinical decision policy.
        
        Decision Policy:
        - If P(Laryngozele) >= threshold AND P(Laryngozele) > P(Normal): predict Laryngozele (1)
        - Else: predict argmax between Normal (0) and Vox Senilis (2).
        
        Args:
            X (np.ndarray): Raw feature array of shape (N, 52).
            threshold (float, optional): Custom threshold; defaults to frozen_threshold.
            
        Returns:
            preds (np.ndarray): Integer predictions of shape (N,).
        """
        th = threshold if threshold is not None else self.frozen_threshold
        probs = self.predict_proba(X)
        
        preds = []
        for p in probs:
            p_norm, p_lar, p_vox = p[0], p[1], p[2]
            if p_lar >= th and p_lar > p_norm:
                preds.append(1)
            else:
                preds.append(0 if p_norm >= p_vox else 2)
                
        return np.array(preds, dtype=int)
