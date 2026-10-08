"""
Classical Machine Learning Models: Support Vector Classifier
------------------------------------------------------------
Implements an SVM with RBF kernel, balanced class weighting, and feature standardization
for multi-class classification of laryngeal voice disorders using acoustic biomarkers.
"""

from typing import List, Optional, Tuple, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# List of the 52 extracted acoustic biomarker feature columns
ACOUSTIC_FEATURE_COLS = [
    "duration",
    "f0_mean", "f0_median", "f0_std", "f0_min", "f0_max", "f0_range_semitones", "voiced_fraction",
    "jitter_local_percent", "jitter_rap_percent", "jitter_ppq5_percent",
    "shimmer_local_percent", "shimmer_db", "shimmer_apq3_percent", "shimmer_apq5_percent",
    "hnr_db", "nhr",
    "spectral_flatness_mean",
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_rolloff85_mean", "spectral_rolloff95_mean",
    "alpha_ratio_db",
    "cpp",
    "formant_f1", "formant_f2", "formant_f3", "formant_dispersion",
    "mfcc1_mean", "mfcc1_std",
    "mfcc2_mean", "mfcc2_std",
    "mfcc3_mean", "mfcc3_std",
    "mfcc4_mean", "mfcc4_std",
    "mfcc5_mean", "mfcc5_std",
    "mfcc6_mean", "mfcc6_std",
    "mfcc7_mean", "mfcc7_std",
    "mfcc8_mean", "mfcc8_std",
    "mfcc9_mean", "mfcc9_std",
    "mfcc10_mean", "mfcc10_std",
    "mfcc11_mean", "mfcc11_std",
    "mfcc12_mean", "mfcc12_std"
]


def create_svm_pipeline(
    C: float = 1.0,
    gamma: Any = "scale",
    class_weight: str = "balanced",
    random_state: int = 42
) -> Pipeline:
    """
    Constructs an end-to-end scikit-learn pipeline:
    1. StandardScaler: Centers and scales features (strictly fit on train partition)
    2. SVC: Support Vector Classifier with RBF kernel, cost-sensitive balanced weighting,
            and Platt scaling for calibrated posterior probabilities.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=C,
            kernel="rbf",
            gamma=gamma,
            class_weight=class_weight,
            probability=True,
            decision_function_shape="ovo",
            random_state=random_state
        ))
    ])
    return pipeline
