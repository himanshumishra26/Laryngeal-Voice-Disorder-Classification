"""
Training-Only Acoustic Augmentation for Minority Pathology Class
----------------------------------------------------------------
Applies synthetic pitch registration shifts and subtle acoustic perturbation
strictly to the training partition (Speaker 1602) to build a pitch-invariant
representation spanning human vocal registers from 95 Hz to 280 Hz.

STRICT INVARIANCE:
- Never accesses or modifies validation or test sets.
- Operates strictly on the training partition.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


def augment_training_features(
    df_train: pd.DataFrame,
    feat_cols: List[str],
    target_minority_class: int = 1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expands training minority class samples across diverse vocal pitch registers.

    Parameters:
        df_train: Training DataFrame containing authentic features and metadata.
        feat_cols: List of acoustic feature column names.
        target_minority_class: Class label to augment (1 for Laryngozele).
        random_state: Random seed for reproducibility.

    Returns:
        X_train_aug: Augmented training feature array.
        y_train_aug: Augmented training label array.
    """
    np.random.seed(random_state)

    X_raw = df_train[feat_cols].values
    y_raw = df_train["class_label"].values

    # Find pitch-related feature indices
    f0_cols = ["f0_mean", "f0_median", "f0_min", "f0_max"]
    f0_indices = [i for i, c in enumerate(feat_cols) if c in f0_cols]

    # Extract minority class samples
    minority_indices = np.where(y_raw == target_minority_class)[0]
    minority_samples = X_raw[minority_indices]

    # Semitone shifts from -4 to +14 (covers 95 Hz to 280 Hz)
    shifts = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    augmented_samples = []

    for s in shifts:
        factor = 2.0 ** (s / 12.0)
        for sample in minority_samples:
            new_sample = sample.copy()
            # Scale pitch features
            for idx in f0_indices:
                new_sample[idx] = new_sample[idx] * factor
            # Add subtle acoustic variance (2% gaussian perturbation)
            noise = np.random.normal(0, 0.02, size=len(new_sample))
            new_sample = new_sample * (1.0 + noise)
            augmented_samples.append(new_sample)

    augmented_samples = np.array(augmented_samples)

    # Combine with authentic majority classes
    X_maj = X_raw[y_raw != target_minority_class]
    y_maj = y_raw[y_raw != target_minority_class]

    X_train_aug = np.vstack([X_maj, augmented_samples])
    y_train_aug = np.concatenate([
        y_maj,
        np.full(len(augmented_samples), target_minority_class, dtype=int)
    ])

    # Shuffle training set
    shuffle_idx = np.random.permutation(len(y_train_aug))
    X_train_aug = X_train_aug[shuffle_idx]
    y_train_aug = y_train_aug[shuffle_idx]

    return X_train_aug, y_train_aug
