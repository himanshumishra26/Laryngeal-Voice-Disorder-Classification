"""
Dataset Module: PyTorch Dataset for SVD Laryngeal Voice Disorder Classification
-------------------------------------------------------------------------------
Loads precomputed 2D Log-Mel Spectrogram arrays (.npy) with corresponding class labels,
metadata, and optional data augmentation (SpecAugment) for CNN-LSTM training.
"""

import os
from typing import Optional, Callable, Dict, Any, Tuple, List
import pandas as pd
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset:
        pass


CLASS_MAP = {
    "normal": 0,
    "laryngozele": 1,
    "vox_senilis": 2
}

INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}


class SpecAugment:
    """
    Applies time and frequency masking to Log-Mel Spectrograms for acoustic regularization.
    """

    def __init__(
        self,
        freq_mask_max: int = 8,
        time_mask_max: int = 16,
        num_freq_masks: int = 1,
        num_time_masks: int = 1
    ):
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """
        Args:
            spec (np.ndarray): Array of shape (channels, n_mels, time_steps) or (n_mels, time_steps).
        Returns:
            augmented_spec (np.ndarray): Masked spectrogram.
        """
        spec = spec.copy()
        is_3d = (spec.ndim == 3)
        data = spec[0] if is_3d else spec
        n_mels, time_steps = data.shape

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = np.random.randint(0, self.freq_mask_max + 1)
            if f > 0 and f < n_mels:
                f0 = np.random.randint(0, n_mels - f)
                data[f0:f0 + f, :] = 0.0

        # Time masking
        for _ in range(self.num_time_masks):
            t = np.random.randint(0, self.time_mask_max + 1)
            if t > 0 and t < time_steps:
                t0 = np.random.randint(0, time_steps - t)
                data[:, t0:t0 + t] = 0.0

        if is_3d:
            spec[0] = data
            return spec
        return data


class SVDVoiceDataset(Dataset):
    """
    PyTorch Dataset for SVD Laryngeal Voice Disorder Classification.
    Reads manifest CSV files (train.csv, val.csv, test.csv) and loads precomputed
    Log-Mel Spectrograms of shape (1, n_mels, time_steps).
    """

    def __init__(
        self,
        manifest_csv: str,
        transform: Optional[Callable] = None,
        normalize: bool = False,
        base_dir: Optional[str] = None
    ):
        if not os.path.exists(manifest_csv):
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

        self.df = pd.read_csv(manifest_csv)
        self.transform = transform
        self.normalize = normalize
        self.base_dir = base_dir or ""

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        feat_path = row["feature_path"]
        if not os.path.isabs(feat_path) and self.base_dir:
            feat_path = os.path.join(self.base_dir, feat_path)

        spec = np.load(feat_path).astype(np.float32)  # shape: (n_mels, time_steps)

        if self.normalize:
            mean = np.mean(spec)
            std = np.std(spec) + 1e-6
            spec = (spec - mean) / std

        # Add channel dimension: (1, n_mels, time_steps)
        spec = np.expand_dims(spec, axis=0)

        if self.transform is not None:
            spec = self.transform(spec)

        label = int(row["class_label"])

        metadata = {
            "session_id": str(row["session_id"]),
            "speaker_id": str(row["speaker_id"]),
            "class_name": str(row["class_name"]),
            "sample_weight": float(row.get("sample_weight", 1.0)),
            "vowel_type": str(row.get("vowel_type", "")),
        }

        if TORCH_AVAILABLE:
            spec_tensor = torch.from_numpy(spec).float()
            label_tensor = torch.tensor(label, dtype=torch.long)
            return spec_tensor, label_tensor, metadata

        return spec, label, metadata

    def get_sample_weights(self) -> List[float]:
        """Returns the sample weights array for WeightedRandomSampler."""
        if "sample_weight" in self.df.columns:
            return self.df["sample_weight"].tolist()
        return [1.0] * len(self.df)


def get_class_weights(manifest_csv: str) -> np.ndarray:
    """
    Computes balanced class weights from a manifest CSV using inverse class frequency:
    w_c = N_total / (N_classes * N_c)
    """
    df = pd.read_csv(manifest_csv)
    class_counts = df["class_label"].value_counts().sort_index()
    n_samples = len(df)
    n_classes = len(CLASS_MAP)
    weights = np.zeros(n_classes, dtype=np.float32)
    for cls_idx in range(n_classes):
        cnt = class_counts.get(cls_idx, 0)
        weights[cls_idx] = n_samples / (n_classes * max(cnt, 1))
    return weights


def create_dataloaders(
    splits_dir: str = "data/splits",
    batch_size: int = 32,
    num_workers: int = 0,
    use_balanced_sampler: bool = True,
    use_balanced_manifests: bool = False,
    apply_spec_augment: bool = True
) -> Dict[str, Any]:
    """
    Factory function to create PyTorch DataLoaders for train, val, and test splits.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to build DataLoaders.")

    suffix = "_balanced.csv" if use_balanced_manifests else ".csv"
    train_csv = os.path.join(splits_dir, f"train{suffix}")
    val_csv = os.path.join(splits_dir, f"val{suffix}")
    test_csv = os.path.join(splits_dir, f"test{suffix}")

    train_transform = SpecAugment() if apply_spec_augment else None

    train_ds = SVDVoiceDataset(train_csv, transform=train_transform)
    val_ds = SVDVoiceDataset(val_csv)
    test_ds = SVDVoiceDataset(test_csv)

    if use_balanced_sampler and not use_balanced_manifests:
        weights = torch.DoubleTensor(train_ds.get_sample_weights())
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_weights": get_class_weights(train_csv)
    }
