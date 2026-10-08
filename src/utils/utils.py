"""
Utility Module
--------------
Provides helper functions for configuration loading, random seed setting
(for experimental reproducibility), and hardware device detection.
"""

import os
import random
import yaml
import numpy as np
import torch


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration YAML file into a dictionary."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across python, numpy, and torch for strict reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detect and return torch device (CUDA GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"[Device] Using CUDA GPU: {device_name}")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA not available. Using CPU.")
    return device
