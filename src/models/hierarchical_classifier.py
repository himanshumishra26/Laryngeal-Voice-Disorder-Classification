"""
Two-Stage Hierarchical Voice Classifier Architecture
---------------------------------------------------
Implements clinical two-stage hierarchical inference:
- Stage 1: Screening Model (Healthy vs. Pathological)
- Stage 2: Differential Diagnosis Model (Laryngozele vs. Vox Senilis)

Computes unified 3-class posterior probabilities via the Law of Total Probability:
  P(Normal)      = P(Healthy)
  P(Laryngozele) = P(Pathological) * P(Laryngozele | Pathological)
  P(Vox Senilis) = P(Pathological) * P(Vox Senilis | Pathological)
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.cnn_lstm import CNNLSTMVoiceClassifier


class HierarchicalVoiceClassifier(nn.Module):
    """
    Ensemble container combining Stage 1 and Stage 2 CNN-LSTM classifiers.
    """

    def __init__(
        self,
        n_mels: int = 64,
        stage1_model: Optional[nn.Module] = None,
        stage2_model: Optional[nn.Module] = None
    ):
        super().__init__()
        self.n_mels = n_mels
        self.stage1_model = stage1_model or CNNLSTMVoiceClassifier(num_classes=2, n_mels=n_mels)
        self.stage2_model = stage2_model or CNNLSTMVoiceClassifier(num_classes=2, n_mels=n_mels)

    @classmethod
    def load_from_checkpoints(
        cls,
        stage1_checkpoint_path: str,
        stage2_checkpoint_path: str,
        device: torch.device,
        n_mels: int = 64
    ) -> "HierarchicalVoiceClassifier":
        """
        Loads both trained Stage 1 and Stage 2 checkpoints into a unified classifier.
        """
        # Load Stage 1
        s1_ckpt = torch.load(stage1_checkpoint_path, map_location=device)
        model_s1 = CNNLSTMVoiceClassifier(num_classes=2, n_mels=n_mels)
        s1_state = s1_ckpt.get("model_state_dict", s1_ckpt)
        model_s1.load_state_dict(s1_state, strict=True)
        model_s1.to(device)
        model_s1.eval()

        # Load Stage 2
        s2_ckpt = torch.load(stage2_checkpoint_path, map_location=device)
        model_s2 = CNNLSTMVoiceClassifier(num_classes=2, n_mels=n_mels)
        s2_state = s2_ckpt.get("model_state_dict", s2_ckpt)
        model_s2.load_state_dict(s2_state, strict=True)
        model_s2.to(device)
        model_s2.eval()

        classifier = cls(n_mels=n_mels, stage1_model=model_s1, stage2_model=model_s2)
        classifier.to(device)
        classifier.eval()
        return classifier

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            probs_3class (torch.Tensor): Shape (Batch, 3) -> [P(Norm), P(Lar), P(Vox)]
            probs_stage1 (torch.Tensor): Shape (Batch, 2) -> [P(Healthy), P(Path)]
            probs_stage2 (torch.Tensor): Shape (Batch, 2) -> [P(Lar|Path), P(Vox|Path)]
        """
        # Stage 1: Healthy (0) vs Pathological (1)
        logits1 = self.stage1_model(x)
        probs1 = F.softmax(logits1, dim=1)
        p_healthy = probs1[:, 0:1]
        p_path = probs1[:, 1:2]

        # Stage 2: Laryngozele (0) vs Vox Senilis (1)
        logits2 = self.stage2_model(x)
        probs2 = F.softmax(logits2, dim=1)
        p_lar_given_path = probs2[:, 0:1]
        p_vox_given_path = probs2[:, 1:2]

        # Law of Total Probability
        p_normal = p_healthy
        p_laryngozele = p_path * p_lar_given_path
        p_vox_senilis = p_path * p_vox_given_path

        probs_3class = torch.cat([p_normal, p_laryngozele, p_vox_senilis], dim=1)
        return probs_3class, probs1, probs2
