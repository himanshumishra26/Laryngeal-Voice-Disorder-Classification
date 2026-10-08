"""
Deep Learning Architecture: CNN-LSTM
------------------------------------
Hybrid architecture for automated multi-class classification of laryngeal voice disorders.
- 2D Convolutional layers extract local spectral and harmonic features from Log-Mel Spectrograms.
- Bidirectional LSTM layers model sequential phonatory temporal dynamics over time.
- Fully Connected Dense layer outputs predictions for 3 target classes:
    0: Normal
    1: Laryngozele
    2: Vox senilis
"""

import torch
import torch.nn as nn


class CNNLSTMVoiceClassifier(nn.Module):
    """
    CNN-LSTM Hybrid Classifier for Log-Mel Spectrograms.
    
    Architecture Overview:
    1. Input: (Batch, 1, n_mels=64, time_steps=130)
    2. Conv2D Block 1: Conv2d(1->32, 3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2) -> Dropout2d(0.25)
    3. Conv2D Block 2: Conv2d(32->64, 3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2) -> Dropout2d(0.25)
    4. Conv2D Block 3: Conv2d(64->128, 3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2) -> Dropout2d(0.30)
    5. Reshape / Permute: (Batch, time_steps=16, feature_dim=128*8=1024)
    6. Bidirectional LSTM: 2 layers, hidden_size=128, capturing bidirectional temporal dependencies
    7. Sequence Temporal Pooling: Mean pooling over sequence length -> (Batch, 256)
    8. Dense Classifier Head: Linear(256->64) -> BatchNorm1d -> ReLU -> Dropout(0.40) -> Linear(64->num_classes=3)
    """

    def __init__(
        self,
        num_classes: int = 3,
        n_mels: int = 64,
        conv_channels: tuple = (32, 64, 128),
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout_conv: float = 0.25,
        dropout_lstm: float = 0.30,
        dropout_dense: float = 0.40
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_mels = n_mels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional

        # Conv Block 1: (1, 64, 130) -> (32, 32, 65)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, conv_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=dropout_conv)
        )

        # Conv Block 2: (32, 32, 65) -> (64, 16, 32)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=dropout_conv)
        )

        # Conv Block 3: (64, 16, 32) -> (128, 8, 16)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=dropout_lstm)
        )

        # Dimension calculation after 3 max-pooling layers:
        # Mel freq: 64 -> 32 -> 16 -> 8
        # Feature dimension per time step = channels * freq_bins = 128 * 8 = 1024
        reduced_freq = n_mels // 8
        lstm_input_size = conv_channels[2] * reduced_freq

        # Bidirectional LSTM sequence modeling
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout_lstm if lstm_layers > 1 else 0.0
        )

        lstm_output_dim = lstm_hidden_size * (2 if bidirectional else 1)

        # Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_dense),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input Log-Mel Spectrogram of shape (Batch, 1, n_mels, time_steps)
        Returns:
            logits (torch.Tensor): Unnormalized class logits of shape (Batch, num_classes)
        """
        # 1. 2D CNN Feature Extraction
        x = self.conv_block1(x)  # (Batch, 32, 32, 65)
        x = self.conv_block2(x)  # (Batch, 64, 16, 32)
        x = self.conv_block3(x)  # (Batch, 128, 8, 16)

        # 2. Reshape for Sequential LSTM: (Batch, time_steps=16, feature_dim=1024)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # (Batch, T, C, F)
        x = x.view(B, T, C * F)                 # (Batch, T, C * F = 1024)

        # 3. Temporal Modeling with BiLSTM
        lstm_out, _ = self.lstm(x)              # (Batch, T, lstm_output_dim=256)

        # 4. Temporal Pooling (Mean pooling over sequence time steps)
        pooled = torch.mean(lstm_out, dim=1)    # (Batch, 256)

        # 5. Classification Logits
        logits = self.classifier(pooled)        # (Batch, num_classes)
        return logits
