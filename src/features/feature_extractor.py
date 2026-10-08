"""
Feature Extraction Module
-------------------------
Extracts 2-Dimensional Log-Mel Spectrogram representations from preprocessed
audio signals, converting 1D waveforms into rich time-frequency acoustic images.
"""

from typing import Optional
import numpy as np
import librosa


class LogMelSpectrogramExtractor:
    """
    Computes Log-Mel Spectrogram representations for acoustic analysis.
    
    Parameters:
        sample_rate (int): Audio sampling rate (Hz). Default: 22050.
        n_fft (int): FFT window size. Default: 2048.
        hop_length (int): Number of audio samples between adjacent STFT columns. Default: 512.
        n_mels (int): Number of Mel frequency bands. Default: 64.
        f_min (float): Lowest frequency (Hz). Default: 0.0.
        f_max (float, optional): Highest frequency (Hz). Default: None (Nyquist).
        top_db (float): Threshold for logarithmic decibel dynamic range. Default: 80.0.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 64,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        top_db: float = 80.0
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.top_db = top_db

    def extract_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        Computes the Log-Mel Spectrogram from an audio time series array.
        Output shape: (n_mels, time_steps)
        
        Args:
            y (np.ndarray): 1D audio time series.
            
        Returns:
            log_mel_spec (np.ndarray): Log-Mel Spectrogram (dB) of shape (n_mels, time_steps).
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=2.0
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max, top_db=self.top_db)
        return log_mel_spec.astype(np.float32)

    def normalize(self, mel_spec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Applies standard z-score normalization to the spectrogram.
        
        Args:
            mel_spec (np.ndarray): 2D spectrogram array.
            eps (float): Numerical stability constant.
            
        Returns:
            norm_spec (np.ndarray): Spectrogram normalized to zero mean and unit variance.
        """
        mean = np.mean(mel_spec)
        std = np.std(mel_spec)
        return ((mel_spec - mean) / (std + eps)).astype(np.float32)
