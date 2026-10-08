"""
Audio Preprocessing Module
--------------------------
Handles loading, channel downmixing, resampling, silence removal,
amplitude normalization, and fixed-length padding/truncation for raw voice recordings.
"""

import os
from typing import Tuple, Optional
import numpy as np
import librosa
import nspfile


class AudioPreprocessor:
    """
    Standardizes raw voice recordings into uniform acoustic signals.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        target_duration: float = 3.0,
        mono: bool = True,
        top_db: float = 20.0
    ):
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.target_length = int(sample_rate * target_duration)
        self.mono = mono
        self.top_db = top_db

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Loads an audio file from disk using nspfile (for .nsp files) or librosa/soundfile.
        Converts PCM integer data to float32 in range [-1.0, 1.0].
        
        Returns:
            y (np.ndarray): 1D float32 audio waveform.
            sr (int): Original sample rate of the recording.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".nsp":
            sr, raw_data = nspfile.read(file_path)
            # Ensure 1D mono array
            if raw_data.ndim > 1:
                if self.mono:
                    if raw_data.shape[1] == 1:
                        raw_data = raw_data[:, 0]
                    else:
                        raw_data = np.mean(raw_data, axis=1)
                else:
                    raw_data = raw_data[:, 0]
            elif raw_data.ndim == 0:
                raise ValueError(f"Empty audio data in {file_path}")

            # Convert int16 to float32 in [-1.0, 1.0]
            if np.issubdtype(raw_data.dtype, np.integer):
                max_val = float(np.iinfo(raw_data.dtype).max)
                y = raw_data.astype(np.float32) / max_val
            else:
                y = raw_data.astype(np.float32)
        else:
            y, sr = librosa.load(file_path, sr=None, mono=self.mono)
            y = y.astype(np.float32)

        return y, sr

    def resample(self, y: np.ndarray, orig_sr: int) -> np.ndarray:
        """
        Resamples audio signal to the target sample rate.
        """
        if orig_sr == self.sample_rate:
            return y
        return librosa.resample(y, orig_sr=orig_sr, target_sr=self.sample_rate)

    def trim_silence(self, y: np.ndarray) -> np.ndarray:
        """
        Removes leading and trailing silence below top_db threshold.
        """
        if len(y) == 0:
            return y
        trimmed_y, _ = librosa.effects.trim(y, top_db=self.top_db)
        if len(trimmed_y) == 0:
            return y
        return trimmed_y

    def normalize_length(self, y: np.ndarray) -> np.ndarray:
        """
        Truncates or zero-pads audio signal to exact target length.
        """
        if len(y) > self.target_length:
            return y[:self.target_length]
        elif len(y) < self.target_length:
            pad_width = self.target_length - len(y)
            return np.pad(y, (0, pad_width), mode="constant")
        return y

    def process_file(
        self,
        file_path: str,
        normalize_len: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Full preprocessing pipeline on a single audio file:
        1. Load audio and normalize amplitude to [-1.0, 1.0]
        2. Resample from native sample rate (e.g. 50 kHz) to target sample rate (22,050 Hz)
        3. Remove leading and trailing silence
        4. Peak normalize
        5. Optionally pad/truncate to fixed target length
        
        Args:
            file_path (str): Path to audio file.
            normalize_len (bool): Whether to pad/truncate to fixed target_duration.
            
        Returns:
            processed_y (np.ndarray): Preprocessed float32 audio signal at target sample_rate.
            orig_duration (float): Original duration in seconds.
            proc_duration (float): Processed duration in seconds.
        """
        raw_y, orig_sr = self.load_audio(file_path)
        orig_duration = len(raw_y) / float(orig_sr)

        # Resample to target sample rate
        y = self.resample(raw_y, orig_sr=orig_sr)

        # Trim leading and trailing silence
        y = self.trim_silence(y)

        # Peak normalization to prevent clipping and balance volume
        max_abs = np.max(np.abs(y)) if len(y) > 0 else 0.0
        if max_abs > 0:
            y = y / max_abs

        # Optionally normalize length
        if normalize_len:
            y = self.normalize_length(y)

        proc_duration = len(y) / float(self.sample_rate)
        return y, orig_duration, proc_duration
