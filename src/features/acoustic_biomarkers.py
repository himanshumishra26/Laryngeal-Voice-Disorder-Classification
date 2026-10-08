"""
Acoustic Biomarker Extraction Module
------------------------------------
Extracts clinically established phoniatric biomarkers from voice recordings:
- Fundamental frequency (F0) dynamics & pitch stability
- Frequency perturbation: Jitter (local, RAP, PPQ5)
- Amplitude perturbation: Shimmer (local, dB, APQ3, APQ5)
- Harmonicity: Harmonics-to-Noise Ratio (HNR), Noise-to-Harmonics Ratio (NHR)
- Spectral geometry: Wiener flatness, Centroid, Bandwidth, Rolloff, Alpha ratio (spectral tilt)
- Vocal tract resonance: Formants (F1, F2, F3) via Linear Predictive Coding (LPC)
- Glottal regularity: Cepstral Peak Prominence (CPP)
- Static spectral envelope: First 12 MFCC Means & Standard Deviations

Designed for low-dimensional, interpretable voice disorder classification.
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.signal
import librosa
from typing import Dict, Any, Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing.audio_preprocessor import AudioPreprocessor


class AcousticBiomarkerExtractor:
    """
    Extracts 52 clinical acoustic biomarkers from preprocessed audio signals.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        target_duration: float = 3.0,
        top_db: float = 20.0
    ):
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.top_db = top_db
        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            target_duration=target_duration,
            mono=True,
            top_db=top_db
        )

    def extract_from_file(self, file_path: str) -> Optional[Dict[str, float]]:
        """
        Loads, resamples, trims silence, peak-normalizes, and extracts all acoustic biomarkers.
        """
        try:
            raw_y, orig_sr = self.preprocessor.load_audio(file_path)
            y = self.preprocessor.resample(raw_y, orig_sr=orig_sr)
            y = self.preprocessor.trim_silence(y)

            if len(y) == 0:
                return None

            # Peak normalize
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val

            return self.extract_from_waveform(y, sr=self.sample_rate)
        except Exception as e:
            print(f"[AcousticBiomarkerExtractor] Error processing {file_path}: {e}", file=sys.stderr)
            return None

    def extract_from_waveform(self, y: np.ndarray, sr: int = 22050) -> Dict[str, float]:
        """
        Computes 52 clinical acoustic features from a 1D audio waveform.
        """
        feats: Dict[str, float] = {}
        feats["duration"] = float(len(y) / sr)

        # -------------------------------------------------------------
        # 1. Fundamental Frequency (F0) & Pitch Stability (pyin)
        # -------------------------------------------------------------
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=256
        )
        valid_f0 = f0[voiced_flag & ~np.isnan(f0)]

        if len(valid_f0) > 3:
            feats["f0_mean"] = float(np.mean(valid_f0))
            feats["f0_median"] = float(np.median(valid_f0))
            feats["f0_std"] = float(np.std(valid_f0))
            feats["f0_min"] = float(np.min(valid_f0))
            feats["f0_max"] = float(np.max(valid_f0))
            feats["f0_range_semitones"] = float(
                12 * np.log2((feats["f0_max"] + 1e-6) / (feats["f0_min"] + 1e-6))
            )
            feats["voiced_fraction"] = float(len(valid_f0) / max(1, len(f0)))

            # Pitch periods T_i (in seconds)
            T = 1.0 / valid_f0
            mean_T = float(np.mean(T))

            # Jitter Local (%)
            diff_T = np.abs(np.diff(T))
            feats["jitter_local_percent"] = float((np.mean(diff_T) / mean_T) * 100.0) if mean_T > 0 else 0.0

            # Jitter RAP (%)
            if len(T) > 3:
                rap_diff = np.abs(T[1:-1] - (T[:-2] + T[1:-1] + T[2:]) / 3.0)
                feats["jitter_rap_percent"] = float((np.mean(rap_diff) / mean_T) * 100.0)
            else:
                feats["jitter_rap_percent"] = feats["jitter_local_percent"]

            # Jitter PPQ5 (%)
            if len(T) > 5:
                ppq_diff = np.abs(T[2:-2] - np.convolve(T, np.ones(5) / 5.0, mode="valid"))
                feats["jitter_ppq5_percent"] = float((np.mean(ppq_diff) / mean_T) * 100.0)
            else:
                feats["jitter_ppq5_percent"] = feats["jitter_local_percent"]
        else:
            feats["f0_mean"] = 0.0
            feats["f0_median"] = 0.0
            feats["f0_std"] = 0.0
            feats["f0_min"] = 0.0
            feats["f0_max"] = 0.0
            feats["f0_range_semitones"] = 0.0
            feats["voiced_fraction"] = 0.0
            feats["jitter_local_percent"] = 0.0
            feats["jitter_rap_percent"] = 0.0
            feats["jitter_ppq5_percent"] = 0.0

        # -------------------------------------------------------------
        # 2. Amplitude Perturbation: Shimmer
        # -------------------------------------------------------------
        hop_amp = 256
        frame_amps = [np.max(np.abs(y[i : i + hop_amp])) for i in range(0, len(y) - hop_amp, hop_amp)]
        frame_amps_arr = np.array([a for a in frame_amps if a > 0.05])

        if len(frame_amps_arr) > 3:
            mean_amp = float(np.mean(frame_amps_arr))
            diff_amp = np.abs(np.diff(frame_amps_arr))
            feats["shimmer_local_percent"] = float((np.mean(diff_amp) / mean_amp) * 100.0) if mean_amp > 0 else 0.0

            ratio_amp = frame_amps_arr[1:] / (frame_amps_arr[:-1] + 1e-6)
            valid_ratios = ratio_amp[ratio_amp > 0]
            feats["shimmer_db"] = float(np.mean(np.abs(20 * np.log10(valid_ratios)))) if len(valid_ratios) > 0 else 0.0

            if len(frame_amps_arr) > 3:
                apq3_diff = np.abs(frame_amps_arr[1:-1] - (frame_amps_arr[:-2] + frame_amps_arr[1:-1] + frame_amps_arr[2:]) / 3.0)
                feats["shimmer_apq3_percent"] = float((np.mean(apq3_diff) / mean_amp) * 100.0)
            else:
                feats["shimmer_apq3_percent"] = feats["shimmer_local_percent"]

            if len(frame_amps_arr) > 5:
                apq5_diff = np.abs(frame_amps_arr[2:-2] - np.convolve(frame_amps_arr, np.ones(5) / 5.0, mode="valid"))
                feats["shimmer_apq5_percent"] = float((np.mean(apq5_diff) / mean_amp) * 100.0)
            else:
                feats["shimmer_apq5_percent"] = feats["shimmer_local_percent"]
        else:
            feats["shimmer_local_percent"] = 0.0
            feats["shimmer_db"] = 0.0
            feats["shimmer_apq3_percent"] = 0.0
            feats["shimmer_apq5_percent"] = 0.0

        # -------------------------------------------------------------
        # 3. Harmonics-to-Noise Ratio (HNR) via Autocorrelation
        # -------------------------------------------------------------
        if len(y) >= 2048:
            mid = len(y) // 2
            chunk = y[max(0, mid - 1024) : min(len(y), mid + 1024)]
            if len(chunk) == 2048:
                autocorr = np.correlate(chunk, chunk, mode="full")[2047:]
                if autocorr[0] > 0:
                    autocorr_norm = autocorr / autocorr[0]
                    min_lag = int(sr / 500)
                    max_lag = min(int(sr / 50), len(autocorr_norm) - 1)
                    if max_lag > min_lag:
                        peak_idx = min_lag + np.argmax(autocorr_norm[min_lag:max_lag])
                        r_peak = float(np.clip(autocorr_norm[peak_idx], 0.001, 0.999))
                        feats["hnr_db"] = float(10 * np.log10(r_peak / (1.0 - r_peak)))
                        feats["nhr"] = float((1.0 - r_peak) / r_peak)
                    else:
                        feats["hnr_db"] = 10.0
                        feats["nhr"] = 0.1
                else:
                    feats["hnr_db"] = 10.0
                    feats["nhr"] = 0.1
            else:
                feats["hnr_db"] = 10.0
                feats["nhr"] = 0.1
        else:
            feats["hnr_db"] = 10.0
            feats["nhr"] = 0.1

        # -------------------------------------------------------------
        # 4. Spectral Descriptors
        # -------------------------------------------------------------
        flatness = librosa.feature.spectral_flatness(y=y)
        feats["spectral_flatness_mean"] = float(np.mean(flatness))

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        feats["spectral_centroid_mean"] = float(np.mean(centroid))
        feats["spectral_centroid_std"] = float(np.std(centroid))

        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        feats["spectral_bandwidth_mean"] = float(np.mean(bandwidth))

        rolloff85 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        feats["spectral_rolloff85_mean"] = float(np.mean(rolloff85))

        rolloff95 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
        feats["spectral_rolloff95_mean"] = float(np.mean(rolloff95))

        # -------------------------------------------------------------
        # 5. Spectral Tilt (Alpha Ratio: [50-1000Hz] / [1000-5000Hz])
        # -------------------------------------------------------------
        stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        low_band = float(np.sum(stft[(freqs >= 50) & (freqs < 1000), :]))
        mid_band = float(np.sum(stft[(freqs >= 1000) & (freqs < 5000), :]))
        feats["alpha_ratio_db"] = float(10 * np.log10((low_band + 1e-8) / (mid_band + 1e-8)))

        # -------------------------------------------------------------
        # 6. Cepstral Peak Prominence (CPP)
        # -------------------------------------------------------------
        spec_mag = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        log_spec = np.log(spec_mag + 1e-6)
        cepstrum = np.real(np.fft.ifft(log_spec, axis=0))
        q_min = int(sr / 400)
        q_max = min(int(sr / 60), cepstrum.shape[0] - 1)
        if q_max > q_min:
            mean_cepstrum = np.mean(cepstrum, axis=1)
            q_range = mean_cepstrum[q_min:q_max]
            x = np.arange(q_min, q_max)
            poly = np.polyfit(x, q_range, 1)
            trend = np.polyval(poly, x)
            cpp = float(np.max(q_range - trend))
            feats["cpp"] = float(max(0.0, cpp))
        else:
            feats["cpp"] = 0.0

        # -------------------------------------------------------------
        # 7. Formants F1, F2, F3 via LPC
        # -------------------------------------------------------------
        try:
            y_filt = scipy.signal.lfilter([1, -0.97], [1], y)
            mid = len(y_filt) // 2
            win_len = min(2048, len(y_filt))
            window = y_filt[max(0, mid - win_len // 2) : min(len(y_filt), mid + win_len // 2)]
            window = window * np.hamming(len(window))
            order = min(24, max(4, 2 + sr // 1000))
            a = librosa.lpc(window, order=order)
            roots = np.roots(a)
            roots = roots[np.imag(roots) >= 0]
            angz = np.arctan2(np.imag(roots), np.real(roots))
            formants = sorted(angz * (sr / (2 * np.pi)))
            valid_formants = [f for f in formants if 200 < f < 4000]
            feats["formant_f1"] = float(valid_formants[0]) if len(valid_formants) > 0 else 500.0
            feats["formant_f2"] = float(valid_formants[1]) if len(valid_formants) > 1 else 1500.0
            feats["formant_f3"] = float(valid_formants[2]) if len(valid_formants) > 2 else 2500.0
            feats["formant_dispersion"] = float((feats["formant_f3"] - feats["formant_f1"]) / 2.0)
        except Exception:
            feats["formant_f1"] = 500.0
            feats["formant_f2"] = 1500.0
            feats["formant_f3"] = 2500.0
            feats["formant_dispersion"] = 1000.0

        # -------------------------------------------------------------
        # 8. First 12 MFCC Summary (Mean & Std)
        # -------------------------------------------------------------
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[1:]
        for idx in range(12):
            feats[f"mfcc{idx+1}_mean"] = float(np.mean(mfccs[idx]))
            feats[f"mfcc{idx+1}_std"] = float(np.std(mfccs[idx]))

        return feats


def _worker_extract(args):
    audio_path, row_meta = args
    extractor = AcousticBiomarkerExtractor()
    feats = extractor.extract_from_file(audio_path)
    if feats is None:
        return None
    res = dict(row_meta)
    res.update(feats)
    return res


def extract_tabular_dataset(
    manifest_csv: str,
    out_csv: str,
    base_dir: str = BASE_DIR,
    max_workers: int = 4
) -> pd.DataFrame:
    """
    Extracts acoustic biomarker table from a manifest CSV and saves it.
    """
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 1000:
        print(f"[AcousticBiomarkers] Loading existing tabular features from {out_csv}")
        return pd.read_csv(out_csv)

    print(f"[AcousticBiomarkers] Extracting acoustic biomarkers for {manifest_csv}...")
    df = pd.read_csv(manifest_csv)

    tasks = []
    for _, row in df.iterrows():
        audio_rel = row["audio_path"]
        audio_abs = os.path.join(base_dir, audio_rel) if not os.path.isabs(audio_rel) else audio_rel
        meta = {
            "audio_path": audio_rel,
            "class_name": row["class_name"],
            "class_label": int(row["class_label"]),
            "session_id": str(row["session_id"]),
            "speaker_id": str(row["speaker_id"]),
            "gender": str(row.get("gender", "unknown")),
            "age": float(row.get("age", -1.0)),
            "vowel_type": str(row.get("vowel_type", "")),
        }
        tasks.append((audio_abs, meta))

    results = []
    total = len(tasks)
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(_worker_extract, tasks):
            completed += 1
            if res is not None:
                results.append(res)
            if completed % 100 == 0 or completed == total:
                print(f"  Processed {completed}/{total} ({completed/total*100:.1f}%)")

    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    print(f"[AcousticBiomarkers] Successfully saved {len(res_df)} samples to {out_csv}")
    return res_df
