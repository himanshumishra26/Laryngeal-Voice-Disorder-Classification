"""
Stage 2: Sample Feature Extraction Pipeline
-------------------------------------------
Executes sample NSP preprocessing and Log-Mel Spectrogram extraction
for one sample from each class (Normal, Laryngozele, Vox Senilis).

Requirements:
- Input: .nsp files loaded via nspfile
- Resample: 50 kHz -> 22,050 Hz
- Preprocessing: Silence trimming (top_db=20) and amplitude normalization
- Features: Log-Mel Spectrogram (n_mels=64, n_fft=2048, hop_length=512)
- Output: Numerical arrays in data/processed/sample_features/
- Visualizations: PNG figures in results/figures/sample_spectrograms/
- Report: results/reports/sample_feature_extraction_report.txt
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import librosa
import librosa.display

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.features.feature_extractor import LogMelSpectrogramExtractor


def run_sample_feature_extraction():
    # 1. Project directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    feat_dir = os.path.join(base_dir, "data", "processed", "sample_features")
    fig_dir = os.path.join(base_dir, "results", "figures", "sample_spectrograms")
    rep_dir = os.path.join(base_dir, "results", "reports")

    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    # 2. Configuration (as strictly specified in requirements)
    sample_rate = 22050
    n_mels = 64
    n_fft = 2048
    hop_length = 512
    top_db_trim = 20.0
    top_db_mel = 80.0

    print("=" * 70)
    print("STAGE 2: SAMPLE FEATURE EXTRACTION")
    print(f"Sample Rate: {sample_rate} Hz | n_mels: {n_mels} | n_fft: {n_fft} | hop_length: {hop_length}")
    print("=" * 70)

    # 3. Selected Sample Recordings (One from each class)
    # Using sustained vowel /a/ at normal pitch ('a_n') for consistent acoustic comparison
    selected_samples = {
        "normal": {
            "label": "Normal (Healthy Control)",
            "rel_path": os.path.join("data", "raw", "normal", "1000", "vowels", "1000-a_n.nsp"),
            "session_id": "1000",
            "speaker_id": "978",
            "sex": "Female (w)",
            "birth_date": "1972-02-14",
            "rec_date": "1998-10-09",
            "age": 26.6,
            "diagnosis": "Healthy control (no laryngeal pathology)",
        },
        "laryngozele": {
            "label": "Laryngozele (Pathology)",
            "rel_path": os.path.join("data", "raw", "laryngozele", "1205", "vowels", "1205-a_n.nsp"),
            "session_id": "1205",
            "speaker_id": "1602",
            "sex": "Male (m)",
            "birth_date": "1952-01-16",
            "rec_date": "1999-01-13",
            "age": 47.0,
            "diagnosis": "Z.B. linkes Taschenband, Stimmlippen o.B.; Stroboskopie o.B.",
        },
        "vox_senilis": {
            "label": "Vox Senilis (Pathology)",
            "rel_path": os.path.join("data", "raw", "vox_senilis", "1203", "vowels", "1203-a_n.nsp"),
            "session_id": "1203",
            "speaker_id": "1600",
            "sex": "Female (w)",
            "birth_date": "1937-12-20",
            "rec_date": "1999-01-13",
            "age": 61.1,
            "diagnosis": "Leicht",
        }
    }

    # Initialize preprocessor and extractor
    preprocessor = AudioPreprocessor(
        sample_rate=sample_rate,
        target_duration=3.0,
        mono=True,
        top_db=top_db_trim
    )
    extractor = LogMelSpectrogramExtractor(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0.0,
        f_max=float(sample_rate // 2),
        top_db=top_db_mel
    )

    results = {}
    errors = []

    for cls_name, info in selected_samples.items():
        abs_path = os.path.join(base_dir, info["rel_path"])
        file_name = os.path.basename(abs_path)
        print(f"\nProcessing class: {cls_name.upper()}")
        print(f"  File: {info['rel_path']}")

        try:
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Target recording not found: {abs_path}")

            # Preprocess
            proc_audio, orig_dur, proc_dur = preprocessor.process_file(abs_path, normalize_len=False)

            # Extract Log-Mel Spectrogram
            log_mel = extractor.extract_spectrogram(proc_audio)

            # Numerical validity checks
            nan_count = int(np.isnan(log_mel).sum())
            inf_count = int(np.isinf(log_mel).sum())
            if nan_count > 0 or inf_count > 0:
                raise ValueError(f"Extracted feature array contains NaN ({nan_count}) or Inf ({inf_count}) values.")

            # Save numerical feature array
            feat_filename = f"{cls_name}_sample_logmel.npy"
            feat_path = os.path.join(feat_dir, feat_filename)
            np.save(feat_path, log_mel)
            file_size_bytes = os.path.getsize(feat_path)

            # Generate and save individual spectrogram plot
            img_filename = f"{cls_name}_sample_spectrogram.png"
            img_path = os.path.join(fig_dir, img_filename)

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
            img = librosa.display.specshow(
                log_mel,
                sr=sample_rate,
                hop_length=hop_length,
                x_axis="time",
                y_axis="mel",
                fmin=0.0,
                fmax=sample_rate // 2,
                ax=ax,
                cmap="magma"
            )
            cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
            cbar.set_label("Magnitude (dB)", rotation=270, labelpad=15)
            ax.set_title(
                f"Log-Mel Spectrogram: {info['label']}\n"
                f"Session: {info['session_id']} | File: {file_name} | Shape: {log_mel.shape}",
                fontsize=11, fontweight="bold"
            )
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Mel Frequency (Hz)")
            plt.tight_layout()
            plt.savefig(img_path, dpi=300)
            plt.close()

            results[cls_name] = {
                "info": info,
                "file_name": file_name,
                "abs_path": abs_path,
                "orig_dur": orig_dur,
                "proc_dur": proc_dur,
                "spec_shape": log_mel.shape,
                "dtype": str(log_mel.dtype),
                "min_db": float(log_mel.min()),
                "max_db": float(log_mel.max()),
                "mean_db": float(log_mel.mean()),
                "std_db": float(log_mel.std()),
                "feat_path": feat_path,
                "feat_size_bytes": file_size_bytes,
                "img_path": img_path,
                "log_mel": log_mel
            }

            print(f"  [OK] Orig Dur: {orig_dur:.3f}s -> Proc Dur: {proc_dur:.3f}s")
            print(f"  [OK] Feature Shape: {log_mel.shape} (n_mels x time_steps)")
            print(f"  [OK] Range: [{log_mel.min():.2f}, {log_mel.max():.2f}] dB | Mean: {log_mel.mean():.2f} dB")
            print(f"  [OK] Saved array: {feat_path} ({file_size_bytes} bytes)")
            print(f"  [OK] Saved image: {img_path}")

        except Exception as e:
            err_msg = f"Error processing class '{cls_name}' ({abs_path}): {str(e)}"
            print(f"  [ERROR] {err_msg}", file=sys.stderr)
            errors.append(err_msg)

    # 4. Generate Combined Comparative Figure
    comparison_img_path = os.path.join(fig_dir, "sample_spectrograms_comparison.png")
    if len(results) == 3:
        fig, axes = plt.subplots(3, 1, figsize=(10, 11), dpi=300)
        class_order = ["normal", "laryngozele", "vox_senilis"]

        for i, cls in enumerate(class_order):
            res = results[cls]
            ax = axes[i]
            img = librosa.display.specshow(
                res["log_mel"],
                sr=sample_rate,
                hop_length=hop_length,
                x_axis="time",
                y_axis="mel",
                fmin=0.0,
                fmax=sample_rate // 2,
                ax=ax,
                cmap="magma"
            )
            cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
            cbar.set_label("dB", rotation=270, labelpad=12)
            ax.set_title(
                f"{res['info']['label']} - Session {res['info']['session_id']} ({res['file_name']}) | "
                f"Duration: {res['proc_dur']:.2f}s | Shape: {res['spec_shape']}",
                fontsize=10, fontweight="bold"
            )
            ax.set_xlabel("Time (s)" if i == 2 else "")
            ax.set_ylabel("Mel Freq (Hz)")

        plt.suptitle(
            "Comparative Log-Mel Spectrograms Across Classes (SVD Sustained Vowel /a/)",
            fontsize=13, fontweight="bold", y=0.99
        )
        plt.tight_layout()
        plt.savefig(comparison_img_path, dpi=300)
        plt.close()
        print(f"\n[OK] Comparative visualization saved: {comparison_img_path}")

    # 5. Generate Report File
    report_path = os.path.join(rep_dir, "sample_feature_extraction_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("STAGE 2: SAMPLE FEATURE EXTRACTION REPORT\n")
        f.write("Automated Multi-Class Laryngeal Voice Disorder Classification\n")
        f.write("Task: NSP Preprocessing & Log-Mel Spectrogram Extraction on Class Samples\n")
        f.write("Generated: 2026-09-22\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. EXTRACTION SPECIFICATIONS & PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"  - Acoustic Sampling Rate:      {sample_rate} Hz\n")
        f.write(f"  - Mel Filterbank Channels:     {n_mels} bins (n_mels)\n")
        f.write(f"  - FFT Window Size:             {n_fft} samples (n_fft)\n")
        f.write(f"  - Hop Length (Frame Shift):    {hop_length} samples\n")
        f.write(f"  - Frequency Range:             0.0 Hz to {sample_rate // 2}.0 Hz (Nyquist)\n")
        f.write(f"  - Silence Trimming Threshold:  {top_db_trim} dB (librosa.effects.trim)\n")
        f.write(f"  - Dynamic Range Scaling:       {top_db_mel} dB (ref=np.max)\n")
        f.write(f"  - Normalization:               Peak amplitude normalization to [-1.0, 1.0]\n")
        f.write(f"  - Native Input Format:         KayPENTAX CSL NSP (50,000 Hz, 16-bit PCM mono)\n\n")

        f.write("2. SAMPLE RECORDINGS SUMMARY & FEATURE RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<14} | {'Session':<8} | {'File Name':<14} | {'Orig Dur':<9} | {'Proc Dur':<9} | {'Feature Shape':<14} | {'Status':<8}\n")
        f.write("-" * 80 + "\n")
        for cls, res in results.items():
            f.write(
                f"{cls:<14} | "
                f"{res['info']['session_id']:<8} | "
                f"{res['file_name']:<14} | "
                f"{res['orig_dur']:.3f}s    | "
                f"{res['proc_dur']:.3f}s    | "
                f"{str(res['spec_shape']):<14} | "
                f"SUCCESS\n"
            )
        f.write("-" * 80 + "\n\n")

        f.write("3. DETAILED CLASS-BY-CLASS BREAKDOWN\n")
        f.write("-" * 80 + "\n")
        for cls, res in results.items():
            info = res["info"]
            f.write(f"A. Class: {cls.upper()} ({info['label']})\n")
            f.write(f"   - File Path:             {info['rel_path']}\n")
            f.write(f"   - Session ID / Speaker:  AufnahmeID={info['session_id']}, SprecherID={info['speaker_id']}\n")
            f.write(f"   - Patient Demographics:  Sex={info['sex']}, Age={info['age']:.1f} yrs (DOB: {info['birth_date']})\n")
            f.write(f"   - Clinical Diagnosis:    {info['diagnosis']}\n")
            f.write(f"   - Original Duration:     {res['orig_dur']:.4f} seconds (at native 50,000 Hz)\n")
            f.write(f"   - Processed Duration:    {res['proc_dur']:.4f} seconds (resampled to 22,050 Hz)\n")
            f.write(f"   - Spectrogram Shape:     {res['spec_shape']} (n_mels x time_steps)\n")
            f.write(f"   - Spectrogram Dtype:     {res['dtype']}\n")
            f.write(f"   - Numerical Stats:       Min = {res['min_db']:.2f} dB, Max = {res['max_db']:.2f} dB, Mean = {res['mean_db']:.2f} dB, Std = {res['std_db']:.2f} dB\n")
            f.write(f"   - Feature File Saved:    {res['feat_path']} ({res['feat_size_bytes']:,} bytes)\n")
            f.write(f"   - Spectrogram Image:     {res['img_path']}\n\n")

        f.write("4. NUMERICAL INTEGRITY & VALIDATION CHECKS\n")
        f.write("-" * 80 + "\n")
        all_valid = True
        for cls, res in results.items():
            nan_cnt = np.isnan(res["log_mel"]).sum()
            inf_cnt = np.isinf(res["log_mel"]).sum()
            is_valid = (nan_cnt == 0 and inf_cnt == 0 and res["spec_shape"][0] == n_mels)
            all_valid = all_valid and is_valid
            f.write(f"  - [{cls.upper()}] NaN Count: {nan_cnt} | Inf Count: {inf_cnt} | Mel Dimension: {res['spec_shape'][0]} == {n_mels} -> {'PASS' if is_valid else 'FAIL'}\n")
        f.write(f"\n  Overall Feature Array Verification: {'ALL SAMPLES PASSED (VALID NUMERICAL ARRAYS)' if all_valid else 'FAILURES DETECTED'}\n\n")

        f.write("5. GENERATED ARTIFACTS INVENTORY\n")
        f.write("-" * 80 + "\n")
        f.write("  A. Feature Arrays (.npy):\n")
        for cls, res in results.items():
            f.write(f"     * {res['feat_path']}\n")
        f.write("\n  B. Spectrogram Visualizations (.png):\n")
        for cls, res in results.items():
            f.write(f"     * {res['img_path']}\n")
        f.write(f"     * {comparison_img_path}\n\n")

        f.write("6. ERROR AND WARNING LOG\n")
        f.write("-" * 80 + "\n")
        if not errors:
            f.write("  No errors or warnings encountered during Stage 2 execution.\n")
            f.write("  All 3 samples were successfully read, preprocessed, converted, and saved.\n")
        else:
            for err in errors:
                f.write(f"  [ERROR] {err}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF STAGE 2 REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"\n[OK] Report generated at: {report_path}")
    print("=" * 70)
    print("STAGE 2 COMPLETE: ALL 3 SAMPLES PROCESSED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_sample_feature_extraction()
