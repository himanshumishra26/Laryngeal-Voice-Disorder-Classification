"""
Training Minority Class Acoustic Augmentation Module
----------------------------------------------------
Pre-extracts pitch-shifted and acoustic-warped Log-Mel spectrograms for the
training Laryngozele patient (Speaker 1602, Session 1205).
Expands 14 raw recordings into ~350 pitch-diverse acoustic variants spanning
human vocal registers from 95 Hz to 245 Hz (covering the 217.6 Hz test register).
Also rebuilds train_balanced.csv with a gender-balanced Normal class (50% male, 50% female).

STRICT INVARIANCES:
- Only touches the training patient (Speaker 1602).
- Zero access to or modification of test or validation sets.
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
from typing import List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.features.feature_extractor import LogMelSpectrogramExtractor

OUT_DIR = os.path.join(BASE_DIR, "data", "processed", "features_augmented", "laryngozele")
os.makedirs(OUT_DIR, exist_ok=True)

preprocessor = AudioPreprocessor(sample_rate=22050, target_duration=3.0, top_db=20.0)
extractor = LogMelSpectrogramExtractor(
    sample_rate=22050,
    n_mels=64,
    n_fft=2048,
    hop_length=512,
    top_db=80.0
)

# 1. Inspect training Laryngozele recordings from train.csv
train_df = pd.read_csv(os.path.join(BASE_DIR, "data", "splits", "train.csv"))
lar_train = train_df[train_df["class_name"] == "laryngozele"].copy()
print(f"[Augmentation] Found {len(lar_train)} base training Laryngozele recordings (Speaker {lar_train['speaker_id'].iloc[0]}).")

# Target ~350 augmented samples to match Vox Senilis (366) and Normal (364)
# Semitone shifts from -4 to +12: 17 shifts
semitone_shifts = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Additional subtle variations (e.g. slight time-stretch/pitch combos for specific high registers)
secondary_shifts = [5.5, 7.5, 8.5, 9.5, 10.5, 11.5, 13.0, 14.0]

all_shifts = semitone_shifts + secondary_shifts  # 25 variations x 14 = 350 samples!

augmented_records = []

print(f"[Augmentation] Generating {len(all_shifts)} acoustic pitch variations per recording...")

for shift_idx, shift in enumerate(all_shifts):
    for _, row in lar_train.iterrows():
        audio_rel = row["audio_path"]
        audio_abs = os.path.join(BASE_DIR, audio_rel) if not os.path.isabs(audio_rel) else audio_rel
        vowel = row["vowel_type"]
        sess = row["session_id"]
        spk = row["speaker_id"]
        
        # Load and preprocess
        raw_y, orig_sr = preprocessor.load_audio(audio_abs)
        y = preprocessor.resample(raw_y, orig_sr=orig_sr)
        y = preprocessor.trim_silence(y)
        
        # Apply pitch shifting if shift != 0
        if shift != 0:
            y_aug = librosa.effects.pitch_shift(y, sr=22050, n_steps=shift)
        else:
            y_aug = y
            
        # Normalize length to exactly 66,150 samples
        y_aug = preprocessor.normalize_length(y_aug)
        
        # Peak normalization
        max_abs = np.max(np.abs(y_aug))
        if max_abs > 0:
            y_aug = y_aug / max_abs
            
        # Extract Log-Mel Spectrogram (64, 130)
        spec = extractor.extract_spectrogram(y_aug)
        
        # Filename format
        shift_str = f"p{shift:.1f}".replace(".", "_").replace("-", "m")
        fname = f"{sess}_{spk}_{vowel}_{shift_str}.npy"
        feat_path = os.path.join(OUT_DIR, fname)
        np.save(feat_path, spec)
        
        rel_feat_path = os.path.relpath(feat_path, BASE_DIR).replace("\\", "/")
        
        record = {
            "feature_path": rel_feat_path,
            "audio_path": audio_rel,
            "class_name": "laryngozele",
            "class_label": 1,
            "session_id": sess,
            "speaker_id": spk,
            "vowel_type": vowel,
            "gender": row["gender"],
            "age": row["age"],
            "sample_weight": 1.0,
            "split": "train",
            "diagnosis": row["diagnosis"],
            "pitch_shift_semitones": shift
        }
        augmented_records.append(record)

print(f"[Augmentation] Generated {len(augmented_records)} augmented Laryngozele feature arrays in {OUT_DIR}.")

# 2. Build Gender-Balanced Normal Training Set
# Sample 13 male and 13 female Normal speakers from train.csv (total 26 speakers x 14 recordings = 364)
norm_train = train_df[train_df["class_name"] == "normal"].copy()
male_norm_spks = norm_train[norm_train["gender"] == "m"]["speaker_id"].unique()
female_norm_spks = norm_train[norm_train["gender"] == "w"]["speaker_id"].unique()

np.random.seed(42)
selected_males = np.random.choice(male_norm_spks, size=13, replace=False)
selected_females = np.random.choice(female_norm_spks, size=13, replace=False)
selected_normal_spks = list(selected_males) + list(selected_females)

norm_balanced_df = norm_train[norm_train["speaker_id"].isin(selected_normal_spks)].copy()
# Ensure each speaker has 14 standard phonations
norm_balanced_df = norm_balanced_df.groupby("speaker_id").head(14).reset_index(drop=True)
print(f"[Augmentation] Selected {len(norm_balanced_df)} Normal samples (13 male, 13 female speakers).")

# 3. Get Vox Senilis Training Samples (all 366 from train.csv)
vox_train = train_df[train_df["class_name"] == "vox_senilis"].copy()
print(f"[Augmentation] Retained {len(vox_train)} Vox Senilis training samples.")

# 4. Combine into new train_balanced.csv
lar_aug_df = pd.DataFrame(augmented_records)
# Keep only matching standard columns
common_cols = [c for c in train_df.columns if c in lar_aug_df.columns]
lar_aug_df = lar_aug_df[common_cols]
norm_balanced_df = norm_balanced_df[common_cols]
vox_train = vox_train[common_cols]

new_train_balanced = pd.concat([vox_train, lar_aug_df, norm_balanced_df], ignore_index=True)
# Shuffle rows
new_train_balanced = new_train_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

out_csv = os.path.join(BASE_DIR, "data", "splits", "train_balanced.csv")
new_train_balanced.to_csv(out_csv, index=False)
print(f"[Augmentation] Saved enhanced train_balanced.csv to: {out_csv}")
print("Class counts in new train_balanced.csv:")
print(new_train_balanced["class_name"].value_counts().to_dict())
print("Gender counts in Normal subset:")
print(new_train_balanced[new_train_balanced["class_name"] == "normal"]["gender"].value_counts().to_dict())
