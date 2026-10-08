"""
Hierarchical Dataset Builder
----------------------------
Prepares task-specific training and validation manifests for Two-Stage Hierarchical Classification:
- Stage 1: Healthy (Normal, label 0) vs. Pathological (Laryngozele + Vox Senilis, label 1)
- Stage 2: Laryngozele (label 0) vs. Vox Senilis (label 1)

STRICT INVARIANCE:
- Test sets (test.csv and test_balanced.csv) remain 100% UNTOUCHED and UNSEEN.
- Only uses samples strictly from training and validation partitions.
- Zero speaker leakage. Quarantined Speaker 73 excluded.
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

splits_dir = os.path.join(BASE_DIR, "data", "splits")
out_dir = os.path.join(splits_dir, "hierarchical")
os.makedirs(out_dir, exist_ok=True)

# Load existing balanced train and val manifests
train_balanced_path = os.path.join(splits_dir, "train_balanced.csv")
val_balanced_path = os.path.join(splits_dir, "val_balanced.csv")

train_df = pd.read_csv(train_balanced_path)
val_df = pd.read_csv(val_balanced_path)

print(f"[Hierarchical] Loaded train_balanced: {len(train_df)} samples, val_balanced: {len(val_df)} samples.")

# -------------------------------------------------------------
# STAGE 1 DATASETS: Healthy (Normal = 0) vs. Pathological (= 1)
# -------------------------------------------------------------
# Train Stage 1
train_norm = train_df[train_df["class_name"] == "normal"].copy()
train_path = train_df[train_df["class_name"].isin(["laryngozele", "vox_senilis"])].copy()

# 1:1 balanced Stage 1: 363 Normal vs 363 Pathological (sample 182 Vox Senilis + 181 Laryngozele)
np.random.seed(42)
vox_sample = train_path[train_path["class_name"] == "vox_senilis"].sample(n=182, random_state=42)
lar_sample = train_path[train_path["class_name"] == "laryngozele"].sample(n=181, random_state=42)
train_path_balanced = pd.concat([vox_sample, lar_sample], ignore_index=True)

train_norm["class_label"] = 0
train_norm["stage1_label"] = 0
train_norm["stage1_class_name"] = "healthy"

train_path_balanced["class_label"] = 1
train_path_balanced["stage1_label"] = 1
train_path_balanced["stage1_class_name"] = "pathological"

train_stage1 = pd.concat([train_norm, train_path_balanced], ignore_index=True)
train_stage1 = train_stage1.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Val Stage 1
val_norm = val_df[val_df["class_name"] == "normal"].copy()
val_path = val_df[val_df["class_name"].isin(["laryngozele", "vox_senilis"])].copy()

val_norm["class_label"] = 0
val_norm["stage1_label"] = 0
val_norm["stage1_class_name"] = "healthy"

val_path["class_label"] = 1
val_path["stage1_label"] = 1
val_path["stage1_class_name"] = "pathological"

val_stage1 = pd.concat([val_norm, val_path], ignore_index=True)
val_stage1 = val_stage1.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Save Stage 1 manifests
train_stage1_path = os.path.join(out_dir, "train_stage1.csv")
val_stage1_path = os.path.join(out_dir, "val_stage1.csv")
train_stage1.to_csv(train_stage1_path, index=False)
val_stage1.to_csv(val_stage1_path, index=False)

print(f"[Hierarchical] Stage 1 Train saved: {train_stage1_path} ({len(train_stage1)} rows: {train_stage1['stage1_class_name'].value_counts().to_dict()})")
print(f"[Hierarchical] Stage 1 Val saved: {val_stage1_path} ({len(val_stage1)} rows: {val_stage1['stage1_class_name'].value_counts().to_dict()})")

# -------------------------------------------------------------
# STAGE 2 DATASETS: Laryngozele (0) vs. Vox Senilis (1)
# -------------------------------------------------------------
# Train Stage 2: all pathological samples from train_balanced
train_lar = train_df[train_df["class_name"] == "laryngozele"].copy()
train_vox = train_df[train_df["class_name"] == "vox_senilis"].copy()

train_lar["class_label"] = 0
train_lar["stage2_label"] = 0
train_lar["stage2_class_name"] = "laryngozele"

train_vox["class_label"] = 1
train_vox["stage2_label"] = 1
train_vox["stage2_class_name"] = "vox_senilis"

train_stage2 = pd.concat([train_lar, train_vox], ignore_index=True)
train_stage2 = train_stage2.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Val Stage 2: all pathological samples from val_balanced
val_lar = val_df[val_df["class_name"] == "laryngozele"].copy()
val_vox = val_df[val_df["class_name"] == "vox_senilis"].copy()

val_lar["class_label"] = 0
val_lar["stage2_label"] = 0
val_lar["stage2_class_name"] = "laryngozele"

val_vox["class_label"] = 1
val_vox["stage2_label"] = 1
val_vox["stage2_class_name"] = "vox_senilis"

val_stage2 = pd.concat([val_lar, val_vox], ignore_index=True)
val_stage2 = val_stage2.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Save Stage 2 manifests
train_stage2_path = os.path.join(out_dir, "train_stage2.csv")
val_stage2_path = os.path.join(out_dir, "val_stage2.csv")
train_stage2.to_csv(train_stage2_path, index=False)
val_stage2.to_csv(val_stage2_path, index=False)

print(f"[Hierarchical] Stage 2 Train saved: {train_stage2_path} ({len(train_stage2)} rows: {train_stage2['stage2_class_name'].value_counts().to_dict()})")
print(f"[Hierarchical] Stage 2 Val saved: {val_stage2_path} ({len(val_stage2)} rows: {val_stage2['stage2_class_name'].value_counts().to_dict()})")

# -------------------------------------------------------------
# ZERO-LEAKAGE VERIFICATION
# -------------------------------------------------------------
test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
test_speakers = set(test_df["speaker_id"].astype(str))

all_stage_speakers = set(train_stage1["speaker_id"].astype(str)) | \
                     set(val_stage1["speaker_id"].astype(str)) | \
                     set(train_stage2["speaker_id"].astype(str)) | \
                     set(val_stage2["speaker_id"].astype(str))

overlap = all_stage_speakers & test_speakers
assert len(overlap) == 0, f"LEAKAGE DETECTED! Overlapping speakers: {overlap}"
assert "73" not in all_stage_speakers, "Quarantined Speaker 73 found in hierarchical datasets!"

print("\n[Hierarchical] Zero-leakage verification: PASSED!")
print("  - Overlap with test speakers: 0")
print("  - Quarantined Speaker 73 excluded: TRUE")
print("  - test.csv and test_balanced.csv remain 100% UNTOUCHED.")
