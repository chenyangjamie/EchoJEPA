"""
Generate EchoNet-Dynamic CSV manifests for EchoJEPA probe training and inference.

Output format (space-delimited, no header):
  /absolute/path/to/video.avi  <z-score-normalized EF>

Scaler is fitted on TRAIN split only to prevent data leakage.
target_mean and target_std must match the values in the eval/inference YAML configs.
"""

import os
import pickle
import pandas as pd
import numpy as np

ECHONET_DIR = "/raid2/nextgen/new_projects/EchoJEPA/EchoNet_Dynamic"
VIDEOS_DIR  = os.path.join(ECHONET_DIR, "Videos")
FILELIST    = os.path.join(ECHONET_DIR, "FileList.csv")
OUT_DIR     = "/raid2/nextgen/new_projects/EchoJEPA/data/csv"

# Must match configs/eval/vitl/echonet_dynamic_lvef.yaml
TARGET_MEAN = 55.7776
TARGET_STD  = 12.4064

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(FILELIST)
    print(f"Loaded {len(df)} entries from FileList.csv")
    print(f"Columns: {list(df.columns)}")
    print(f"Splits:  {df['Split'].value_counts().to_dict()}")

    # Build absolute video paths
    df["path"] = df["FileName"].apply(
        lambda fn: os.path.join(VIDEOS_DIR, f"{fn}.avi")
    )

    # Verify a few paths exist
    missing = df["path"].apply(lambda p: not os.path.exists(p)).sum()
    if missing:
        print(f"WARNING: {missing} video files not found on disk")
    else:
        print(f"All {len(df)} video files found")

    # Z-score normalise EF using fixed mean/std (matching YAML config)
    # We use the pre-computed constants rather than re-fitting to ensure
    # exact match with target_mean/target_std in the YAML.
    df["ef_norm"] = (df["EF"] - TARGET_MEAN) / TARGET_STD

    print(f"\nEF stats (train split):")
    train_ef = df.loc[df["Split"] == "TRAIN", "EF"]
    print(f"  mean={train_ef.mean():.4f}  std={train_ef.std():.4f}")
    print(f"  (using fixed target_mean={TARGET_MEAN}, target_std={TARGET_STD})")

    # Write per-split CSVs
    splits = {"TRAIN": "echonet_dynamic_train.csv",
              "VAL":   "echonet_dynamic_val.csv",
              "TEST":  "echonet_dynamic_test.csv"}

    for split, fname in splits.items():
        subset = df[df["Split"] == split][["path", "ef_norm"]]
        out_path = os.path.join(OUT_DIR, fname)
        subset.to_csv(out_path, sep=" ", header=False, index=False, float_format="%.6f")
        print(f"  {split:5s}: {len(subset):5d} rows  →  {out_path}")

    # Save scaler info for reference (inverse-transform predictions later)
    scaler_info = {"mean": TARGET_MEAN, "std": TARGET_STD}
    scaler_path = os.path.join(OUT_DIR, "lvef_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_info, f)
    print(f"\nScaler saved to {scaler_path}")
    print("Done.")

if __name__ == "__main__":
    main()
