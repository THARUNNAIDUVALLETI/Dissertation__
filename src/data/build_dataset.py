"""
build_dataset.py
=================
Rebuilds data/raw/UNSW-NB15_clean.csv from scratch, from the original
public UNSW-NB15 dataset. Run this once to recover from losing the
original clean CSV / trained models.

What it does:
  1. Downloads the official UNSW_NB15_training-set.csv (175,341 rows) and
     UNSW_NB15_testing-set.csv (82,332 rows) partitions -- these are the
     standard pre-engineered partitions published alongside the dataset
     (Moustafa & Slay, 2015), re-hosted on GitHub for direct download.
  2. Combines them (257,673 rows total).
  3. Drops `id` (not a feature) and `attack_cat` (multi-class label --
     kept aside in a separate file in case you want it later; this
     project trains a BINARY classifier using `label` only, matching
     train_simple.py).
  4. Label-encodes the 3 categorical columns (proto, service, state) so
     every feature is numeric. The encoders are saved so anything that
     needs to reproduce this encoding later (e.g. a future live-capture
     pipeline) can reuse the exact same mapping.
  5. Stratified-samples down to ~50,000 rows (matching the sample size
     already documented in PROJECT_SUMMARY.md / DISSERTATION_RESULTS.md),
     preserving the original benign/malicious ratio.
  6. Writes data/raw/UNSW-NB15_clean.csv: 42 numeric feature columns +
     `label` (43 columns total). Columns keep their REAL names (dur,
     proto, sttl, ...) rather than being anonymized -- this is an
     improvement over the lost original, since it lets you discuss
     feature importance meaningfully in your results chapter.

IMPORTANT: this produces a NEW clean dataset, not a byte-for-byte
recovery of the one that was lost. Re-run src/models/train_simple.py
afterwards to get fresh (real, genuinely computed) model files and
evaluation metrics -- your old confusion-matrix numbers in
DISSERTATION_RESULTS.md were specific to the lost data/model and won't
match exactly. Document in your methodology chapter that the dataset
was rebuilt from the original UNSW-NB15 partitions after a local data
loss; that's a normal, explainable thing to note, not something to hide.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

TRAIN_URL = "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv"
TEST_URL = "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv"

CATEGORICAL_COLS = ["proto", "service", "state"]
DROP_COLS = ["id", "attack_cat"]  # attack_cat saved separately, not used as a model feature
TARGET_SAMPLE_SIZE = 50_000
RANDOM_STATE = 42


def download_or_load(url: str, local_fallback: Path) -> pd.DataFrame:
    if local_fallback.exists():
        print(f"Using local file: {local_fallback}")
        return pd.read_csv(local_fallback)
    print(f"Downloading: {url}")
    df = pd.read_csv(url)
    local_fallback.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(local_fallback, index=False)
    return df


def main():
    project_root = Path(__file__).resolve().parents[2]  # src/data/build_dataset.py -> project root
    raw_dir = project_root / "data" / "raw"
    models_dir = project_root / "models" / "saved"
    raw_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("REBUILDING UNSW-NB15 DATASET")
    print("=" * 60)

    train = download_or_load(TRAIN_URL, raw_dir / "_unsw_train_raw.csv")
    test = download_or_load(TEST_URL, raw_dir / "_unsw_test_raw.csv")

    df = pd.concat([train, test], ignore_index=True)
    print(f"\nCombined: {df.shape[0]} rows, {df.shape[1]} columns")

    # Save attack_cat mapping aside before dropping it, in case multi-class
    # work is wanted later.
    attack_cat_path = raw_dir / "attack_categories.csv"
    df[["attack_cat", "label"]].to_csv(attack_cat_path, index=False)
    print(f"Saved attack category reference -> {attack_cat_path}")

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"Label-encoded '{col}': {len(le.classes_)} categories")

    encoders_path = models_dir / "categorical_encoders.pkl"
    joblib.dump(encoders, encoders_path)
    print(f"Saved categorical encoders -> {encoders_path}")

    if df.isnull().sum().sum() > 0:
        print("WARNING: nulls found, dropping affected rows")
        df = df.dropna()

    # Stratified sample down to TARGET_SAMPLE_SIZE, preserving class ratio.
    if len(df) > TARGET_SAMPLE_SIZE:
        frac = TARGET_SAMPLE_SIZE / len(df)
        parts = [
            group.sample(frac=frac, random_state=RANDOM_STATE)
            for _, group in df.groupby("label")
        ]
        df = pd.concat(parts, ignore_index=True)

    feature_cols = [c for c in df.columns if c != "label"]
    print(f"\nFinal dataset: {len(df)} rows, {len(feature_cols)} feature columns + label")
    print(f"Class balance -> benign: {(df['label'] == 0).sum()}, malicious: {(df['label'] == 1).sum()}")

    out_path = raw_dir / "UNSW-NB15_clean.csv"
    df[feature_cols + ["label"]].to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    # Clean up the large intermediate raw downloads -- keep only the final clean CSV.
    for tmp in [raw_dir / "_unsw_train_raw.csv", raw_dir / "_unsw_test_raw.csv"]:
        if tmp.exists():
            tmp.unlink()

    print("\nDone. Next step: python src/models/train_simple.py")


if __name__ == "__main__":
    main()
