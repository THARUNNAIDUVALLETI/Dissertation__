"""
train_all_models.py
====================
Trains three baseline ML models for comparison against the hybrid
RF + Autoencoder system. Run AFTER train_simple.py (which trains
the main RF and Autoencoder).

Models trained:
  - Decision Tree      (fast, interpretable baseline)
  - Logistic Regression (linear baseline)
  - Gradient Boosting  (strong ensemble baseline)

All models are saved to models/saved/ and their metrics are appended
to results/metrics/comparison.csv so the dashboard bar chart updates
automatically after running `python manage.py seed_metrics`.

NOTE: You do NOT need to download any pre-trained models. Everything
trains from scratch on your local UNSW-NB15_clean.csv. Typical time:
  - Decision Tree:       < 30 seconds
  - Logistic Regression: 1-2 minutes
  - Gradient Boosting:   3-5 minutes

No GPU required. All models use scikit-learn (CPU only).
"""
import os
import sys
import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

print("=" * 60)
print("BASELINE MODEL TRAINING")
print("=" * 60)

DATASET_PATH = "../../data/raw/UNSW-NB15_clean.csv"
MODELS_DIR = Path("../../models/saved")
RESULTS_PATH = Path("../../results/metrics/comparison.csv")

if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset not found at {DATASET_PATH}")
    print("Run: python main.py build-dataset first")
    sys.exit(1)

print("\nLoading dataset...")
df = pd.read_csv(DATASET_PATH)
X = df.drop("label", axis=1).values.astype(np.float32)
y = df["label"].values.astype(np.int32)
print(f"Shape: {X.shape} | Benign: {np.sum(y==0)} | Malicious: {np.sum(y==1)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Reuse the saved scaler from train_simple.py so everything is on the same scale
scaler_path = MODELS_DIR / "scaler_final.pkl"
if scaler_path.exists():
    print("Reusing saved scaler from train_simple.py...")
    scaler = joblib.load(scaler_path)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
else:
    print("Scaler not found, fitting fresh scaler...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"\n  {name}:")
    print(f"    Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    FPR:       {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"    FNR:       {fnr:.4f} ({fnr*100:.2f}%)")
    print(f"    Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    return {
        "model": name, "accuracy": acc, "precision": prec,
        "recall": rec, "f1_score": f1,
        "false_positive_rate": fpr, "false_negative_rate": fnr
    }


models_to_train = [
    ("Decision Tree", DecisionTreeClassifier(max_depth=20, random_state=42)),
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
    ("Gradient Boosting", GradientBoostingClassifier(
        n_estimators=100, max_depth=5, random_state=42, subsample=0.8
    )),
]

filename_map = {
    "Decision Tree": "decision_tree.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
}

new_metrics = []
MODELS_DIR.mkdir(parents=True, exist_ok=True)

for name, model in models_to_train:
    print(f"\nTraining {name}...")
    model.fit(X_train_s, y_train)
    metrics = evaluate(name, model, X_test_s, y_test)
    new_metrics.append(metrics)
    save_path = MODELS_DIR / filename_map[name]
    joblib.dump(model, save_path)
    print(f"  Saved: {save_path}")

# Update comparison.csv: keep existing rows, replace any with same model name,
# then append new results so the dashboard chart shows everything
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
existing = {}
if RESULTS_PATH.exists():
    with open(RESULTS_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing[row["model"]] = row

for m in new_metrics:
    existing[m["model"]] = {k: str(v) for k, v in m.items()}

fieldnames = ["model", "accuracy", "precision", "recall",
              "f1_score", "false_positive_rate", "false_negative_rate"]
with open(RESULTS_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in existing.values():
        writer.writerow({k: row[k] for k in fieldnames})

print(f"\nUpdated: {RESULTS_PATH}")
print("\nBaseline training complete!")
print("\nNext steps:")
print("  cd nids_dashboard")
print("  python manage.py seed_metrics")
print("  python manage.py runserver")
print("\nThe 'Algorithm Accuracy Comparison' chart will now show all 5 models.")
