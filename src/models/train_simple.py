"""
train_simple.py
================
Trains the hybrid Random Forest + Autoencoder NIDS on
data/raw/UNSW-NB15_clean.csv (produced by src/data/build_dataset.py).

This is the ONE training script to use. If you have other variants lying
around in src/models/ (train_models.py, train_models_fixed.py,
train_final_model.py, retrain_and_save_model.py) -- archive them; having
five scripts that may or may not match the current dataset schema is how
the 25-vs-49-column confusion happened in the first place. Keep one
source of truth.
"""
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

print("=" * 60)
print("HYBRID NIDS TRAINING")
print("=" * 60)

dataset_path = "../../data/raw/UNSW-NB15_clean.csv"
if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset not found at {dataset_path}")
    print("Run: python src/data/build_dataset.py first")
    exit(1)

print("\nLoading dataset...")
df = pd.read_csv(dataset_path)
print(f"Loaded {len(df)} samples with {df.shape[1]} columns "
      f"({df.shape[1] - 1} features + label)")

X = df.drop("label", axis=1).values.astype(np.float32)
y = df["label"].values.astype(np.int32)
N_FEATURES = X.shape[1]
print(f"Benign: {np.sum(y == 0)}, Malicious: {np.sum(y == 1)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_benign = X_train[y_train == 0]
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Benign for autoencoder: {len(X_benign)}")

print("\nScaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_benign = scaler.transform(X_benign)

print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print("Training Autoencoder...")
input_layer = keras.layers.Input(shape=(N_FEATURES,))
encoded = keras.layers.Dense(32, activation="relu")(input_layer)
decoded = keras.layers.Dense(N_FEATURES, activation="linear")(encoded)
autoencoder = keras.Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(X_benign, X_benign, epochs=20, batch_size=256, verbose=0)

reconstructions = autoencoder.predict(X_benign, verbose=0)
errors = np.mean(np.square(X_benign - reconstructions), axis=1)
threshold = np.percentile(errors, 95)
print(f"Threshold: {threshold:.4f}")

y_pred_rf = rf.predict(X_test)
test_recon = autoencoder.predict(X_test, verbose=0)
test_errors = np.mean(np.square(X_test - test_recon), axis=1)
y_pred_hybrid = np.logical_or(y_pred_rf == 1, test_errors > threshold).astype(int)

print("\nRandom Forest Results:")
print(classification_report(y_test, y_pred_rf, target_names=["Benign", "Malicious"]))
print("\nHybrid Model Results:")
print(classification_report(y_test, y_pred_hybrid, target_names=["Benign", "Malicious"]))

print("\nSaving models...")
os.makedirs("../../models/saved", exist_ok=True)
joblib.dump(rf, "../../models/saved/rf_final.pkl")
autoencoder.save("../../models/saved/autoencoder_final.keras")
np.save("../../models/saved/threshold_final.npy", threshold)
joblib.dump(scaler, "../../models/saved/scaler_final.pkl")
print(f"Models saved. Trained on {N_FEATURES} features "
      f"(update ml_engine.py EXPECTED_FEATURE_COUNT if this differs from before).")
print("\nTraining complete!")
