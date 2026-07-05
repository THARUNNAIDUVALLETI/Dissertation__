"""
ml_engine.py
============
Loads the trained Random Forest + Autoencoder hybrid model and runs real
predictions. This replaces the np.random.random() placeholder logic that
was in the old views.py.

Expected files (paths are configurable via Django settings.MODELS_DIR or
the NIDS_MODELS_DIR environment variable):
    rf_final.pkl            -- joblib-dumped sklearn RandomForestClassifier
    autoencoder_final.keras -- keras.Model, trained on benign-only data
    scaler_final.pkl        -- joblib-dumped sklearn StandardScaler
    threshold_final.npy     -- numpy scalar, 95th percentile reconstruction
                                error on benign training data

IMPORTANT: train_simple.py builds X as `df.drop('label', axis=1).values`,
i.e. plain positional column order. Anything that builds a feature vector
for this engine MUST supply the same number of columns in the same order
as data/raw/UNSW-NB15_clean.csv -- the model has no column names to
validate against, only a count (EXPECTED_FEATURE_COUNT below).

EXPECTED_FEATURE_COUNT = 42 matches the dataset produced by
src/data/build_dataset.py (dur, proto, service, state, spkts, ... 42
named UNSW-NB15 features, label-encoded categoricals, label dropped).
If you regenerate the dataset with a different feature set, update this
constant to match -- this is exactly the kind of mismatch that caused
the "Expected 49, found 25" error earlier.
"""
from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

EXPECTED_FEATURE_COUNT = 42

MODEL_FILENAMES = {
    "rf": "rf_final.pkl",
    "autoencoder": "autoencoder_final.keras",
    "scaler": "scaler_final.pkl",
    "threshold": "threshold_final.npy",
}


def _candidate_dirs() -> list[Path]:
    """Where to look for the models/saved/ directory, in priority order."""
    dirs = []

    env_dir = os.environ.get("NIDS_MODELS_DIR")
    if env_dir:
        dirs.append(Path(env_dir))

    try:
        from django.conf import settings as dj_settings

        configured = getattr(dj_settings, "MODELS_DIR", None)
        if configured:
            dirs.append(Path(configured))
    except Exception:
        pass

    here = Path(__file__).resolve()
    dirs.append(here.parents[2] / "models" / "saved")
    dirs.append(here.parents[1] / "models" / "saved")
    dirs.append(here.parents[3] / "models" / "saved")
    # Common Kali layouts
    dirs.append(Path("/home/kali/Desktop/Dissertation-/models/saved"))
    dirs.append(Path("/home/kali/project/models/saved"))
    dirs.append(Path("/home/kali/Dissertation-/models/saved"))
    dirs.append(Path.home() / "Desktop" / "Dissertation-" / "models" / "saved")
    dirs.append(Path.home() / "project" / "models" / "saved")
    dirs.append(Path.home() / "Dissertation-" / "models" / "saved")

    # de-dupe while preserving order
    seen, unique = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


@dataclass
class PredictionResult:
    rf_prediction: int
    rf_confidence: float
    reconstruction_error: float
    is_anomaly: bool
    verdict: str
    severity: str


class HybridNIDSEngine:
    """Singleton-style wrapper around the four model artifacts."""

    def __init__(self):
        self.rf_model = None
        self.autoencoder = None
        self.scaler = None
        self.threshold: Optional[float] = None
        self.models_dir: Optional[Path] = None
        self.load_error: Optional[str] = None
        self._load()

    # -- loading -----------------------------------------------------
    def _load(self):
        import joblib

        for candidate in _candidate_dirs():
            rf_path = candidate / MODEL_FILENAMES["rf"]
            if rf_path.exists():
                self.models_dir = candidate
                break
        else:
            self.load_error = (
                "Could not find rf_final.pkl in any candidate models/saved/ "
                "directory. Set NIDS_MODELS_DIR or settings.MODELS_DIR."
            )
            logger.warning(self.load_error)
            return

        try:
            self.rf_model = joblib.load(self.models_dir / MODEL_FILENAMES["rf"])
            logger.info("Loaded Random Forest from %s", self.models_dir)
        except Exception as exc:
            self.load_error = f"Failed to load rf_final.pkl: {exc}"
            logger.error(self.load_error)
            return

        try:
            self.scaler = joblib.load(self.models_dir / MODEL_FILENAMES["scaler"])
            logger.info("Loaded scaler from %s", self.models_dir)
        except Exception as exc:
            self.load_error = f"Failed to load scaler_final.pkl: {exc}"
            logger.error(self.load_error)
            return

        try:
            self.threshold = float(np.load(self.models_dir / MODEL_FILENAMES["threshold"]))
            logger.info("Loaded anomaly threshold = %.4f", self.threshold)
        except Exception as exc:
            self.load_error = f"Failed to load threshold_final.npy: {exc}"
            logger.error(self.load_error)
            return

        try:
            # Imported lazily: tensorflow is heavy and some dev machines
            # (e.g. this sandbox) won't have it installed. The dashboard
            # still functions in "RF-only" mode if this fails.
            from tensorflow import keras

            self.autoencoder = keras.models.load_model(
                self.models_dir / MODEL_FILENAMES["autoencoder"]
            )
            logger.info("Loaded autoencoder from %s", self.models_dir)
        except Exception as exc:
            self.load_error = f"Autoencoder unavailable, running RF-only: {exc}"
            logger.warning(self.load_error)
            self.autoencoder = None

    # -- status --------------------------------------------------------
    @property
    def rf_ready(self) -> bool:
        return self.rf_model is not None and self.scaler is not None

    @property
    def autoencoder_ready(self) -> bool:
        return self.autoencoder is not None and self.threshold is not None

    @property
    def fully_ready(self) -> bool:
        return self.rf_ready and self.autoencoder_ready

    def status(self) -> dict:
        if self.fully_ready:
            health = "ONLINE"
        elif self.rf_ready:
            health = "DEGRADED"
        else:
            health = "OFFLINE"
        return {
            "health": health,
            "rf_ready": self.rf_ready,
            "autoencoder_ready": self.autoencoder_ready,
            "models_dir": str(self.models_dir) if self.models_dir else None,
            "error": self.load_error,
        }

    # -- inference -------------------------------------------------------
    def predict(self, X: np.ndarray) -> tuple[list[PredictionResult], float]:
        """
        X: shape (n_samples, 49), already in feature_00..feature_48 order,
           NOT yet scaled (this function applies the saved scaler).
        Returns (results, elapsed_ms).
        """
        if not self.rf_ready:
            raise RuntimeError("Random Forest model is not loaded: " + str(self.load_error))

        start = time.perf_counter()

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_FEATURE_COUNT} feature columns, got {X.shape[1]}. "
                "Check the CSV matches feature_00..feature_48 column order."
            )

        X_scaled = self.scaler.transform(X)

        rf_preds = self.rf_model.predict(X_scaled)
        # Confidence = probability assigned to the predicted class
        proba = self.rf_model.predict_proba(X_scaled)
        rf_confidences = proba[np.arange(len(proba)), rf_preds]

        if self.autoencoder_ready:
            reconstructions = self.autoencoder.predict(X_scaled, verbose=0)
            recon_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
            is_anomaly = recon_errors > self.threshold
        else:
            recon_errors = np.zeros(len(X_scaled))
            is_anomaly = np.zeros(len(X_scaled), dtype=bool)

        results = []
        for rf_pred, rf_conf, err, anomaly in zip(rf_preds, rf_confidences, recon_errors, is_anomaly):
            verdict, severity = self._classify(int(rf_pred), bool(anomaly))
            results.append(
                PredictionResult(
                    rf_prediction=int(rf_pred),
                    rf_confidence=float(rf_conf),
                    reconstruction_error=float(err),
                    is_anomaly=bool(anomaly),
                    verdict=verdict,
                    severity=severity,
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        return results, elapsed_ms

    @staticmethod
    def _classify(rf_pred: int, is_anomaly: bool) -> tuple[str, str]:
        if rf_pred == 1 and is_anomaly:
            return "CONFIRMED_ATTACK", "CRITICAL"
        if rf_pred == 1:
            return "KNOWN_ATTACK", "HIGH"
        if is_anomaly:
            return "ZERO_DAY", "MEDIUM"
        return "BENIGN", "LOW"


_engine: Optional[HybridNIDSEngine] = None


def get_engine() -> HybridNIDSEngine:
    """Module-level singleton so models are loaded once per process, not per-request."""
    global _engine
    if _engine is None:
        _engine = HybridNIDSEngine()
    return _engine
