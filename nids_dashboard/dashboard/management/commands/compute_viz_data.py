"""
manage.py compute_viz_data

Generates the confusion matrix and ROC curve data for both the Random
Forest and Hybrid (RF+Autoencoder) models, using the same 20% test split
that train_simple.py held out (same random_state=42 + stratify=y).

Saves results to results/viz_data.json. The dashboard's /api/confusion-matrices/
and /api/roc-curves/ endpoints read from that file — they don't recompute
on every request, so the page loads fast even on large datasets.

Run any time you retrain:
    python manage.py compute_viz_data
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _find(name: str, candidates: list[Path]) -> Path | None:
    return next((p for p in candidates if p.exists()), None)


class Command(BaseCommand):
    help = "Compute confusion matrix + ROC curve data and save to results/viz_data.json"

    def add_arguments(self, parser):
        parser.add_argument("--dataset", type=str, default=None,
                            help="Path to UNSW-NB15_clean.csv (auto-detected if omitted)")
        parser.add_argument("--models-dir", type=str, default=None,
                            help="Path to models/saved/ (auto-detected if omitted)")

    def handle(self, *args, **options):
        from django.conf import settings as dj_settings

        # ── locate files ───────────────────────────────────────────────
        base = Path(dj_settings.BASE_DIR)
        project_root = base.parent  # nids_dashboard/ -> Dissertation-/

        data_candidates = [
            Path(options["dataset"]) if options["dataset"] else None,
            project_root / "data" / "raw" / "UNSW-NB15_clean.csv",
            Path.home() / "Desktop" / "Dissertation-" / "data" / "raw" / "UNSW-NB15_clean.csv",
            Path.home() / "project" / "data" / "raw" / "UNSW-NB15_clean.csv",
        ]
        dataset_path = _find("dataset", [p for p in data_candidates if p])
        if not dataset_path:
            raise CommandError(
                "Cannot find UNSW-NB15_clean.csv. Pass --dataset /path/to/file.csv"
            )

        model_candidates_base = (
            Path(options["models-dir"]) if options["models-dir"] else None
        )
        model_dirs = [
            model_candidates_base,
            project_root / "models" / "saved",
            Path.home() / "Desktop" / "Dissertation-" / "models" / "saved",
            Path.home() / "project" / "models" / "saved",
            Path(getattr(dj_settings, "MODELS_DIR", "/nonexistent")),
        ]
        model_dir = _find("rf_final.pkl", [
            d / "rf_final.pkl" for d in model_dirs if d
        ])
        if model_dir is None:
            raise CommandError(
                "Cannot find rf_final.pkl. Run src/models/train_simple.py first."
            )
        model_dir = model_dir.parent  # strip /rf_final.pkl

        # ── load dataset ───────────────────────────────────────────────
        self.stdout.write(f"Loading dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
        X = df.drop("label", axis=1).values.astype(np.float32)
        y = df["label"].values.astype(np.int32)
        self.stdout.write(f"  {len(df)} rows, {X.shape[1]} features")

        # Same split as train_simple.py
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ── load models ────────────────────────────────────────────────
        self.stdout.write(f"Loading models from: {model_dir}")
        scaler: StandardScaler = joblib.load(model_dir / "scaler_final.pkl")
        rf_model = joblib.load(model_dir / "rf_final.pkl")
        threshold = float(np.load(model_dir / "threshold_final.npy"))

        X_test_scaled = scaler.transform(X_test)

        # ── RF predictions ─────────────────────────────────────────────
        self.stdout.write("Running Random Forest predictions...")
        rf_preds = rf_model.predict(X_test_scaled)
        rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]  # P(malicious)

        # ── Autoencoder ────────────────────────────────────────────────
        ae_errors = np.zeros(len(X_test_scaled))
        try:
            from tensorflow import keras
            ae = keras.models.load_model(model_dir / "autoencoder_final.keras")
            self.stdout.write("Running Autoencoder predictions...")
            recon = ae.predict(X_test_scaled, verbose=0)
            ae_errors = np.mean(np.square(X_test_scaled - recon), axis=1)
            self.stdout.write(f"  Anomaly threshold: {threshold:.4f}")
        except Exception as exc:
            self.stdout.write(self.style.WARNING(
                f"Autoencoder unavailable ({exc}) — hybrid = RF-only for this run"
            ))

        is_anomaly = ae_errors > threshold

        # Hybrid verdict: malicious if RF says 1 OR autoencoder flags it
        hybrid_preds = np.logical_or(rf_preds == 1, is_anomaly).astype(int)

        # Hybrid "probability" for ROC: max of RF proba and normalised AE error
        ae_max = ae_errors.max() or 1.0
        ae_norm = ae_errors / ae_max
        hybrid_score = np.maximum(rf_proba, ae_norm)

        # ── confusion matrices ─────────────────────────────────────────
        def cm_to_dict(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            return {
                "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp),
                "total": int(len(y_true)),
            }

        # ── ROC curves (downsample to ≤200 points for lean JSON) ───────
        def roc_to_dict(y_true, scores, n_points=200):
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = float(roc_auc_score(y_true, scores))
            if len(fpr) > n_points:
                idx = np.round(np.linspace(0, len(fpr) - 1, n_points)).astype(int)
                fpr, tpr = fpr[idx], tpr[idx]
            return {
                "fpr": [round(float(v), 4) for v in fpr],
                "tpr": [round(float(v), 4) for v in tpr],
                "auc": round(auc, 4),
            }

        self.stdout.write("Computing metrics...")
        viz_data = {
            "confusion_matrices": {
                "random_forest": cm_to_dict(y_test, rf_preds),
                "hybrid": cm_to_dict(y_test, hybrid_preds),
            },
            "roc_curves": {
                "random_forest": roc_to_dict(y_test, rf_proba),
                "hybrid": roc_to_dict(y_test, hybrid_score),
            },
            "test_set_size": int(len(y_test)),
            "benign_count": int(np.sum(y_test == 0)),
            "malicious_count": int(np.sum(y_test == 1)),
        }

        # ── save ───────────────────────────────────────────────────────
        out_dir = project_root / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "viz_data.json"
        out_path.write_text(json.dumps(viz_data, indent=2))

        rf_cm = viz_data["confusion_matrices"]["random_forest"]
        h_cm  = viz_data["confusion_matrices"]["hybrid"]
        rf_roc = viz_data["roc_curves"]["random_forest"]
        h_roc  = viz_data["roc_curves"]["hybrid"]

        self.stdout.write(self.style.SUCCESS(f"\nSaved to {out_path}"))
        self.stdout.write(
            f"\nRandom Forest  — TP:{rf_cm['tp']} TN:{rf_cm['tn']} "
            f"FP:{rf_cm['fp']} FN:{rf_cm['fn']}  AUC:{rf_roc['auc']}"
        )
        self.stdout.write(
            f"Hybrid Model   — TP:{h_cm['tp']} TN:{h_cm['tn']} "
            f"FP:{h_cm['fp']} FN:{h_cm['fn']}  AUC:{h_roc['auc']}"
        )
