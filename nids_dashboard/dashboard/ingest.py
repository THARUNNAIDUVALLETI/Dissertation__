"""
ingest.py — turns uploaded CSV/parquet files into Detection + Alert rows.
Accepts raw UNSW-NB15 files (45 cols) and auto-cleans them, or pre-cleaned
files (42 cols). No file size limits enforced here.
"""
from __future__ import annotations

import ipaddress
import random
import time
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.utils import timezone

from .ml_engine import get_engine, EXPECTED_FEATURE_COUNT
from .models import Alert, Detection, UploadBatch

_INTERNAL_SUBNETS = ["10.0.1.", "10.0.2.", "192.168.10.", "172.16.5."]
_DROP_COLS = {"id", "attack_cat"}
_CATEGORICAL_COLS = ["proto", "service", "state"]


def _random_external_ip() -> str:
    while True:
        ip = ipaddress.IPv4Address(random.randint(0, 2**32 - 1))
        if ip.is_global and not ip.is_multicast:
            return str(ip)


def _random_internal_ip() -> str:
    return f"{random.choice(_INTERNAL_SUBNETS)}{random.randint(2, 254)}"


def _find_encoders() -> dict:
    """Try to load the categorical encoders saved by build_dataset.py."""
    candidates = []
    try:
        from django.conf import settings
        d = getattr(settings, "MODELS_DIR", None)
        if d:
            candidates.append(Path(d) / "categorical_encoders.pkl")
    except Exception:
        pass
    candidates += [
        Path.home() / "Desktop" / "Dissertation-" / "models" / "saved" / "categorical_encoders.pkl",
        Path.home() / "project" / "models" / "saved" / "categorical_encoders.pkl",
        Path("/home/kali/Desktop/Dissertation-/models/saved/categorical_encoders.pkl"),
        Path("/home/kali/project/models/saved/categorical_encoders.pkl"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return joblib.load(p)
            except Exception:
                pass
    return {}


def _looks_like_index_col(series: pd.Series) -> bool:
    """True if this column looks like a pandas-injected row index (0,1,2,...)."""
    try:
        vals = pd.to_numeric(series, errors="coerce")
        if vals.isna().any():
            return False
        arr = vals.astype(int).values
        return bool(
            (arr == list(range(len(arr)))).all() or
            (arr == list(range(1, len(arr) + 1))).all()
        )
    except Exception:
        return False


def _auto_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Makes any UNSW-NB15-derived file model-ready:
      1. Drops known non-feature columns (id, attack_cat)
      2. Drops phantom pandas index columns (Unnamed: 0 or bare 0,1,2 sequences)
      3. Label-encodes categorical columns (proto, service, state)
    """
    from sklearn.preprocessing import LabelEncoder

    # 1. Drop known non-feature columns
    drop = [c for c in df.columns if c.lower() in _DROP_COLS]
    if drop:
        df = df.drop(columns=drop)

    # 2. Drop phantom pandas index columns.
    #    Catches both:
    #      - "Unnamed: 0" (default name when index=True was used)
    #      - Any column whose values are exactly 0,1,2,...,n-1 or 1,2,...,n
    index_cols = []
    for col in df.columns:
        if col.lower() in ("unnamed: 0", "unnamed:0", "index", "row"):
            index_cols.append(col)
        elif _looks_like_index_col(df[col]):
            index_cols.append(col)
    if index_cols:
        df = df.drop(columns=index_cols)

    saved_encoders = _find_encoders()

    for col in _CATEGORICAL_COLS:
        if col in df.columns and (df[col].dtype == object or str(df[col].dtype) in ('string', 'str', 'large_string') or hasattr(df[col].dtype, 'pyarrow_dtype')):
            if col in saved_encoders:
                le = saved_encoders[col]
                df[col] = (
                    df[col].astype(str)
                    .map(lambda v, le=le: int(le.transform([v])[0])
                         if v in le.classes_ else 0)
                    .astype(np.int32)
                )
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

    return df


class CSVValidationError(ValueError):
    pass


def validate_csv(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Auto-cleans then validates. Returns (X, y_or_None)."""
    df = _auto_clean(df)

    columns = list(df.columns)
    has_label = "label" in columns
    feature_cols = [c for c in columns if c != "label"]

    if len(feature_cols) != EXPECTED_FEATURE_COUNT:
        raise CSVValidationError(
            f"After cleaning, found {len(feature_cols)} feature columns but "
            f"the model expects {EXPECTED_FEATURE_COUNT}. "
            f"Upload either a pre-cleaned UNSW-NB15_clean.csv slice or a raw "
            f"UNSW_NB15 training/testing parquet file."
        )

    try:
        X = df[feature_cols].astype(np.float32).values
    except (ValueError, TypeError) as exc:
        raise CSVValidationError(f"Non-numeric value in feature columns: {exc}")

    y = None
    if has_label:
        try:
            y = df["label"].astype(np.int32).values
        except (ValueError, TypeError):
            y = None

    return X, y


def process_upload(batch: UploadBatch, df: pd.DataFrame) -> None:
    batch.status = UploadBatch.STATUS_PROCESSING
    batch.save(update_fields=["status"])

    try:
        X, y = validate_csv(df)
    except CSVValidationError as exc:
        batch.status = UploadBatch.STATUS_FAILED
        batch.error_message = str(exc)
        batch.save(update_fields=["status", "error_message"])
        return

    engine = get_engine()
    if not engine.rf_ready:
        batch.status = UploadBatch.STATUS_FAILED
        batch.error_message = "ML models not loaded: " + str(engine.load_error)
        batch.save(update_fields=["status", "error_message"])
        return

    start = time.perf_counter()
    results, _ = engine.predict(X)
    elapsed_ms = (time.perf_counter() - start) * 1000

    now = timezone.now()
    counts = {"BENIGN": 0, "KNOWN_ATTACK": 0, "ZERO_DAY": 0, "CONFIRMED_ATTACK": 0}

    detections = []
    for i, result in enumerate(results):
        true_label = int(y[i]) if y is not None else None
        row_timestamp = now - timedelta(minutes=random.uniform(0, 24 * 60))
        detections.append(
            Detection(
                batch=batch,
                row_index=i,
                timestamp=row_timestamp,
                source_ip=_random_external_ip(),
                destination_ip=_random_internal_ip(),
                source_port=random.randint(1024, 65535),
                destination_port=random.choice([80, 443, 22, 3389, 53, 8080, 21, 25]),
                rf_prediction=result.rf_prediction,
                rf_confidence=result.rf_confidence,
                reconstruction_error=result.reconstruction_error,
                anomaly_threshold=engine.threshold or 0.0,
                is_anomaly=result.is_anomaly,
                true_label=true_label,
                verdict=result.verdict,
                severity=result.severity,
            )
        )
        counts[result.verdict] += 1

    created = Detection.objects.bulk_create(detections, batch_size=1000)

    alerts = [
        Alert(detection=d)
        for d in created
        if d.verdict != Detection.VERDICT_BENIGN
    ]
    Alert.objects.bulk_create(alerts, batch_size=1000)

    batch.row_count = len(created)
    batch.benign_count = counts["BENIGN"]
    batch.known_attack_count = counts["KNOWN_ATTACK"]
    batch.zero_day_count = counts["ZERO_DAY"]
    batch.confirmed_attack_count = counts["CONFIRMED_ATTACK"]
    batch.processing_ms = elapsed_ms
    batch.status = UploadBatch.STATUS_DONE
    batch.save()
