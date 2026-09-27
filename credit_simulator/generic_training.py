from __future__ import annotations

import json
import hashlib
import platform
import sys
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .datasets import DatasetBundle
from .registry import register_model
from .splitting import temporal_split


def _binary_metrics(y_true, probabilities) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def _feature_summary(frame: pd.DataFrame, features: list[str]) -> dict[str, dict]:
    summary = {}
    for feature in features:
        series = frame[feature]
        values = {"missing_fraction": float(series.isna().mean()), "unique_values": int(series.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            values.update({"type": "numeric", "minimum": float(numeric.min()) if not numeric.empty else None, "maximum": float(numeric.max()) if not numeric.empty else None, "mean": float(numeric.mean()) if not numeric.empty else None, "median": float(numeric.median()) if not numeric.empty else None, "standard_deviation": float(numeric.std(ddof=0)) if not numeric.empty else None, "p01": float(numeric.quantile(0.01)) if not numeric.empty else None, "p99": float(numeric.quantile(0.99)) if not numeric.empty else None})
        else:
            values.update({"type": "categorical", "top_values": {str(key): int(value) for key, value in series.value_counts(dropna=True).head(10).items()}})
        summary[feature] = values
    return summary


def train_tabular(bundle: DatasetBundle, output_dir: str | Path, random_state: int = 42) -> dict:
    """Train a schema-agnostic calibrated model for alternate dataset adapters.

    This produces an experiment artifact for comparison. The production-style
    API remains tied to the UCI credit-card input contract until a versioned
    endpoint schema is added for each alternate dataset.
    """
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    if bundle.target not in bundle.frame.columns:
        raise ValueError(f"Target column is missing: {bundle.target}")
    features = [column for column in bundle.feature_columns if column not in set(bundle.protected_attributes)]
    missing_features = sorted(set(features).difference(bundle.frame.columns))
    if missing_features or not features:
        raise ValueError(f"Alternate dataset feature schema is invalid: {missing_features or 'no model features'}")
    suspicious = [name for name in features if str(name).lower() in {"target", "label", "default"} or str(name).lower() == str(bundle.target).lower()]
    if suspicious:
        raise ValueError(f"Potential target leakage in alternate model features: {suspicious}")
    target = pd.to_numeric(bundle.frame[bundle.target], errors="coerce")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("Alternate dataset target must be binary and non-null")
    X, y = bundle.frame[features], target.astype(int)
    numeric = X.select_dtypes(include=np.number).columns.tolist(); categorical = [column for column in features if column not in numeric]
    preprocess = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric), ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)], remainder="drop")
    base = Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=random_state))])
    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    time_column = bundle.metadata.get("time_column")
    if bundle.split_strategy == "temporal" or time_column:
        if not time_column:
            raise ValueError(f"{bundle.name}: temporal split strategy requires metadata['time_column']")
        splits = temporal_split(bundle.frame, str(time_column), validation_fraction=0.20, test_fraction=0.20)
        train_frame, validation_frame, test_frame = splits.train, splits.validation, splits.test
        X_train, y_train = train_frame[features], train_frame[bundle.target].astype(int)
        X_valid, y_valid = validation_frame[features], validation_frame[bundle.target].astype(int)
        X_test, y_test = test_frame[features], test_frame[bundle.target].astype(int)
        split_strategy = "temporal"
        validation_rows = int(len(validation_frame))
    else:
        X_development, X_test, y_development, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
        X_train, X_valid, y_train, y_valid = train_test_split(X_development, y_development, test_size=0.25, stratify=y_development, random_state=random_state)
        split_strategy = "stratified_random"
        validation_rows = int(len(X_valid))
    started = time.perf_counter(); training_timestamp = datetime.now(timezone.utc).isoformat(); model.fit(X_train, y_train); validation_probabilities = model.predict_proba(X_valid)[:, 1]; probabilities = model.predict_proba(X_test)[:, 1]
    dataset_hash = hashlib.sha256(bundle.frame.to_csv(index=False).encode("utf-8")).hexdigest()
    fingerprint_payload = json.dumps({"dataset_sha256": dataset_hash, "dataset": bundle.name, "target": bundle.target, "features": features, "random_state": random_state}, sort_keys=True).encode("utf-8")
    training_config_sha256 = hashlib.sha256(fingerprint_payload).hexdigest()
    artifact_fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()[:12]
    metrics_validation = _binary_metrics(y_valid, validation_probabilities)
    metrics_test = _binary_metrics(y_test, probabilities)
    data_quality = {"rows": int(len(bundle.frame)), "columns": int(len(bundle.frame.columns)), "missing_by_column": {str(key): float(value) for key, value in bundle.frame.isna().mean().items()}, "feature_dtypes": {str(key): str(value) for key, value in X.dtypes.items()}, "duplicate_rows": int(bundle.frame.duplicated().sum())}
    feature_summary = _feature_summary(bundle.frame, features)
    dataset_summary = {"name": bundle.name, "source_url": bundle.metadata.get("source_url"), "target_name": bundle.target, "positive_class": int(y.sum()), "negative_class": int((1 - y).sum()), "default_rate": float(y.mean()), "train_rows": int(len(X_train)), "validation_rows": validation_rows, "test_rows": int(len(X_test)), "random_state": int(random_state), "split_strategy": split_strategy, "dataset_sha256": dataset_hash}
    runtime = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}
    metadata = {"schema_version": "1.0", "model_version": f"0.2.0+{artifact_fingerprint}", "policy_version": "alternate-experiment-0.1.0", "artifact_fingerprint": artifact_fingerprint, "training_config_sha256": training_config_sha256, "runtime": runtime, "dataset_sha256": dataset_hash, "dataset": bundle.name, "target": bundle.target, "feature_names": features, "protected_attributes": bundle.protected_attributes, "numeric_features": numeric, "categorical_features": categorical, "feature_schema": {str(column): str(bundle.frame[column].dtype) for column in features}, "metrics_validation": metrics_validation, "metrics_test": metrics_test, "dataset_summary": dataset_summary, "data_quality": data_quality, "feature_summary": feature_summary, "training_rows": int(len(X_train)), "validation_rows": validation_rows, "test_rows": int(len(X_test)), "split_strategy": split_strategy, "training_timestamp": training_timestamp, "training_seconds": time.perf_counter() - started, "source_metadata": bundle.metadata, "disclaimer": "Educational experiment; not for real lending decisions."}
    report = {"dataset_summary": dataset_summary, "data_quality": data_quality, "feature_summary": feature_summary, "feature_names": features, "metrics_validation": metrics_validation, "metrics_test": metrics_test, "training_timestamp": training_timestamp, "model_version": metadata["model_version"], "policy_version": metadata["policy_version"], "source_metadata": bundle.metadata}
    joblib.dump(model, output / "model.joblib"); metadata["model_sha256"] = hashlib.sha256((output / "model.joblib").read_bytes()).hexdigest(); (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8"); (output / "training_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8"); version_dir = output / "versions" / artifact_fingerprint; version_dir.mkdir(parents=True, exist_ok=True); shutil.copy2(output / "model.joblib", version_dir / "model.joblib"); shutil.copy2(output / "metadata.json", version_dir / "metadata.json"); register_model(metadata, version_dir, output.parent / "model_registry.json")
    return metadata
