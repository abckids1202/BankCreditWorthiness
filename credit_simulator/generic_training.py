from __future__ import annotations

import json
import hashlib
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    started = time.perf_counter(); model.fit(X_train, y_train); probabilities = model.predict_proba(X_test)[:, 1]
    dataset_hash = hashlib.sha256(bundle.frame.to_csv(index=False).encode("utf-8")).hexdigest()
    fingerprint_payload = json.dumps({"dataset_sha256": dataset_hash, "dataset": bundle.name, "target": bundle.target, "features": features, "random_state": random_state}, sort_keys=True).encode("utf-8")
    artifact_fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()[:12]
    metrics_test = {"roc_auc": float(roc_auc_score(y_test, probabilities)), "pr_auc": float(average_precision_score(y_test, probabilities)), "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])), "brier_score": float(brier_score_loss(y_test, probabilities))}
    data_quality = {"rows": int(len(bundle.frame)), "columns": int(len(bundle.frame.columns)), "missing_by_column": {str(key): float(value) for key, value in bundle.frame.isna().mean().items()}, "feature_dtypes": {str(key): str(value) for key, value in X.dtypes.items()}, "duplicate_rows": int(bundle.frame.duplicated().sum())}
    dataset_summary = {"positive_class": int(y.sum()), "negative_class": int((1 - y).sum()), "default_rate": float(y.mean()), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "random_state": int(random_state), "dataset_sha256": dataset_hash}
    metadata = {"model_version": f"0.2.0+{artifact_fingerprint}", "policy_version": "alternate-experiment-0.1.0", "artifact_fingerprint": artifact_fingerprint, "dataset_sha256": dataset_hash, "dataset": bundle.name, "target": bundle.target, "feature_names": features, "protected_attributes": bundle.protected_attributes, "numeric_features": numeric, "categorical_features": categorical, "metrics_test": metrics_test, "dataset_summary": dataset_summary, "data_quality": data_quality, "training_rows": int(len(X_train)), "test_rows": int(len(X_test)), "training_seconds": time.perf_counter() - started, "source_metadata": bundle.metadata, "disclaimer": "Educational experiment; not for real lending decisions."}
    joblib.dump(model, output / "model.joblib"); metadata["model_sha256"] = hashlib.sha256((output / "model.joblib").read_bytes()).hexdigest(); (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8"); (output / "training_report.json").write_text(json.dumps({"dataset_summary": dataset_summary, "data_quality": data_quality, "feature_names": features, "metrics_test": metrics_test, "model_version": metadata["model_version"], "policy_version": metadata["policy_version"]}, indent=2, default=str), encoding="utf-8"); version_dir = output / "versions" / artifact_fingerprint; version_dir.mkdir(parents=True, exist_ok=True); shutil.copy2(output / "model.joblib", version_dir / "model.joblib"); shutil.copy2(output / "metadata.json", version_dir / "metadata.json"); register_model(metadata, version_dir, output.parent / "model_registry.json")
    return metadata
