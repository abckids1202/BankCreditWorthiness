from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
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
    features = [column for column in bundle.feature_columns if column not in set(bundle.protected_attributes)]
    X, y = bundle.frame[features], bundle.frame[bundle.target].astype(int)
    numeric = X.select_dtypes(include=np.number).columns.tolist(); categorical = [column for column in features if column not in numeric]
    preprocess = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric), ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)], remainder="drop")
    base = Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=random_state))])
    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    started = time.perf_counter(); model.fit(X_train, y_train); probabilities = model.predict_proba(X_test)[:, 1]
    metadata = {"model_version": "0.1.0", "dataset": bundle.name, "target": bundle.target, "feature_names": features, "protected_attributes": bundle.protected_attributes, "numeric_features": numeric, "categorical_features": categorical, "metrics_test": {"roc_auc": float(roc_auc_score(y_test, probabilities)), "pr_auc": float(average_precision_score(y_test, probabilities)), "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])), "brier_score": float(brier_score_loss(y_test, probabilities))}, "training_rows": int(len(X_train)), "test_rows": int(len(X_test)), "training_seconds": time.perf_counter() - started, "source_metadata": bundle.metadata, "disclaimer": "Educational experiment; not for real lending decisions."}
    joblib.dump(model, output / "model.joblib"); (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8"); register_model(metadata, output, output.parent / "model_registry.json")
    return metadata
