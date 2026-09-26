from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_config
from .data import PROTECTED, TARGET, load_uci_data, model_features, validate_frame, validate_no_leakage, validate_missingness


def _pipeline(kind: str, random_state: int):
    if kind == "logistic_regression":
        base = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=random_state)
        estimator = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", estimator)])
    base = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, random_state=random_state)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3))])


def _fairness(frame: pd.DataFrame, y_true: np.ndarray, probabilities: np.ndarray, protected: list[str]) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    result = {}
    for column in protected:
        if column not in frame:
            continue
        groups = frame[column].astype(str)
        result[column] = {}
        for group in sorted(groups.unique()):
            mask = groups.to_numpy() == group
            if mask.sum() == 0:
                continue
            result[column][group] = {
                "count": int(mask.sum()),
                "predicted_default_rate": float(predictions[mask].mean()),
                "actual_default_rate": float(y_true[mask].mean()),
                "approval_rate_at_0_5": float((1 - predictions[mask]).mean()),
            }
    return result


def train(output_dir: str | Path = "artifacts", raw_dir: str | Path = "data/raw", config_path: str | Path | None = None) -> dict:
    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports = Path("outputs")
    reports.mkdir(parents=True, exist_ok=True)
    frame = load_uci_data(raw_dir)
    validate_frame(frame)
    validate_missingness(frame, config["thresholds"]["max_missing_fraction"])
    features = model_features(frame)
    validate_no_leakage(features)
    X = frame[features]
    y = frame[TARGET].astype(int)
    X_train, X_temp, y_train, y_temp, frame_train, frame_temp = train_test_split(X, y, frame, test_size=config["test_size"] + config["validation_size"], stratify=y, random_state=config["random_state"])
    relative_test = config["test_size"] / (config["test_size"] + config["validation_size"])
    X_valid, X_test, y_valid, y_test, frame_valid, frame_test = train_test_split(X_temp, y_temp, frame_temp, test_size=relative_test, stratify=y_temp, random_state=config["random_state"])
    metrics = {}
    candidates = {}
    for name in ("logistic_regression", "gradient_boosting"):
        model = _pipeline(name, config["random_state"])
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_valid)[:, 1]
        metrics[name] = {"roc_auc": float(roc_auc_score(y_valid, probabilities)), "pr_auc": float(average_precision_score(y_valid, probabilities)), "brier_score": float(brier_score_loss(y_valid, probabilities))}
        candidates[name] = model
    selected_name = max(metrics, key=lambda name: (metrics[name]["pr_auc"], -metrics[name]["brier_score"]))
    selected = candidates[selected_name]
    test_probabilities = selected.predict_proba(X_test)[:, 1]
    test_metrics = {"roc_auc": float(roc_auc_score(y_test, test_probabilities)), "pr_auc": float(average_precision_score(y_test, test_probabilities)), "brier_score": float(brier_score_loss(y_test, test_probabilities))}
    joblib.dump(selected, output / "model.joblib")
    metadata = {"model_version": "0.1.0", "selected_model": selected_name, "feature_names": features, "protected_attributes": PROTECTED, "metrics_validation": metrics, "metrics_test": test_metrics, "thresholds": config["thresholds"], "training_rows": int(len(X_train)), "test_rows": int(len(X_test)), "disclaimer": "Educational prototype; not for real lending decisions."}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fairness_frame = frame_test.copy()
    fairness_frame["AGE_BIN"] = pd.cut(fairness_frame["AGE"], bins=[0, 25, 35, 50, np.inf], labels=["<=25", "26-35", "36-50", "51+"])
    fairness = _fairness(fairness_frame, y_test.to_numpy(), test_probabilities, ["SEX", "AGE_BIN"])
    report = {"candidate_metrics": metrics, "selected_model": selected_name, "test_metrics": test_metrics, "fairness": fairness}
    (reports / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    plt.figure(figsize=(6, 4))
    order = np.argsort(test_probabilities)
    rolling = pd.Series(y_test.to_numpy()[order]).groupby(np.arange(len(y_test)) // max(1, len(y_test) // 10)).mean()
    plt.plot(np.linspace(0.05, 0.95, len(rolling)), rolling, marker="o")
    plt.xlabel("Risk-score decile"); plt.ylabel("Observed default rate"); plt.title("Reliability by score decile"); plt.tight_layout()
    plt.savefig(reports / "reliability.png", dpi=140); plt.close()
    return metadata


if __name__ == "__main__":
    train()
