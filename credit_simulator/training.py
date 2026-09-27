from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_config
from .data import PROTECTED, TARGET, load_uci_data, model_features, validate_frame, validate_no_leakage, validate_missingness
from .features import engineer_features, engineered_feature_descriptions
from .fairness import group_metrics
from .registry import register_model


def _metrics(y_true, probabilities, threshold=0.5):
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y_true, probabilities)), "pr_auc": float(average_precision_score(y_true, probabilities)), "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])), "brier_score": float(brier_score_loss(y_true, probabilities)), "accuracy": float(accuracy_score(y_true, predicted)), "precision": float(precision_score(y_true, predicted, zero_division=0)), "recall": float(recall_score(y_true, predicted, zero_division=0)), "f1": float(f1_score(y_true, predicted, zero_division=0)), "specificity": float(tn / max(tn + fp, 1)), "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)), "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def _calibration(y_true, probabilities, bins=10):
    edges = np.linspace(0, 1, bins + 1); rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & ((probabilities < right) if right < 1 else (probabilities <= right))
        if mask.any():
            rows.append({"bin_start": float(left), "bin_end": float(right), "count": int(mask.sum()), "mean_predicted": float(probabilities[mask].mean()), "observed_default_rate": float(np.asarray(y_true)[mask].mean())})
    ece = sum(row["count"] * abs(row["mean_predicted"] - row["observed_default_rate"]) for row in rows) / max(len(y_true), 1)
    return {"expected_calibration_error": float(ece), "maximum_calibration_error": float(max((abs(row["mean_predicted"] - row["observed_default_rate"]) for row in rows), default=0.0)), "deciles": rows}


def _threshold_report(y_true, probabilities, config):
    rows = []; actual = np.asarray(y_true); costs = config["costs"]
    for approve_max in np.arange(0.05, 0.50, 0.05):
        for decline_min in np.arange(max(approve_max + 0.05, 0.10), 0.96, 0.05):
            decisions = np.where(probabilities <= approve_max, "approve", np.where(probabilities >= decline_min, "decline", "manual_review")); approved = decisions == "approve"; reviewed = decisions == "manual_review"; declined = decisions == "decline"
            predicted_default = probabilities >= decline_min; tn, fp, fn, tp = confusion_matrix(actual, predicted_default, labels=[0, 1]).ravel()
            rows.append({"approve_max_risk": round(float(approve_max), 2), "decline_min_risk": round(float(decline_min), 2), "approval_rate": float(approved.mean()), "review_rate": float(reviewed.mean()), "decline_rate": float(declined.mean()), "default_rate_approved": float(actual[approved].mean()) if approved.any() else None, "default_rate_reviewed": float(actual[reviewed].mean()) if reviewed.any() else None, "default_rate_declined": float(actual[declined].mean()) if declined.any() else None, "precision_at_decline_threshold": float(precision_score(actual, predicted_default, zero_division=0)), "recall_at_decline_threshold": float(recall_score(actual, predicted_default, zero_division=0)), "false_positive_rate_at_decline_threshold": float(fp / max(fp + tn, 1)), "false_negative_rate_at_decline_threshold": float(fn / max(fn + tp, 1)), "expected_cost": float((actual[approved] == 1).sum() * costs["approve_default"] + (actual[declined] == 0).sum() * costs["decline_good"] + reviewed.sum() * costs["manual_review"])})
    return rows


def _data_quality(frame, features):
    numeric = frame[features].select_dtypes(include=np.number)
    return {"rows": int(len(frame)), "columns": int(len(frame.columns)), "duplicate_rows": int(frame.duplicated().sum()), "missing_by_column": {str(k): float(v) for k, v in frame.isna().mean().items()}, "constant_columns": [str(c) for c in frame.columns if frame[c].nunique(dropna=False) <= 1], "numeric_summary": json.loads(numeric.describe(percentiles=[0.01, 0.5, 0.99]).transpose().to_json())}


def _plots(frame, y, probabilities, reports):
    reports.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4)); y.value_counts().sort_index().plot(kind="bar"); plt.title("Target class balance"); plt.xlabel("Default"); plt.ylabel("Rows"); plt.tight_layout(); plt.savefig(reports / "target_balance.png", dpi=140); plt.close()
    plt.figure(figsize=(6, 4)); plt.hist(probabilities[np.asarray(y) == 0], bins=30, alpha=0.6, label="non-default"); plt.hist(probabilities[np.asarray(y) == 1], bins=30, alpha=0.6, label="default"); plt.legend(); plt.title("Predicted-risk distributions"); plt.xlabel("Predicted probability"); plt.tight_layout(); plt.savefig(reports / "risk_distribution.png", dpi=140); plt.close()
    calibration = _calibration(y, probabilities)
    points = calibration["deciles"]
    plt.figure(figsize=(5, 5)); plt.plot([0, 1], [0, 1], "--", label="perfect calibration"); plt.plot([p["mean_predicted"] for p in points], [p["observed_default_rate"] for p in points], marker="o", label="model"); plt.xlabel("Mean predicted risk"); plt.ylabel("Observed default rate"); plt.legend(); plt.title("Calibration curve"); plt.tight_layout(); plt.savefig(reports / "calibration_curve.png", dpi=140); plt.close()
    for column in ["LIMIT_BAL", "PAY_0", "current_utilization", "late_payment_count"]:
        if column in frame:
            bins = pd.qcut(frame[column], q=10, duplicates="drop")
            plt.figure(figsize=(6, 4)); frame.assign(_bin=bins).groupby("_bin", observed=True)[TARGET].mean().plot(kind="bar"); plt.title(f"Default rate by {column}"); plt.ylabel("Default rate"); plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.savefig(reports / f"default_rate_{column}.png", dpi=140); plt.close()


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
    started = time.perf_counter()
    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports = Path("outputs")
    report_dir = reports / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    frame = load_uci_data(raw_dir)
    validate_frame(frame)
    validate_missingness(frame, config["thresholds"]["max_missing_fraction"])
    frame = engineer_features(frame)
    features = model_features(frame)
    validate_no_leakage(features)
    X = frame[features]
    y = frame[TARGET].astype(int)
    X_train, X_temp, y_train, y_temp, frame_train, frame_temp = train_test_split(X, y, frame, test_size=config["test_size"] + config["validation_size"], stratify=y, random_state=config["random_state"])
    relative_test = config["test_size"] / (config["test_size"] + config["validation_size"])
    X_valid, X_test, y_valid, y_test, frame_valid, frame_test = train_test_split(X_temp, y_temp, frame_temp, test_size=relative_test, stratify=y_temp, random_state=config["random_state"])
    metrics = {"majority_baseline": _metrics(y_valid, np.full(len(y_valid), float(y_train.mean())))}
    candidates = {}
    for name in ("logistic_regression", "gradient_boosting"):
        model = _pipeline(name, config["random_state"])
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_valid)[:, 1]
        metrics[name] = _metrics(y_valid, probabilities)
        candidates[name] = model
    selected_name = max(candidates, key=lambda name: (metrics[name]["pr_auc"], -metrics[name]["brier_score"]))
    selected = candidates[selected_name]
    test_probabilities = selected.predict_proba(X_test)[:, 1]
    test_metrics = _metrics(y_test, test_probabilities)
    calibration = _calibration(y_test, test_probabilities)
    threshold_rows = _threshold_report(y_test, test_probabilities, config)
    joblib.dump(selected, output / "model.joblib")
    dataset_hash = hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()
    feature_stats = {name: {"mean": float(X_train[name].mean()), "std": float(max(X_train[name].std(), 1e-9))} for name in features if pd.api.types.is_numeric_dtype(X_train[name])}
    metadata = {"model_version": "0.2.0", "selected_model": selected_name, "feature_names": features, "raw_feature_names": model_features(load_uci_data(raw_dir, download=False)), "protected_attributes": PROTECTED, "metrics_validation": metrics, "metrics_test": test_metrics, "thresholds": config["thresholds"], "score": config["score"], "risk_bands": config["risk_bands"], "feature_stats": feature_stats, "training_rows": int(len(X_train)), "validation_rows": int(len(X_valid)), "test_rows": int(len(X_test)), "dataset": {"name": "UCI Default of Credit Card Clients", "source_url": "https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients", "sha256": dataset_hash}, "feature_descriptions": engineered_feature_descriptions(), "training_seconds": time.perf_counter() - started, "disclaimer": "Educational prototype; not for real lending decisions."}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fairness_frame = frame_test.copy()
    fairness_frame["AGE_BIN"] = pd.cut(fairness_frame["AGE"], bins=[0, 25, 35, 50, np.inf], labels=["<=25", "26-35", "36-50", "51+"])
    fairness = _fairness(fairness_frame, y_test.to_numpy(), test_probabilities, ["SEX", "AGE_BIN"])
    fairness_detailed = group_metrics(fairness_frame, y_test.to_numpy(), test_probabilities, ["SEX", "AGE_BIN"], config["thresholds"]["approve_max_risk"], config["thresholds"]["decline_min_risk"])
    report = {"dataset_summary": {"positive_class": int(y.sum()), "negative_class": int((1-y).sum()), "default_rate": float(y.mean()), "train_rows": len(X_train), "validation_rows": len(X_valid), "test_rows": len(X_test), "random_state": config["random_state"], "dataset_sha256": dataset_hash}, "data_quality": _data_quality(frame, features), "feature_engineering": engineered_feature_descriptions(), "candidate_metrics": metrics, "selected_model": selected_name, "test_metrics": test_metrics, "calibration": calibration, "threshold_analysis": threshold_rows, "fairness": fairness, "fairness_detailed": fairness_detailed}
    (reports / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    register_model(metadata, output)
    (report_dir / "training_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(threshold_rows).to_csv(report_dir / "threshold_analysis.csv", index=False)
    (report_dir / "feature_summary.json").write_text(json.dumps(_data_quality(frame, features), indent=2, default=str), encoding="utf-8")
    _plots(frame_test, y_test, test_probabilities, report_dir)
    plt.figure(figsize=(6, 4))
    order = np.argsort(test_probabilities)
    rolling = pd.Series(y_test.to_numpy()[order]).groupby(np.arange(len(y_test)) // max(1, len(y_test) // 10)).mean()
    plt.plot(np.linspace(0.05, 0.95, len(rolling)), rolling, marker="o")
    plt.xlabel("Risk-score decile"); plt.ylabel("Observed default rate"); plt.title("Reliability by score decile"); plt.tight_layout()
    plt.savefig(reports / "reliability.png", dpi=140); plt.close()
    return metadata


if __name__ == "__main__":
    train()
