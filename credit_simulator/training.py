from __future__ import annotations

import json
import hashlib
import platform
import re
import sys
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
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
from .fairness import group_metrics, threshold_sensitivity
from .registry import register_model


def _metrics(y_true, probabilities, threshold=0.5):
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y_true, probabilities)), "pr_auc": float(average_precision_score(y_true, probabilities)), "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])), "brier_score": float(brier_score_loss(y_true, probabilities)), "accuracy": float(accuracy_score(y_true, predicted)), "precision": float(precision_score(y_true, predicted, zero_division=0)), "recall": float(recall_score(y_true, predicted, zero_division=0)), "f1": float(f1_score(y_true, predicted, zero_division=0)), "specificity": float(tn / max(tn + fp, 1)), "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)), "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def _bootstrap_intervals(y_true, probabilities, random_state=42, n_bootstrap=200):
    actual = np.asarray(y_true).astype(int)
    scores = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(random_state)
    samples = {"roc_auc": [], "pr_auc": [], "brier_score": []}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(actual), len(actual))
        sampled_y, sampled_scores = actual[indices], scores[indices]
        if np.unique(sampled_y).size < 2:
            continue
        samples["roc_auc"].append(roc_auc_score(sampled_y, sampled_scores))
        samples["pr_auc"].append(average_precision_score(sampled_y, sampled_scores))
        samples["brier_score"].append(brier_score_loss(sampled_y, sampled_scores))
    intervals = {}
    for metric, values in samples.items():
        if not values:
            intervals[metric] = {"estimate": None, "lower_95": None, "upper_95": None}
        else:
            intervals[metric] = {"estimate": float(np.mean(values)), "lower_95": float(np.percentile(values, 2.5)), "upper_95": float(np.percentile(values, 97.5))}
    return {"n_bootstrap": int(n_bootstrap), "successful_samples": int(len(samples["roc_auc"])), "metrics": intervals}


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


def _approval_rate_report(y_true, probabilities, target_rates=(0.50, 0.70, 0.80, 0.90)):
    """Summarize binary accept/reject performance at target approval rates.

    The cutoff approves the lowest-risk applicants. ``default_recall`` measures
    the fraction of observed defaults that were not approved; it is not a
    guarantee about future applicants or a production underwriting objective.
    """
    actual = np.asarray(y_true).astype(int)
    scores = np.asarray(probabilities, dtype=float)
    rows = []
    for target_rate in target_rates:
        cutoff = float(np.quantile(scores, target_rate))
        approved = scores <= cutoff
        flagged = ~approved
        rows.append({
            "target_approval_rate": float(target_rate),
            "risk_cutoff": cutoff,
            "achieved_approval_rate": float(approved.mean()),
            "default_rate_approved": float(actual[approved].mean()) if approved.any() else None,
            "approved_count": int(approved.sum()),
            "not_approved_count": int(flagged.sum()),
            "default_precision_not_approved": float(precision_score(actual, flagged, zero_division=0)),
            "default_recall_not_approved": float(recall_score(actual, flagged, zero_division=0)),
        })
    return rows


def _global_feature_importance(model, X_test, y_test, features, descriptions, random_state):
    result = permutation_importance(model, X_test[features], y_test, scoring="roc_auc", n_repeats=5, random_state=random_state, n_jobs=1)
    rows = []
    for index, feature in enumerate(features):
        rows.append({"feature": feature, "description": descriptions.get(feature, feature), "importance_mean": float(result.importances_mean[index]), "importance_std": float(result.importances_std[index])})
    return sorted(rows, key=lambda row: row["importance_mean"], reverse=True)


def _data_quality(frame, features):
    numeric = frame[features].select_dtypes(include=np.number)
    numeric_summary = json.loads(numeric.describe(percentiles=[0.01, 0.5, 0.99]).transpose().to_json())
    outlier_counts = {}
    for column in numeric.columns:
        values = numeric[column].dropna()
        if values.empty:
            outlier_counts[column] = 0
            continue
        lower, upper = values.quantile(0.25), values.quantile(0.75)
        spread = upper - lower
        outlier_counts[column] = int(((values < lower - 1.5 * spread) | (values > upper + 1.5 * spread)).sum())
    categorical_summary = {}
    for column in frame.columns:
        if column not in numeric.columns and column != TARGET:
            counts = frame[column].astype("string").fillna("<missing>").value_counts(dropna=False)
            categorical_summary[column] = {"unique_values": int(frame[column].nunique(dropna=True)), "missing_fraction": float(frame[column].isna().mean()), "rare_category_count": int((counts < max(5, len(frame) * 0.01)).sum()), "top_values": {str(key): int(value) for key, value in counts.head(10).items()}}
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_id_rows": int(frame["ID"].duplicated().sum()) if "ID" in frame else None,
        "missing_by_column": {str(k): float(v) for k, v in frame.isna().mean().items()},
        "constant_columns": [str(c) for c in frame.columns if frame[c].nunique(dropna=False) <= 1],
        "target_values": sorted({int(value) for value in frame[TARGET].dropna().unique()}) if TARGET in frame else [],
        "target_outside_binary_count": int((~frame[TARGET].isin([0, 1])).sum()) if TARGET in frame else None,
        "numeric_summary": numeric_summary,
        "numeric_outlier_counts_iqr": outlier_counts,
        "categorical_summary": categorical_summary,
    }


def _plots(frame, y, probabilities, reports, threshold_rows=None, features=None, config=None):
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
    if features:
        numeric_features = [name for name in features if name in frame and pd.api.types.is_numeric_dtype(frame[name])]
        if numeric_features:
            overview_features = numeric_features[:12]
            columns = 3; rows = int(np.ceil(len(overview_features) / columns))
            figure, axes = plt.subplots(rows, columns, figsize=(12, 3.0 * rows)); axes = np.atleast_1d(axes).ravel()
            for axis, name in zip(axes, overview_features):
                axis.hist(frame[name].dropna(), bins=20, color="#4472c4", alpha=0.85)
                axis.set_title(name); axis.set_ylabel("Rows")
            for axis in axes[len(overview_features):]: axis.axis("off")
            figure.suptitle("Training-test feature distributions (overview)", y=1.01); figure.tight_layout(); figure.savefig(reports / "feature_distributions.png", dpi=140, bbox_inches="tight"); plt.close(figure)
            for name in numeric_features:
                figure, axis = plt.subplots(figsize=(6, 4))
                axis.hist(frame[name].dropna(), bins=30, color="#4472c4", alpha=0.85)
                axis.set(title=f"Distribution: {name}", xlabel=name, ylabel="Rows")
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "feature"
                figure.tight_layout(); figure.savefig(reports / f"feature_distribution_{safe_name}.png", dpi=140); plt.close(figure)
    if config is not None:
        decline_threshold = float(config["thresholds"]["decline_min_risk"])
        predicted_default = np.asarray(probabilities) >= decline_threshold
        matrix = confusion_matrix(np.asarray(y), predicted_default, labels=[0, 1])
        figure, axis = plt.subplots(figsize=(5, 4)); image = axis.imshow(matrix, cmap="Blues")
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-default", "Default"], yticklabels=["Non-default", "Default"], xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix at {decline_threshold:.2f} risk")
        for row in range(2):
            for column in range(2): axis.text(column, row, int(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > matrix.max() / 2 else "black")
        figure.colorbar(image, ax=axis); figure.tight_layout(); figure.savefig(reports / "confusion_matrix.png", dpi=140); plt.close(figure)
    if threshold_rows:
        table = pd.DataFrame(threshold_rows)
        if config is not None:
            target_approve = float(config["thresholds"]["approve_max_risk"])
            table = table.iloc[(table["approve_max_risk"] - target_approve).abs().argsort()[:max(1, len(table) // 20)]]
        table = table.sort_values("decline_min_risk")
        figure, axis = plt.subplots(figsize=(7, 4));
        for column, label in [("approval_rate", "Approve"), ("review_rate", "Manual review"), ("decline_rate", "Decline")]: axis.plot(table["decline_min_risk"], table[column], marker="o", label=label)
        axis.set(xlabel="Decline threshold", ylabel="Population rate", title="Decision population by threshold"); axis.set_ylim(0, 1); axis.legend(); figure.tight_layout(); figure.savefig(reports / "threshold_comparison.png", dpi=140); plt.close(figure)


def _fairness_threshold_plot(sensitivity: list[dict], reports: Path) -> None:
    """Plot approval-rate disparity across the tested uniform policy bands."""
    if not sensitivity:
        return
    labels = [f"{row['approve_max_risk']:.2f}/{row['decline_min_risk']:.2f}" for row in sensitivity]
    figure, axis = plt.subplots(figsize=(8, 4))
    plotted = False
    columns = sorted({key.removesuffix("_comparisons") for row in sensitivity for key in row.get("metrics", {}) if key.endswith("_comparisons")})
    for column in columns:
        comparison_groups = sorted({group for row in sensitivity for group in row.get("metrics", {}).get(f"{column}_comparisons", {})})
        for group in comparison_groups:
            values = [row["metrics"].get(f"{column}_comparisons", {}).get(group, {}).get("demographic_parity_difference") for row in sensitivity]
            if any(value is not None for value in values):
                axis.plot(labels, values, marker="o", label=f"{column}: {group}")
                plotted = True
    if not plotted:
        plt.close(figure)
        return
    axis.axhline(0, color="black", linewidth=0.8); axis.set(xlabel="Approve / decline thresholds", ylabel="Approval-rate difference vs reference", title="Fairness threshold sensitivity"); axis.legend(fontsize=8); figure.tight_layout(); figure.savefig(reports / "fairness_threshold_sensitivity.png", dpi=140); plt.close(figure)


def _pipeline(kind: str, random_state: int, calibrated: bool = True):
    if kind == "logistic_regression":
        base = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=random_state)
        estimator = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3) if calibrated else base
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", estimator)])
    base = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, random_state=random_state)
    estimator = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3) if calibrated else base
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def _candidate_specs() -> list[dict[str, object]]:
    """Return the model families reported by every primary training run."""
    return [
        {"name": "logistic_regression_uncalibrated", "family": "logistic_regression", "calibrated": False},
        {"name": "logistic_regression_calibrated", "family": "logistic_regression", "calibrated": True},
        {"name": "gradient_boosting_uncalibrated", "family": "gradient_boosting", "calibrated": False},
        {"name": "gradient_boosting_calibrated", "family": "gradient_boosting", "calibrated": True},
    ]


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
    config_sha256 = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    runtime = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}
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
    baseline_probabilities = np.full(len(y_valid), float(y_train.mean()))
    metrics = {"majority_baseline": {**_metrics(y_valid, baseline_probabilities), "model_family": "majority_baseline", "calibrated": False, "training_seconds": 0.0, "inference_seconds": 0.0}}
    candidates = {}
    for specification in _candidate_specs():
        name = str(specification["name"])
        family = str(specification["family"])
        calibrated = bool(specification["calibrated"])
        model = _pipeline(family, config["random_state"], calibrated=calibrated)
        fit_started = time.perf_counter()
        model.fit(X_train, y_train)
        training_seconds = time.perf_counter() - fit_started
        inference_started = time.perf_counter()
        probabilities = model.predict_proba(X_valid)[:, 1]
        inference_seconds = time.perf_counter() - inference_started
        metrics[name] = {**_metrics(y_valid, probabilities), "model_family": family, "calibrated": calibrated, "training_seconds": float(training_seconds), "inference_seconds": float(inference_seconds), "inference_seconds_per_row": float(inference_seconds / max(len(X_valid), 1))}
        candidates[name] = model
    calibrated_candidates = [name for name in candidates if metrics[name]["calibrated"]]
    selected_name = max(calibrated_candidates, key=lambda name: (metrics[name]["pr_auc"], -metrics[name]["brier_score"]))
    selected = candidates[selected_name]
    test_probabilities = selected.predict_proba(X_test)[:, 1]
    test_metrics = _metrics(y_test, test_probabilities)
    bootstrap = _bootstrap_intervals(y_test, test_probabilities, config["random_state"])
    calibration = _calibration(y_test, test_probabilities)
    threshold_rows = _threshold_report(y_test, test_probabilities, config)
    approval_rate_rows = _approval_rate_report(y_test, test_probabilities)
    feature_descriptions = engineered_feature_descriptions()
    global_importance = _global_feature_importance(selected, X_test, y_test, features, feature_descriptions, config["random_state"])
    dataset_hash = hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()
    training_timestamp = datetime.now(timezone.utc).isoformat()
    split_overlap = {"train_validation": 0, "train_test": 0, "validation_test": 0}
    if "ID" in frame:
        train_ids, valid_ids, test_ids = (set(part["ID"].dropna().tolist()) for part in (frame_train, frame_valid, frame_test))
        split_overlap = {"train_validation": len(train_ids & valid_ids), "train_test": len(train_ids & test_ids), "validation_test": len(valid_ids & test_ids)}
    version_payload = json.dumps({"dataset_sha256": dataset_hash, "config": config, "selected_model": selected_name}, sort_keys=True, default=str).encode("utf-8")
    artifact_fingerprint = hashlib.sha256(version_payload).hexdigest()[:12]
    experiment_id = f"exp-{config_sha256[:16]}"
    model_version = f"0.3.0+{artifact_fingerprint}"
    joblib.dump(selected, output / "model.joblib")
    model_sha256 = hashlib.sha256((output / "model.joblib").read_bytes()).hexdigest()
    feature_stats = {name: {"mean": float(X_train[name].mean()), "std": float(max(X_train[name].std(), 1e-9))} for name in features if pd.api.types.is_numeric_dtype(X_train[name])}
    metadata = {"schema_version": "1.0", "experiment_id": experiment_id, "model_version": model_version, "policy_version": config["policy_version"], "artifact_fingerprint": artifact_fingerprint, "training_config_sha256": config_sha256, "runtime": runtime, "model_sha256": model_sha256, "selected_model": selected_name, "training_config": config, "feature_names": features, "raw_feature_names": model_features(load_uci_data(raw_dir, download=False)), "protected_attributes": PROTECTED, "metrics_validation": metrics, "metrics_test": test_metrics, "test_metric_bootstrap": bootstrap, "thresholds": config["thresholds"], "score": config["score"], "risk_bands": config["risk_bands"], "feature_stats": feature_stats, "training_rows": int(len(X_train)), "validation_rows": int(len(X_valid)), "test_rows": int(len(X_test)), "dataset": {"name": "UCI Default of Credit Card Clients", "source_url": "https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients", "license": "UCI Machine Learning Repository dataset terms; cite Yeh and Lien (2009)", "sha256": dataset_hash}, "feature_descriptions": feature_descriptions, "global_feature_importance": global_importance, "training_timestamp": training_timestamp, "training_seconds": time.perf_counter() - started, "disclaimer": "Educational prototype; not for real lending decisions."}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    version_dir = output / "versions" / artifact_fingerprint
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output / "model.joblib", version_dir / "model.joblib")
    shutil.copy2(output / "metadata.json", version_dir / "metadata.json")
    fairness_frame = frame_test.copy()
    fairness_frame["AGE_BIN"] = pd.cut(fairness_frame["AGE"], bins=[0, 25, 35, 50, np.inf], labels=["<=25", "26-35", "36-50", "51+"])
    fairness = _fairness(fairness_frame, y_test.to_numpy(), test_probabilities, ["SEX", "AGE_BIN"])
    fairness_detailed = group_metrics(fairness_frame, y_test.to_numpy(), test_probabilities, ["SEX", "AGE_BIN"], config["thresholds"]["approve_max_risk"], config["thresholds"]["decline_min_risk"])
    fairness_sensitivity = threshold_sensitivity(fairness_frame, y_test.to_numpy(), test_probabilities, ["SEX", "AGE_BIN"], [(0.10, 0.30), (0.20, 0.45), (0.30, 0.60)])
    report = {"experiment_id": experiment_id, "dataset_summary": {"name": "UCI Default of Credit Card Clients", "source_url": "https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients", "license": "UCI Machine Learning Repository dataset terms; cite Yeh and Lien (2009)", "rows": int(len(frame)), "feature_count": int(len(features)), "target_name": TARGET, "positive_class": int(y.sum()), "negative_class": int((1-y).sum()), "default_rate": float(y.mean()), "train_rows": len(X_train), "validation_rows": len(X_valid), "test_rows": len(X_test), "random_state": config["random_state"], "training_timestamp": training_timestamp, "dataset_sha256": dataset_hash, "split_overlap": split_overlap, "class_imbalance_ratio": float((y == 0).sum() / max((y == 1).sum(), 1))}, "training_config": config, "training_config_sha256": config_sha256, "runtime": runtime, "data_quality": _data_quality(frame, features), "feature_engineering": feature_descriptions, "candidate_metrics": metrics, "selected_model": selected_name, "test_metrics": test_metrics, "test_metric_bootstrap": bootstrap, "calibration": calibration, "threshold_analysis": threshold_rows, "approval_rate_analysis": approval_rate_rows, "global_feature_importance": global_importance, "fairness": fairness, "fairness_detailed": fairness_detailed, "fairness_threshold_sensitivity": fairness_sensitivity}
    (reports / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    register_model(metadata, version_dir)
    (report_dir / "training_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(threshold_rows).to_csv(report_dir / "threshold_analysis.csv", index=False)
    pd.DataFrame(approval_rate_rows).to_csv(report_dir / "approval_rate_analysis.csv", index=False)
    comparison_rows = [{"model": name, **values} for name, values in metrics.items()]
    pd.DataFrame(comparison_rows).to_csv(report_dir / "model_comparison.csv", index=False)
    pd.DataFrame(global_importance).to_csv(report_dir / "global_feature_importance.csv", index=False)
    (report_dir / "feature_summary.json").write_text(json.dumps(_data_quality(frame, features), indent=2, default=str), encoding="utf-8")
    (report_dir / "fairness_threshold_sensitivity.json").write_text(json.dumps(fairness_sensitivity, indent=2, default=str), encoding="utf-8")
    _fairness_threshold_plot(fairness_sensitivity, report_dir)
    _plots(frame_test, y_test, test_probabilities, report_dir, threshold_rows=threshold_rows, features=features, config=config)
    plt.figure(figsize=(6, 4))
    order = np.argsort(test_probabilities)
    rolling = pd.Series(y_test.to_numpy()[order]).groupby(np.arange(len(y_test)) // max(1, len(y_test) // 10)).mean()
    plt.plot(np.linspace(0.05, 0.95, len(rolling)), rolling, marker="o")
    plt.xlabel("Risk-score decile"); plt.ylabel("Observed default rate"); plt.title("Reliability by score decile"); plt.tight_layout()
    plt.savefig(reports / "reliability.png", dpi=140); plt.close()
    return metadata


if __name__ == "__main__":
    train()
