from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from .scoring import probability_to_score


def calibration_error(y_true, probabilities, bins: int = 10) -> float:
    """Compute expected calibration error for one audit group."""
    actual = np.asarray(y_true).astype(int)
    scores = np.asarray(probabilities, dtype=float)
    if len(actual) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    weighted_error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (scores >= left) & ((scores < right) if right < 1 else (scores <= right))
        if mask.any():
            weighted_error += float(mask.mean()) * abs(float(scores[mask].mean()) - float(actual[mask].mean()))
    return float(weighted_error)


def group_metrics(frame, y_true, probabilities, protected_columns, approve_max_risk=0.20, decline_min_risk=0.45):
    y_true = np.asarray(y_true); probabilities = np.asarray(probabilities)
    decisions = np.where(probabilities <= approve_max_risk, "approve", np.where(probabilities >= decline_min_risk, "decline", "manual_review"))
    report = {}
    for column in protected_columns:
        if column not in frame:
            continue
        groups = frame[column].astype(str).to_numpy(); report[column] = {}
        for group in sorted(set(groups)):
            mask = groups == group; actual, scores = y_true[mask], probabilities[mask]; predicted = scores >= 0.5
            tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
            report[column][group] = {"count": int(mask.sum()), "default_rate": float(actual.mean()), "approval_rate": float((decisions[mask] == "approve").mean()), "review_rate": float((decisions[mask] == "manual_review").mean()), "decline_rate": float((decisions[mask] == "decline").mean()), "true_positive_rate": float(tp / max(tp + fn, 1)), "true_negative_rate": float(tn / max(tn + fp, 1)), "false_positive_rate": float(fp / max(fp + tn, 1)), "false_negative_rate": float(fn / max(fn + tp, 1)), "precision": float(precision_score(actual, predicted, zero_division=0)), "recall": float(recall_score(actual, predicted, zero_division=0)), "calibration_error": calibration_error(actual, scores), "average_predicted_risk": float(scores.mean()), "average_credit_score": float(np.mean([probability_to_score(float(value)) for value in np.clip(scores, 1e-6, 1 - 1e-6)]))}
        groups_report = report[column]; reference = sorted(groups_report)[0] if groups_report else None; report[column + "_comparisons"] = {}
        if reference:
            base = groups_report[reference]
            for group, values in groups_report.items():
                if group != reference:
                    approval_difference = values["approval_rate"] - base["approval_rate"]
                    approval_ratio = values["approval_rate"] / max(base["approval_rate"], 1e-9)
                    report[column + "_comparisons"][group] = {"reference_group": reference, "approval_rate_difference": approval_difference, "approval_rate_ratio": approval_ratio, "demographic_parity_difference": approval_difference, "demographic_parity_ratio": approval_ratio, "equal_opportunity_difference": values["true_positive_rate"] - base["true_positive_rate"], "false_positive_rate_difference": values["false_positive_rate"] - base["false_positive_rate"], "false_negative_rate_difference": values["false_negative_rate"] - base["false_negative_rate"], "calibration_error_difference": values["calibration_error"] - base["calibration_error"]}
    return report


def threshold_sensitivity(frame, y_true, probabilities, protected_columns, threshold_pairs):
    """Run the same group audit across policy threshold pairs."""
    return [{"approve_max_risk": float(approve), "decline_min_risk": float(decline), "metrics": group_metrics(frame, y_true, probabilities, protected_columns, approve, decline)} for approve, decline in threshold_pairs]
