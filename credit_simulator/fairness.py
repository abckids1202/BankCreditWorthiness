from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from .scoring import probability_to_score


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
            report[column][group] = {"count": int(mask.sum()), "default_rate": float(actual.mean()), "approval_rate": float((decisions[mask] == "approve").mean()), "review_rate": float((decisions[mask] == "manual_review").mean()), "decline_rate": float((decisions[mask] == "decline").mean()), "true_positive_rate": float(tp / max(tp + fn, 1)), "true_negative_rate": float(tn / max(tn + fp, 1)), "false_positive_rate": float(fp / max(fp + tn, 1)), "false_negative_rate": float(fn / max(fn + tp, 1)), "precision": float(precision_score(actual, predicted, zero_division=0)), "recall": float(recall_score(actual, predicted, zero_division=0)), "calibration_error": float(abs(scores.mean() - actual.mean())), "average_predicted_risk": float(scores.mean()), "average_credit_score": float(np.mean([probability_to_score(float(value)) for value in np.clip(scores, 1e-6, 1 - 1e-6)]))}
        groups_report = report[column]; reference = sorted(groups_report)[0] if groups_report else None; report[column + "_comparisons"] = {}
        if reference:
            base = groups_report[reference]
            for group, values in groups_report.items():
                if group != reference:
                    report[column + "_comparisons"][group] = {"reference_group": reference, "approval_rate_difference": values["approval_rate"] - base["approval_rate"], "approval_rate_ratio": values["approval_rate"] / max(base["approval_rate"], 1e-9), "equal_opportunity_difference": values["true_positive_rate"] - base["true_positive_rate"], "false_positive_rate_difference": values["false_positive_rate"] - base["false_positive_rate"], "false_negative_rate_difference": values["false_negative_rate"] - base["false_negative_rate"], "calibration_error_difference": values["calibration_error"] - base["calibration_error"]}
    return report
