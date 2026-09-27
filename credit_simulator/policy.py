from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Decision:
    decision: str
    rationale: str


def validate_thresholds(approve_max_risk: float, decline_min_risk: float) -> None:
    if not all(math.isfinite(value) for value in (approve_max_risk, decline_min_risk)) or not 0 <= approve_max_risk < decline_min_risk <= 1:
        raise ValueError("thresholds must satisfy 0 <= approve < decline <= 1")


def decide(risk_probability: float, *, approve_max_risk: float = 0.20, decline_min_risk: float = 0.45, out_of_distribution: bool = False) -> Decision:
    validate_thresholds(approve_max_risk, decline_min_risk)
    if out_of_distribution:
        return Decision("manual_review", "Input is outside the training distribution")
    if not math.isfinite(risk_probability) or not 0 <= risk_probability <= 1:
        raise ValueError("risk_probability must be between 0 and 1")
    if risk_probability <= approve_max_risk:
        return Decision("approve", "Predicted risk is below the approval threshold")
    if risk_probability >= decline_min_risk:
        return Decision("decline", "Predicted risk is above the decline threshold")
    return Decision("manual_review", "Predicted risk is in the human-review band")


def simulate_thresholds(
    probabilities: list[float],
    approve_max_risk: float,
    decline_min_risk: float,
    actual_defaults: list[int] | None = None,
    costs: dict[str, float] | None = None,
) -> dict:
    validate_thresholds(approve_max_risk, decline_min_risk)
    if not probabilities or any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("probabilities must be non-empty and between 0 and 1")
    if actual_defaults is not None and (len(actual_defaults) != len(probabilities) or any(value not in (0, 1) for value in actual_defaults)):
        raise ValueError("actual_defaults must match probabilities and contain only 0/1")
    decisions = ["approve" if value <= approve_max_risk else "decline" if value >= decline_min_risk else "manual_review" for value in probabilities]
    result = {"count": len(probabilities), "approval_rate": decisions.count("approve") / len(decisions), "review_rate": decisions.count("manual_review") / len(decisions), "decline_rate": decisions.count("decline") / len(decisions), "approve_max_risk": approve_max_risk, "decline_min_risk": decline_min_risk}
    if actual_defaults is not None:
        actual = [int(value) for value in actual_defaults]
        predicted_default = [value >= decline_min_risk for value in probabilities]
        true_positive = sum(y == 1 and predicted for y, predicted in zip(actual, predicted_default))
        true_negative = sum(y == 0 and not predicted for y, predicted in zip(actual, predicted_default))
        false_positive = sum(y == 0 and predicted for y, predicted in zip(actual, predicted_default))
        false_negative = sum(y == 1 and not predicted for y, predicted in zip(actual, predicted_default))
        configured_costs = costs or {"approve_default": 5.0, "decline_good": 1.0, "manual_review": 0.2}
        required_costs = {"approve_default", "decline_good", "manual_review"}
        if set(configured_costs) != required_costs or any(not math.isfinite(float(value)) or float(value) < 0 for value in configured_costs.values()):
            raise ValueError("costs must contain exactly approve_default, decline_good, and manual_review as non-negative finite values")
        expected_cost = sum(y == 1 and decision == "approve" for y, decision in zip(actual, decisions)) * configured_costs["approve_default"]
        expected_cost += sum(y == 0 and decision == "decline" for y, decision in zip(actual, decisions)) * configured_costs["decline_good"]
        expected_cost += decisions.count("manual_review") * configured_costs["manual_review"]
        result.update({"default_rate_approved": sum(y for y, decision in zip(actual, decisions) if decision == "approve") / max(decisions.count("approve"), 1), "default_rate_reviewed": sum(y for y, decision in zip(actual, decisions) if decision == "manual_review") / max(decisions.count("manual_review"), 1), "default_rate_declined": sum(y for y, decision in zip(actual, decisions) if decision == "decline") / max(decisions.count("decline"), 1), "true_positive": true_positive, "true_negative": true_negative, "false_positive": false_positive, "false_negative": false_negative, "precision": true_positive / max(true_positive + false_positive, 1), "recall": true_positive / max(true_positive + false_negative, 1), "false_positive_rate": false_positive / max(false_positive + true_negative, 1), "false_negative_rate": false_negative / max(false_negative + true_positive, 1), "expected_cost": float(expected_cost), "decision_costs": configured_costs})
    return result

