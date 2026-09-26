from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    decision: str
    rationale: str


def decide(risk_probability: float, *, approve_max_risk: float = 0.20, decline_min_risk: float = 0.45, out_of_distribution: bool = False) -> Decision:
    if out_of_distribution:
        return Decision("manual_review", "Input is outside the training distribution")
    if not 0 <= risk_probability <= 1:
        raise ValueError("risk_probability must be between 0 and 1")
    if risk_probability <= approve_max_risk:
        return Decision("approve", "Predicted risk is below the approval threshold")
    if risk_probability >= decline_min_risk:
        return Decision("decline", "Predicted risk is above the decline threshold")
    return Decision("manual_review", "Predicted risk is in the human-review band")

