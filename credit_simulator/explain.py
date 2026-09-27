from __future__ import annotations

import numpy as np
import pandas as pd


def structured_reasons(model, frame: pd.DataFrame, feature_names: list[str], descriptions: dict[str, str] | None = None, top_n: int = 3) -> list[dict]:
    """Return transparent, directional reasons; these are not adverse-action notices."""
    values = frame[feature_names].astype(float).iloc[0].to_numpy()
    imputer = model.named_steps.get("imputer")
    scaler = model.named_steps.get("scaler")
    estimator = model.named_steps.get("model")
    transformed = imputer.transform(frame[feature_names])
    if scaler is not None:
        transformed = scaler.transform(transformed)
    if hasattr(estimator, "coef_"):
        contributions = transformed[0] * estimator.coef_[0]
    else:
        importance = getattr(estimator, "feature_importances_", np.ones(len(feature_names)))
        center = np.nan_to_num(transformed[0])
        contributions = center * importance
    ranked = np.argsort(np.abs(contributions))[::-1][:top_n]
    descriptions = descriptions or {}
    reasons = []
    for index in ranked:
        direction = "increased" if contributions[index] > 0 else "reduced"
        reasons.append({"feature": feature_names[index], "description": descriptions.get(feature_names[index], feature_names[index]), "value": None if pd.isna(values[index]) else float(values[index]), "direction": "increased_risk" if contributions[index] > 0 else "reduced_risk", "importance": float(abs(contributions[index]))})
    return reasons


def reason_codes(model, frame: pd.DataFrame, feature_names: list[str], top_n: int = 3) -> list[str]:
    return [f"{reason['feature']} {'increased' if reason['direction'] == 'increased_risk' else 'reduced'} the modeled default risk" for reason in structured_reasons(model, frame, feature_names, top_n=top_n)]

