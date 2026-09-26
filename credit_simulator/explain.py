from __future__ import annotations

import numpy as np
import pandas as pd


def reason_codes(model, frame: pd.DataFrame, feature_names: list[str], top_n: int = 3) -> list[str]:
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
    reasons = []
    for index in ranked:
        direction = "increased" if contributions[index] > 0 else "reduced"
        reasons.append(f"{feature_names[index]} {direction} the modeled default risk")
    return reasons

