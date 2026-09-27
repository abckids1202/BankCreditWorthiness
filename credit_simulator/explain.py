from __future__ import annotations

import numpy as np
import pandas as pd


def _feature_effect_vector(estimator, size: int) -> np.ndarray:
    """Get a stable directional/importance vector through calibration wrappers."""
    estimators = getattr(estimator, "calibrated_classifiers_", None)
    if estimators:
        vectors = [_feature_effect_vector(getattr(item, "estimator", getattr(item, "base_estimator", item)), size) for item in estimators]
        return np.nanmean(vectors, axis=0)
    if hasattr(estimator, "coef_"):
        return np.asarray(estimator.coef_)[0]
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_)
    return np.ones(size)


def structured_reasons(model, frame: pd.DataFrame, feature_names: list[str], descriptions: dict[str, str] | None = None, top_n: int = 3) -> list[dict]:
    """Return transparent, directional reasons; these are not adverse-action notices."""
    values = frame[feature_names].astype(float).iloc[0].to_numpy()
    imputer = model.named_steps.get("imputer")
    scaler = model.named_steps.get("scaler")
    estimator = model.named_steps.get("model")
    transformed = imputer.transform(frame[feature_names])
    if scaler is not None:
        transformed = scaler.transform(transformed)
    contributions = np.nan_to_num(transformed[0]) * _feature_effect_vector(estimator, len(feature_names))
    ranked = np.argsort(np.abs(contributions))[::-1][:top_n]
    descriptions = descriptions or {}
    reasons = []
    for index in ranked:
        direction = "increased" if contributions[index] > 0 else "reduced"
        reasons.append({"feature": feature_names[index], "description": descriptions.get(feature_names[index], feature_names[index]), "value": None if pd.isna(values[index]) else float(values[index]), "direction": "increased_risk" if contributions[index] > 0 else "reduced_risk", "importance": float(abs(contributions[index]))})
    return reasons


def reason_codes(model, frame: pd.DataFrame, feature_names: list[str], top_n: int = 3) -> list[str]:
    return [f"{reason['feature']} {'increased' if reason['direction'] == 'increased_risk' else 'reduced'} the modeled default risk" for reason in structured_reasons(model, frame, feature_names, top_n=top_n)]

