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


def _has_coefficients(estimator) -> bool:
    """Return whether the wrapped estimator exposes signed linear effects."""
    if hasattr(estimator, "coef_"):
        return True
    estimators = getattr(estimator, "calibrated_classifiers_", None)
    return bool(estimators) and all(_has_coefficients(getattr(item, "estimator", getattr(item, "base_estimator", item))) for item in estimators)


def _local_ablation_effects(model, frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Estimate per-applicant effects for nonlinear models by feature ablation.

    Each feature is replaced with the fitted imputer statistic (or the applicant
    frame median when no fitted imputer exists), then the probability change is
    measured. This is a local model explanation, not a causal effect.
    """
    working = frame.copy()
    working[feature_names] = working[feature_names].astype(float)
    imputer = model.named_steps.get("imputer")
    if imputer is not None and hasattr(imputer, "statistics_"):
        baseline_values = np.asarray(imputer.statistics_, dtype=float)
    else:
        baseline_values = working[feature_names].astype(float).median(axis=0).to_numpy(dtype=float)
    original = float(model.predict_proba(working[feature_names])[:, 1][0])
    effects = np.zeros(len(feature_names), dtype=float)
    for index, feature in enumerate(feature_names):
        ablated = working.copy()
        value = baseline_values[index] if index < len(baseline_values) else np.nan
        ablated.loc[ablated.index[0], feature] = value
        effects[index] = original - float(model.predict_proba(ablated[feature_names])[:, 1][0])
    return effects


def structured_reasons(model, frame: pd.DataFrame, feature_names: list[str], descriptions: dict[str, str] | None = None, top_n: int = 3) -> list[dict]:
    """Return transparent, directional reasons; these are not adverse-action notices."""
    values = frame[feature_names].astype(float).iloc[0].to_numpy()
    imputer = model.named_steps.get("imputer")
    scaler = model.named_steps.get("scaler")
    estimator = model.named_steps.get("model")
    transformed = imputer.transform(frame[feature_names])
    if scaler is not None:
        transformed = scaler.transform(transformed)
    if _has_coefficients(estimator):
        contributions = np.nan_to_num(transformed[0]) * _feature_effect_vector(estimator, len(feature_names))
    else:
        contributions = _local_ablation_effects(model, frame, feature_names)
    ranked = np.argsort(np.abs(contributions))[::-1][:top_n]
    descriptions = descriptions or {}
    reasons = []
    for index in ranked:
        direction = "increased" if contributions[index] > 0 else "reduced"
        reasons.append({"feature": feature_names[index], "description": descriptions.get(feature_names[index], feature_names[index]), "value": None if pd.isna(values[index]) else float(values[index]), "direction": "increased_risk" if contributions[index] > 0 else "reduced_risk", "importance": float(abs(contributions[index]))})
    return reasons


def reason_codes(model, frame: pd.DataFrame, feature_names: list[str], top_n: int = 3) -> list[str]:
    return [f"{reason['feature']} {'increased' if reason['direction'] == 'increased_risk' else 'reduced'} the modeled default risk" for reason in structured_reasons(model, frame, feature_names, top_n=top_n)]

