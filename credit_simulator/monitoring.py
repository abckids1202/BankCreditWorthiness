from __future__ import annotations

import numpy as np
import pandas as pd


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = reference.dropna(); current = current.dropna()
    if reference.empty or current.empty:
        return 0.0
    if pd.api.types.is_numeric_dtype(reference):
        edges = np.unique(np.nanquantile(reference.astype(float), np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return 0.0
        # Extend the outer bins to infinity so current values outside the
        # reference range are counted rather than silently discarded.
        histogram_edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
        ref_counts, _ = np.histogram(reference, bins=histogram_edges); cur_counts, _ = np.histogram(current, bins=histogram_edges)
    else:
        categories = sorted(set(reference.astype(str)) | set(current.astype(str)))
        ref_counts = reference.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy(); cur_counts = current.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy()
    ref_share = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None); cur_share = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def _level(value: float, warning: float = 0.10, critical: float = 0.25) -> str:
    return "critical" if value >= critical else "warning" if value >= warning else "ok"


def drift_report(reference_records: list[dict], current_records: list[dict], features: list[str] | None = None) -> dict:
    if not reference_records or not current_records:
        raise ValueError("reference_records and current_records must both be non-empty")
    reference, current = pd.DataFrame(reference_records), pd.DataFrame(current_records)
    if features is None:
        features = sorted(set(reference.columns) & set(current.columns))
    else:
        missing_in_reference = sorted(set(features) - set(reference.columns))
        missing_in_current = sorted(set(features) - set(current.columns))
        if missing_in_reference or missing_in_current:
            raise ValueError(f"Requested drift features are missing: reference={missing_in_reference}, current={missing_in_current}")
    if not features:
        raise ValueError("No common features found")
    metrics = {}
    for feature in features:
        psi = _psi(reference[feature], current[feature])
        missing_delta = float(current[feature].isna().mean() - reference[feature].isna().mean())
        metrics[feature] = {"psi": psi, "level": _level(psi), "reference_missing_rate": float(reference[feature].isna().mean()), "current_missing_rate": float(current[feature].isna().mean()), "missing_rate_delta": missing_delta}
    critical = [feature for feature, values in metrics.items() if values["level"] == "critical"]
    warnings = [feature for feature, values in metrics.items() if values["level"] == "warning"]
    return {"reference_rows": len(reference), "current_rows": len(current), "features_checked": list(metrics), "metrics": metrics, "warning_features": warnings, "critical_features": critical, "recommended_action": "Investigate and pause automated use" if critical else "Investigate drift before retraining" if warnings else "No material drift detected", "automatic_retraining": False}
