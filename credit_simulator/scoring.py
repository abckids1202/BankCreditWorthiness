from __future__ import annotations

import math


def _score_settings(config: dict) -> tuple[float, float, float, float, float]:
    base_score = float(config.get("base_score", 600))
    base_odds = float(config.get("base_odds", 19.0))
    points_to_double_odds = float(config.get("points_to_double_odds", 20.0))
    min_score = float(config.get("min_score", 300))
    max_score = float(config.get("max_score", 850))
    if not all(math.isfinite(value) for value in (base_score, base_odds, points_to_double_odds, min_score, max_score)) or base_odds <= 0 or points_to_double_odds <= 0 or min_score >= max_score:
        raise ValueError("score configuration must be finite with positive odds/points and min_score < max_score")
    return base_score, base_odds, points_to_double_odds, min_score, max_score


def probability_to_score(probability: float, config: dict | None = None) -> int:
    if not 0 < probability < 1:
        raise ValueError("probability must be greater than 0 and less than 1")
    config = config or {}
    base_score, base_odds, points_to_double_odds, min_score, max_score = _score_settings(config)
    odds = (1.0 - probability) / probability
    score = base_score + points_to_double_odds * math.log(odds / base_odds, 2)
    return int(round(max(min_score, min(max_score, score))))


def risk_band(probability: float, bands: dict[str, float] | None = None) -> str:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    bands = bands or {"low": 0.10, "moderate": 0.25, "high": 0.45}
    try:
        low, moderate, high = (float(bands[key]) for key in ("low", "moderate", "high"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("risk bands must contain low, moderate, and high thresholds") from exc
    if not all(math.isfinite(value) for value in (low, moderate, high)) or not 0 <= low < moderate < high <= 1:
        raise ValueError("risk band thresholds must satisfy 0 <= low < moderate < high <= 1")
    if probability < low:
        return "low"
    if probability < moderate:
        return "moderate"
    if probability < high:
        return "high"
    return "very_high"
