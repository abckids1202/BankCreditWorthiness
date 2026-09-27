from __future__ import annotations

import math


def probability_to_score(probability: float, config: dict | None = None) -> int:
    if not 0 < probability < 1:
        raise ValueError("probability must be greater than 0 and less than 1")
    config = config or {}
    base_score = float(config.get("base_score", 600))
    base_odds = float(config.get("base_odds", 19.0))
    points_to_double_odds = float(config.get("points_to_double_odds", 20.0))
    odds = (1.0 - probability) / probability
    score = base_score + points_to_double_odds * math.log(odds / base_odds, 2)
    return int(round(max(float(config.get("min_score", 300)), min(float(config.get("max_score", 850)), score))))


def risk_band(probability: float, bands: dict[str, float] | None = None) -> str:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    bands = bands or {"low": 0.10, "moderate": 0.25, "high": 0.45}
    if probability < bands["low"]:
        return "low"
    if probability < bands["moderate"]:
        return "moderate"
    if probability < bands["high"]:
        return "high"
    return "very_high"
