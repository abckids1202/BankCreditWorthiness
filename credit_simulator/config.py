from pathlib import Path
import math
import yaml


ROOT = Path(__file__).resolve().parents[1]


def validate_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    if not isinstance(config.get("policy_version"), str) or not config["policy_version"].strip():
        raise ValueError("policy_version must be a non-empty string")
    splits = [config.get("test_size"), config.get("validation_size")]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and 0 < value < 1 for value in splits) or sum(splits) >= 1:
        raise ValueError("test_size and validation_size must be positive, finite, and sum to less than 1")
    random_state = config.get("random_state")
    if not isinstance(random_state, int) or random_state < 0:
        raise ValueError("random_state must be a non-negative integer")
    thresholds = config.get("thresholds", {})
    approve = thresholds.get("approve_max_risk")
    decline = thresholds.get("decline_min_risk")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (approve, decline)) or not 0 <= approve < decline <= 1:
        raise ValueError("thresholds must satisfy 0 <= approve_max_risk < decline_min_risk <= 1")
    for key in ("max_missing_fraction",):
        value = thresholds.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"thresholds.{key} must be between 0 and 1")
    outlier_z = thresholds.get("out_of_distribution_z")
    if not isinstance(outlier_z, (int, float)) or not math.isfinite(outlier_z) or outlier_z <= 0:
        raise ValueError("thresholds.out_of_distribution_z must be positive")
    bands = config.get("risk_bands", {})
    band_values = [bands.get(name) for name in ("low", "moderate", "high")]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in band_values) or not 0 <= band_values[0] < band_values[1] < band_values[2] <= 1:
        raise ValueError("risk_bands must satisfy 0 <= low < moderate < high <= 1")
    score = config.get("score", {})
    score_values = [score.get(name) for name in ("min_score", "max_score", "base_score", "base_odds", "points_to_double_odds")]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in score_values) or not score["min_score"] < score["max_score"] or not score["min_score"] <= score["base_score"] <= score["max_score"] or score["base_odds"] <= 0 or score["points_to_double_odds"] <= 0:
        raise ValueError("score settings are invalid")
    costs = config.get("costs", {})
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in costs.values()) or not {"approve_default", "decline_good", "manual_review"}.issubset(costs):
        raise ValueError("costs must contain non-negative finite decision costs")
    return config


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return validate_config(yaml.safe_load(handle))

