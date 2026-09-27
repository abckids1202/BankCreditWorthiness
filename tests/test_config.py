import pytest

from credit_simulator.config import load_config, validate_config


def test_default_config_is_valid():
    assert load_config()["thresholds"]["approve_max_risk"] < load_config()["thresholds"]["decline_min_risk"]


def test_invalid_config_threshold_order_is_rejected():
    config = load_config()
    config["thresholds"]["approve_max_risk"] = 0.8
    config["thresholds"]["decline_min_risk"] = 0.2
    with pytest.raises(ValueError, match="thresholds"):
        validate_config(config)


def test_missing_policy_version_is_rejected():
    config = load_config()
    config["policy_version"] = ""
    with pytest.raises(ValueError, match="policy_version"):
        validate_config(config)


def test_invalid_split_sizes_are_rejected():
    config = load_config()
    config["test_size"] = 0.8
    config["validation_size"] = 0.3
    with pytest.raises(ValueError, match="test_size"):
        validate_config(config)


def test_invalid_score_settings_are_rejected():
    config = load_config()
    config["score"]["base_odds"] = 0
    with pytest.raises(ValueError, match="score"):
        validate_config(config)
