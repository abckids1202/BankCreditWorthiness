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
