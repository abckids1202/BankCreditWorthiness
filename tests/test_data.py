import pandas as pd
import pytest

from credit_simulator.data import model_features, validate_frame


def test_sensitive_columns_are_not_model_features():
    frame = pd.DataFrame({"ID": [1], "LIMIT_BAL": [10], "SEX": [1], "EDUCATION": [2], "MARRIAGE": [1], "AGE": [30], "default": [0], "PAY_0": [0]})
    assert model_features(frame) == ["LIMIT_BAL", "PAY_0"]
    assert not set(model_features(frame)).intersection({"SEX", "EDUCATION", "MARRIAGE", "AGE"})


def test_invalid_target_is_rejected():
    frame = pd.DataFrame({"LIMIT_BAL": [10], "SEX": [1], "EDUCATION": [2], "MARRIAGE": [1], "AGE": [30], "default": [2]})
    with pytest.raises(ValueError):
        validate_frame(frame)


def _valid_frame(rows=100):
    return pd.DataFrame({
        "LIMIT_BAL": [1000] * rows, "SEX": [1] * rows, "EDUCATION": [2] * rows,
        "MARRIAGE": [1] * rows, "AGE": [30] * rows, "default": [0] * rows,
        "PAY_0": [0] * rows, "PAY_2": [0] * rows, "PAY_3": [0] * rows,
        "PAY_4": [0] * rows, "PAY_5": [0] * rows, "PAY_6": [0] * rows,
        "PAY_AMT1": [0] * rows,
    })


def test_out_of_range_feature_is_rejected():
    frame = _valid_frame()
    frame.loc[0, "AGE"] = 5
    with pytest.raises(ValueError, match="AGE"):
        validate_frame(frame)


def test_negative_payment_is_rejected():
    frame = _valid_frame()
    frame.loc[0, "PAY_AMT1"] = -1
    with pytest.raises(ValueError, match="PAY_AMT1"):
        validate_frame(frame)


def test_non_numeric_balance_is_rejected():
    frame = _valid_frame()
    frame["LIMIT_BAL"] = frame["LIMIT_BAL"].astype(object)
    frame.loc[0, "LIMIT_BAL"] = "unknown"
    with pytest.raises(ValueError, match="LIMIT_BAL"):
        validate_frame(frame)


def test_valid_domains_are_accepted():
    validate_frame(_valid_frame())
