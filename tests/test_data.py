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
