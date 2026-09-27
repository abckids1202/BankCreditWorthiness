import pandas as pd
import pytest

from credit_simulator.splitting import temporal_split


def test_temporal_split_is_ordered_and_disjoint():
    frame = pd.DataFrame({
        "application_date": ["2024-01-03", "2024-01-01", "2024-01-05", "2024-01-02", "2024-01-04"],
        "value": [3, 1, 5, 2, 4],
    })
    splits = temporal_split(frame, "application_date", validation_fraction=0.2, test_fraction=0.2)
    assert splits.train["value"].tolist() == [1, 2, 3]
    assert splits.validation["value"].tolist() == [4]
    assert splits.test["value"].tolist() == [5]
    assert set(splits.train.index).isdisjoint(splits.test.index)


def test_temporal_split_rejects_invalid_dates_and_missing_column():
    with pytest.raises(ValueError, match="missing"):
        temporal_split(pd.DataFrame({"value": [1, 2, 3]}), "created_at")
    with pytest.raises(ValueError, match="invalid"):
        temporal_split(pd.DataFrame({"created_at": ["bad", "2024-01-01", "2024-01-02"]}), "created_at")
