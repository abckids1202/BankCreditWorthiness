import pandas as pd
import pytest

from credit_simulator.features import engineer_features


def test_feature_engineering_adds_behavioral_features():
    row = {"LIMIT_BAL": 10000}
    row.update({"PAY_0": 2, "PAY_2": 1, "PAY_3": 0, "PAY_4": -1, "PAY_5": 0, "PAY_6": 0})
    row.update({f"BILL_AMT{i}": 5000 for i in range(1, 7)})
    row.update({f"PAY_AMT{i}": 1000 for i in range(1, 7)})
    output = engineer_features(pd.DataFrame([row]))
    assert output.loc[0, "late_payment_count"] == 2
    assert output.loc[0, "current_utilization"] == 0.5
    assert "payment_to_bill_ratio" not in output.columns
    assert output.loc[0, "average_payment_to_bill_ratio"] == pytest.approx(0.2)
