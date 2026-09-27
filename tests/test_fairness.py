import pandas as pd

from credit_simulator.fairness import group_metrics


def test_fairness_reports_group_comparisons():
    frame = pd.DataFrame({"group": ["a", "a", "b", "b"]})
    report = group_metrics(frame, [0, 1, 0, 1], [0.1, 0.6, 0.2, 0.8], ["group"])
    assert "group_comparisons" in report
    assert "b" in report["group_comparisons"]
    assert "approval_rate_ratio" in report["group_comparisons"]["b"]
