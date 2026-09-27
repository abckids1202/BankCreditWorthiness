import pandas as pd

from credit_simulator.fairness import group_metrics, threshold_sensitivity


def test_fairness_reports_group_comparisons():
    frame = pd.DataFrame({"group": ["a", "a", "b", "b"]})
    report = group_metrics(frame, [0, 1, 0, 1], [0.1, 0.6, 0.2, 0.8], ["group"])
    assert "group_comparisons" in report
    assert "b" in report["group_comparisons"]
    assert "approval_rate_ratio" in report["group_comparisons"]["b"]


def test_fairness_threshold_sensitivity_returns_each_policy_band():
    frame = pd.DataFrame({"group": ["a", "a", "b", "b"]})
    report = threshold_sensitivity(frame, [0, 1, 0, 1], [0.1, 0.6, 0.2, 0.8], ["group"], [(0.1, 0.4), (0.2, 0.6)])
    assert len(report) == 2
    assert report[1]["metrics"]["group"]["a"]["approval_rate"] == 0.5
