import pandas as pd
import pytest

from credit_simulator.fairness import calibration_error, group_metrics, threshold_sensitivity


def test_fairness_reports_group_comparisons():
    frame = pd.DataFrame({"group": ["a", "a", "b", "b"]})
    report = group_metrics(frame, [0, 1, 0, 1], [0.1, 0.6, 0.2, 0.8], ["group"])
    assert "group_comparisons" in report
    assert "b" in report["group_comparisons"]
    assert "approval_rate_ratio" in report["group_comparisons"]["b"]
    assert report["group_comparisons"]["b"]["demographic_parity_difference"] == report["group_comparisons"]["b"]["approval_rate_difference"]


def test_fairness_threshold_sensitivity_returns_each_policy_band():
    frame = pd.DataFrame({"group": ["a", "a", "b", "b"]})
    report = threshold_sensitivity(frame, [0, 1, 0, 1], [0.1, 0.6, 0.2, 0.8], ["group"], [(0.1, 0.4), (0.2, 0.6)])
    assert len(report) == 2
    assert report[1]["metrics"]["group"]["a"]["approval_rate"] == 0.5


def test_group_calibration_error_detects_bin_level_miscalibration():
    # Mean predicted risk equals the mean outcome, but each risk bin is wrong.
    assert calibration_error([0, 1, 0, 1], [0.4, 0.6, 0.6, 0.4]) == pytest.approx(0.1)
    report = group_metrics(pd.DataFrame({"group": ["a"] * 4}), [0, 1, 0, 1], [0.4, 0.6, 0.6, 0.4], ["group"])
    assert report["group"]["a"]["calibration_error"] == pytest.approx(0.1)


def test_group_metrics_accepts_an_explicit_classification_threshold():
    frame = pd.DataFrame({"group": ["a", "a", "b", "b"]})
    report = group_metrics(frame, [0, 1, 1, 0], [0.1, 0.4, 0.6, 0.7], ["group"], 0.2, 0.5, classification_threshold=0.5)
    assert report["group"]["a"]["true_positive_rate"] == 0
    assert report["group"]["b"]["true_positive_rate"] == 1
