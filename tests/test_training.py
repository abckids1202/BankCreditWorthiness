import pytest

from credit_simulator.training import _approval_rate_report, _bootstrap_intervals


def test_approval_rate_report_measures_defaults_not_approved():
    rows = _approval_rate_report([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], target_rates=(0.5,))
    assert rows[0]["achieved_approval_rate"] == pytest.approx(0.5)
    assert rows[0]["default_rate_approved"] == 0
    assert rows[0]["default_recall_not_approved"] == 1


def test_approval_rate_report_returns_expected_operating_points():
    rows = _approval_rate_report([0, 1, 0, 1], [0.1, 0.2, 0.8, 0.9], target_rates=(0.5, 0.9))
    assert len(rows) == 2
    assert rows[0]["approved_count"] == 2
    assert rows[1]["target_approval_rate"] == pytest.approx(0.9)


def test_bootstrap_intervals_are_reproducible_and_bounded():
    y = [0, 1] * 20
    probabilities = [0.1, 0.9] * 20
    first = _bootstrap_intervals(y, probabilities, random_state=7, n_bootstrap=40)
    second = _bootstrap_intervals(y, probabilities, random_state=7, n_bootstrap=40)
    assert first == second
    assert first["successful_samples"] > 0
    assert 0 <= first["metrics"]["roc_auc"]["lower_95"] <= first["metrics"]["roc_auc"]["upper_95"] <= 1
