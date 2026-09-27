import pytest

from credit_simulator.training import _approval_rate_report


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
