import pytest

from credit_simulator.policy import decide, simulate_thresholds


def test_policy_bands():
    assert decide(0.10).decision == "approve"
    assert decide(0.30).decision == "manual_review"
    assert decide(0.60).decision == "decline"


def test_outlier_is_review():
    assert decide(0.01, out_of_distribution=True).decision == "manual_review"


def test_threshold_simulation():
    result = simulate_thresholds([0.1, 0.3, 0.7], 0.2, 0.5, [0, 1, 1])
    assert result["approval_rate"] == pytest.approx(1 / 3)
    assert result["review_rate"] == pytest.approx(1 / 3)
    assert result["decline_rate"] == pytest.approx(1 / 3)
    assert result["default_rate_approved"] == 0

