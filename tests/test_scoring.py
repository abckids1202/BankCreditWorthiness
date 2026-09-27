import pytest

from credit_simulator.scoring import probability_to_score, risk_band


def test_credit_score_is_monotonic_and_bounded():
    scores = [probability_to_score(value) for value in (0.01, 0.10, 0.25, 0.50, 0.90)]
    assert scores == sorted(scores, reverse=True)
    assert all(300 <= score <= 850 for score in scores)
    assert probability_to_score(0.25) == probability_to_score(0.25)


def test_invalid_probability_is_rejected():
    with pytest.raises(ValueError):
        probability_to_score(0)
    with pytest.raises(ValueError):
        probability_to_score(1)


def test_risk_bands():
    assert risk_band(0.05) == "low"
    assert risk_band(0.15) == "moderate"
    assert risk_band(0.30) == "high"
    assert risk_band(0.60) == "very_high"


def test_scoring_rejects_non_finite_or_invalid_configuration():
    with pytest.raises(ValueError):
        risk_band(float("nan"))
    with pytest.raises(ValueError):
        risk_band(0.2, {"low": 0.4, "moderate": 0.3, "high": 0.5})
    with pytest.raises(ValueError):
        probability_to_score(0.2, {"base_odds": 0})
