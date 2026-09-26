from credit_simulator.policy import decide


def test_policy_bands():
    assert decide(0.10).decision == "approve"
    assert decide(0.30).decision == "manual_review"
    assert decide(0.60).decision == "decline"


def test_outlier_is_review():
    assert decide(0.01, out_of_distribution=True).decision == "manual_review"

