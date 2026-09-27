from credit_simulator.review import ReviewStore


def test_review_case_preserves_model_decision(tmp_path):
    store = ReviewStore(tmp_path / "reviews.db")
    case = store.create({"LIMIT_BAL": 1000}, {"model_version": "test", "risk_probability": 0.3, "credit_score": 600, "decision": "manual_review", "reason_codes": ["test"], "warnings": []})
    assert case["automatic_decision"] == "manual_review"
    updated = store.update(case["case_id"], "approved", "Reviewed for prototype")
    assert updated["automatic_decision"] == "manual_review"
    assert updated["reviewer_decision"] == "approved"

