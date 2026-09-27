from fastapi.testclient import TestClient
import pytest

import credit_simulator.api as api
from credit_simulator.api import app
import shutil
from credit_simulator.review import ReviewStore


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers.get("X-Request-ID")


def test_request_id_is_preserved():
    response = TestClient(app).get("/health", headers={"X-Request-ID": "test-request"})
    assert response.headers["X-Request-ID"] == "test-request"


def test_malformed_request_id_is_replaced():
    supplied = "x" * 129
    response = TestClient(app).get("/health", headers={"X-Request-ID": supplied})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != supplied
    assert len(response.headers["X-Request-ID"]) == 36


def test_prediction_response_schema_rejects_invalid_risk_and_score():
    with pytest.raises(ValueError):
        api.Prediction(risk_probability=float("nan"), credit_score=600, risk_band="low", decision="approve", rationale="test", decision_thresholds={}, model_version="test", policy_version="test", reason_codes=[], explanations=[], warnings=[], educational_disclaimer="test")
    with pytest.raises(ValueError):
        api.Prediction(risk_probability=0.2, credit_score=299, risk_band="low", decision="approve", rationale="test", decision_thresholds={}, model_version="test", policy_version="test", reason_codes=[], explanations=[], warnings=[], educational_disclaimer="test")


def test_ready_endpoint():
    response = TestClient(app).get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["policy_version"] == "policy-0.1.0"
    assert response.json()["artifact_fingerprint"]


def test_ready_endpoint_reports_invalid_artifacts(monkeypatch, tmp_path):
    (tmp_path / "model.joblib").write_bytes(b"not-a-model")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(api, "ARTIFACT_DIR", tmp_path)
    response = TestClient(app).get("/ready")
    assert response.status_code == 503


def test_ready_endpoint_reports_checksum_mismatch(monkeypatch, tmp_path):
    source_model = api.ARTIFACT_DIR / "model.joblib"
    source_metadata = api.ARTIFACT_DIR / "metadata.json"
    shutil.copyfile(source_model, tmp_path / "model.joblib")
    shutil.copyfile(source_metadata, tmp_path / "metadata.json")
    with (tmp_path / "model.joblib").open("ab") as handle:
        handle.write(b"tampered")
    monkeypatch.setattr(api, "ARTIFACT_DIR", tmp_path)
    response = TestClient(app).get("/ready")
    assert response.status_code == 503


def test_prediction_stats_do_not_store_raw_inputs():
    response = TestClient(app).get("/prediction-stats")
    assert response.status_code == 200
    assert response.json()["raw_inputs_stored"] is False


def test_batch_prediction_endpoint():
    applicant = {"LIMIT_BAL": 50000, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000, "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000, "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000, "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000}
    response = TestClient(app).post("/predict/batch", json={"applicants": [applicant, applicant]})
    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert len(response.json()["predictions"]) == 2
    assert response.json()["predictions"][0]["policy_version"] == "policy-0.1.0"


def test_out_of_distribution_input_is_reviewed():
    applicant = {"LIMIT_BAL": 1_000_000_000, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000, "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000, "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000, "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000}
    response = TestClient(app).post("/predict", json=applicant)
    assert response.status_code == 200
    assert response.json()["decision"] == "manual_review"


def test_review_queue_rejects_non_manual_recommendations(monkeypatch):
    applicant = {"LIMIT_BAL": 50000, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000, "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000, "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000, "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000}
    prediction = api.Prediction(risk_probability=0.1, credit_score=750, risk_band="low", decision="approve", rationale="low risk", decision_thresholds={"approve_max_risk": 0.2, "decline_min_risk": 0.45}, model_version="test", policy_version="policy-test", reason_codes=[], explanations=[], warnings=[], educational_disclaimer="test")
    monkeypatch.setattr(api, "predict", lambda applicant, request: prediction)
    response = TestClient(app).post("/review-cases", json=applicant)
    assert response.status_code == 422


def test_out_of_range_repayment_status_is_rejected():
    applicant = {"LIMIT_BAL": 50000, "PAY_0": 9, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000, "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000, "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000, "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000}
    assert TestClient(app).post("/predict", json=applicant).status_code == 422


def test_negative_payment_is_rejected_by_api():
    applicant = {"LIMIT_BAL": 50000, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000, "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000, "PAY_AMT1": -1, "PAY_AMT2": 2000, "PAY_AMT3": 2000, "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000}
    assert TestClient(app).post("/predict", json=applicant).status_code == 422


def test_optional_api_key(monkeypatch):
    monkeypatch.setenv("CREDIT_API_KEY", "test-secret")
    client = TestClient(app)
    assert client.get("/model-info").status_code == 401
    assert client.get("/model-info", headers={"X-API-Key": "wrong-secret"}).status_code == 401
    assert client.get("/model-info", headers={"X-API-Key": "test-secret"}).status_code == 200


def test_optional_rate_limit(monkeypatch):
    monkeypatch.setenv("CREDIT_RATE_LIMIT_PER_MINUTE", "1")
    from credit_simulator.api import rate_limiter
    rate_limiter.clear()
    client = TestClient(app)
    first = client.get("/model-info")
    assert first.status_code == 200
    assert first.headers["X-RateLimit-Limit"] == "1"
    assert first.headers["X-RateLimit-Remaining"] == "0"
    limited = client.get("/model-info")
    assert limited.status_code == 429
    assert limited.headers.get("Retry-After")
    assert limited.headers["X-RateLimit-Remaining"] == "0"
    rate_limiter.clear()


def test_unknown_alternate_dataset_is_rejected():
    response = TestClient(app).post("/predict/not_a_dataset", json={"features": {}})
    assert response.status_code == 404


def test_policy_simulation_endpoint():
    response = TestClient(app).post("/policy/simulate", json={"probabilities": [0.1, 0.3, 0.7], "approve_max_risk": 0.2, "decline_min_risk": 0.5})
    assert response.status_code == 200
    assert response.json()["approval_rate"] == 1 / 3


def test_monitoring_drift_endpoint():
    response = TestClient(app).post("/monitoring/drift", json={"reference_records": [{"x": 1}, {"x": 2}], "current_records": [{"x": 100}, {"x": 100}], "features": ["x"]})
    assert response.status_code == 200
    assert response.json()["metrics"]["x"]["psi"] > 0


def test_monitoring_drift_history_endpoint(monkeypatch, tmp_path):
    from credit_simulator.drift_events import DriftEventStore

    monkeypatch.setattr(api, "drift_store", DriftEventStore(tmp_path / "drift.db"))
    client = TestClient(app)
    response = client.post("/monitoring/drift", json={"reference_records": [{"x": 1}], "current_records": [{"x": 2}], "features": ["x"]})
    assert response.status_code == 200
    history = client.get("/monitoring/drift/history")
    assert history.status_code == 200
    assert history.json()[0]["current_rows"] == 1


def test_review_history_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "review_store", ReviewStore(tmp_path / "reviews.db"))
    applicant = {"LIMIT_BAL": 1_000_000_000, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000, "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000, "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000, "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000}
    client = TestClient(app)
    created = client.post("/review-cases", json=applicant)
    assert created.status_code == 200
    case_id = created.json()["case_id"]
    history = client.get(f"/review-cases/{case_id}/history")
    assert history.status_code == 200
    assert history.json()["events"][0]["event_type"] == "created"
    updated = client.patch(f"/review-cases/{case_id}", json={"reviewer_decision": "approved", "reviewer_note": "Educational test"})
    assert updated.status_code == 200
    assert len(client.get(f"/review-cases/{case_id}/history").json()["events"]) == 2


def test_review_openapi_declares_typed_case_responses():
    schema = TestClient(app).get("/openapi.json").json()
    assert "ReviewCase" in schema["components"]["schemas"]
    assert schema["paths"]["/review-cases"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]["type"] == "array"

