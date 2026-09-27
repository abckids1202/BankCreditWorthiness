from fastapi.testclient import TestClient
import json
import hashlib
import joblib
import numpy as np
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
        api.Prediction(risk_probability=float("nan"), default_probability=float("nan"), credit_score=600, risk_band="low", decision="approve", rationale="test", decision_thresholds={}, model_version="test", policy_version="test", reason_codes=[], explanations=[], warnings=[], educational_disclaimer="test")
    with pytest.raises(ValueError):
        api.Prediction(risk_probability=0.2, default_probability=0.2, credit_score=299, risk_band="low", decision="approve", rationale="test", decision_thresholds={}, model_version="test", policy_version="test", reason_codes=[], explanations=[], warnings=[], educational_disclaimer="test")


def test_ready_endpoint():
    response = TestClient(app).get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["policy_version"] == "policy-0.1.0"
    assert response.json()["artifact_fingerprint"]
    assert response.json()["experiment_id"].startswith("exp-")
    assert response.json()["schema_version"] == "1.0"


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


def test_ready_endpoint_reports_incomplete_manifest(monkeypatch, tmp_path):
    source_model = api.ARTIFACT_DIR / "model.joblib"
    source_metadata = api.ARTIFACT_DIR / "metadata.json"
    shutil.copyfile(source_model, tmp_path / "model.joblib")
    metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    metadata.pop("training_config_sha256", None)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
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
    assert len(response.json()["predictions"][0]["dataset_version"]) == 64
    assert response.json()["predictions"][0]["default_probability"] == response.json()["predictions"][0]["risk_probability"]
    assert response.json()["predictions"][0]["fairness_warnings"]
    assert response.json()["predictions"][0]["experiment_id"].startswith("exp-")


def test_out_of_distribution_input_is_reviewed():
    applicant = {"LIMIT_BAL": 1_000_000_000, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000, "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000, "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000, "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000}
    response = TestClient(app).post("/predict", json=applicant)
    assert response.status_code == 200
    assert response.json()["decision"] == "manual_review"


def test_missing_engineered_feature_is_reported(monkeypatch):
    applicant = {"LIMIT_BAL": 50000, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000, "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000, "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000, "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000}
    original_engineer_features = api.engineer_features

    def inject_missing_feature(frame):
        engineered = original_engineer_features(frame)
        engineered.loc[0, "current_utilization"] = np.nan
        return engineered

    monkeypatch.setattr(api, "engineer_features", inject_missing_feature)
    response = TestClient(app).post("/predict", json=applicant)

    assert response.status_code == 200
    assert any("imputed" in warning.lower() for warning in response.json()["warnings"])


def test_review_queue_rejects_non_manual_recommendations(monkeypatch):
    applicant = {"LIMIT_BAL": 50000, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000, "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000, "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000, "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000}
    prediction = api.Prediction(risk_probability=0.1, default_probability=0.1, credit_score=750, risk_band="low", decision="approve", rationale="low risk", decision_thresholds={"approve_max_risk": 0.2, "decline_min_risk": 0.45}, model_version="test", policy_version="policy-test", reason_codes=[], explanations=[], warnings=[], educational_disclaimer="test")
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


def test_alternate_artifact_requires_complete_manifest(monkeypatch, tmp_path):
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="prior").fit(np.array([[0.0]]), np.array([0]))
    dataset_dir = tmp_path / "german_credit"
    dataset_dir.mkdir()
    model_path = dataset_dir / "model.joblib"
    joblib.dump(model, model_path)
    (dataset_dir / "metadata.json").write_text(json.dumps({"model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}), encoding="utf-8")
    monkeypatch.setattr(api, "ARTIFACT_DIR", tmp_path)
    with pytest.raises(Exception):
        api._dataset_artifacts("german_credit")


def test_policy_simulation_endpoint():
    response = TestClient(app).post("/policy/simulate", json={"probabilities": [0.1, 0.3, 0.7], "approve_max_risk": 0.2, "decline_min_risk": 0.5})
    assert response.status_code == 200
    assert response.json()["approval_rate"] == 1 / 3


def test_policy_simulation_endpoint_returns_cost_and_error_metrics():
    response = TestClient(app).post("/policy/simulate", json={"probabilities": [0.1, 0.3, 0.7], "approve_max_risk": 0.2, "decline_min_risk": 0.5, "actual_defaults": [0, 1, 1], "costs": {"approve_default": 8, "decline_good": 2, "manual_review": 0.25}})
    assert response.status_code == 200
    payload = response.json()
    assert payload["expected_cost"] == pytest.approx(0.25)
    assert payload["false_negative_rate"] == pytest.approx(0.5)
    assert payload["decision_costs"]["approve_default"] == 8


def test_policy_simulation_can_report_audit_only_fairness_metrics():
    response = TestClient(app).post("/policy/simulate", json={"probabilities": [0.1, 0.3, 0.7], "approve_max_risk": 0.2, "decline_min_risk": 0.5, "actual_defaults": [0, 1, 1], "audit_groups": {"audit_group": ["A", "A", "B"]}})
    assert response.status_code == 200
    payload = response.json()
    assert "fairness" in payload
    assert payload["fairness"]["audit_group"]["A"]["approval_rate"] == pytest.approx(0.5)


def test_policy_simulation_requires_labels_for_audit_groups():
    response = TestClient(app).post("/policy/simulate", json={"probabilities": [0.1], "approve_max_risk": 0.2, "decline_min_risk": 0.5, "audit_groups": {"audit_group": ["A"]}})
    assert response.status_code == 422


def test_current_policy_endpoint_separates_policy_from_model():
    response = TestClient(app).get("/policy/current")
    assert response.status_code == 200
    payload = response.json()
    assert payload["policy_version"] == "policy-0.1.0"
    assert payload["decision_thresholds"]["approve_max_risk"] < payload["decision_thresholds"]["decline_min_risk"]
    assert "low" in payload["risk_bands"]


def test_monitoring_drift_endpoint():
    response = TestClient(app).post("/monitoring/drift", json={"reference_records": [{"x": 1}, {"x": 2}], "current_records": [{"x": 100}, {"x": 100}], "features": ["x"]})
    assert response.status_code == 200
    assert response.json()["metrics"]["x"]["psi"] > 0


def test_monitoring_prediction_endpoint():
    response = TestClient(app).post("/monitoring/predictions", json={"reference_probabilities": [0.1, 0.2], "current_probabilities": [0.8, 0.9], "reference_decisions": ["approve", "manual_review"], "current_decisions": ["decline", "decline"]})
    assert response.status_code == 200
    assert response.json()["prediction_distribution"]["psi"] > 0
    assert "decline" in response.json()["decision_rates"]


def test_monitoring_prediction_endpoint_accepts_labels():
    response = TestClient(app).post("/monitoring/predictions", json={"reference_probabilities": [0.1, 0.2], "current_probabilities": [0.8, 0.9], "reference_defaults": [0, 0], "current_defaults": [1, 1]})
    assert response.status_code == 200
    assert response.json()["label_metrics"]["default_rate_delta"] == 1.0


def test_monitoring_fairness_endpoint():
    response = TestClient(app).post("/monitoring/fairness", json={"reference_metrics": {"group": {"A": {"approval_rate": 0.8}}}, "current_metrics": {"group": {"A": {"approval_rate": 0.6}}}})
    assert response.status_code == 200
    assert response.json()["metrics"]["group.A.approval_rate"]["level"] == "critical"
    history = TestClient(app).get("/monitoring/fairness/history")
    assert history.status_code == 200
    assert history.json()[0]["experiment_id"]


def test_monitoring_drift_history_endpoint(monkeypatch, tmp_path):
    from credit_simulator.drift_events import DriftEventStore

    monkeypatch.setattr(api, "drift_store", DriftEventStore(tmp_path / "drift.db"))
    client = TestClient(app)
    response = client.post("/monitoring/drift", json={"reference_records": [{"x": 1}], "current_records": [{"x": 2}], "features": ["x"]})
    assert response.status_code == 200
    history = client.get("/monitoring/drift/history")
    assert history.status_code == 200
    assert history.json()[0]["current_rows"] == 1
    assert history.json()[0]["model_version"]
    assert history.json()[0]["policy_version"] == "policy-0.1.0"


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

