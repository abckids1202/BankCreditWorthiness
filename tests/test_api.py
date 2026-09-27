from fastapi.testclient import TestClient

from credit_simulator.api import app


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


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

