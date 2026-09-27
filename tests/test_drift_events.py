from credit_simulator.drift_events import DriftEventStore


def test_drift_event_store_persists_aggregate_only(tmp_path):
    store = DriftEventStore(tmp_path / "drift.db")
    store.record({"reference_rows": 10, "current_rows": 8, "features_checked": ["x"], "warning_features": ["x"], "critical_features": [], "thresholds": {"psi_warning": 0.1}, "recommended_action": "Investigate drift"}, "request-1", "model-1", "policy-1", "exp-1")
    event = store.list()[0]
    assert event["request_id"] == "request-1"
    assert event["features_checked"] == ["x"]
    assert event["model_version"] == "model-1"
    assert event["policy_version"] == "policy-1"
    assert event["experiment_id"] == "exp-1"
    assert "reference_records" not in event


def test_drift_event_store_persists_fairness_summary_only(tmp_path):
    store = DriftEventStore(tmp_path / "fairness.db")
    store.record_fairness({"metrics": {"group.A.approval_rate": {"delta": 0.2}}, "warning_metrics": [], "critical_metrics": ["group.A.approval_rate"], "missing_from_current": [], "new_in_current": [], "thresholds": {"delta_critical": 0.15}, "recommended_action": "Investigate"}, "request-2", "model-2", "policy-2", "exp-2")
    event = store.list_fairness()[0]
    assert event["metric_count"] == 1
    assert event["critical_metrics"] == ["group.A.approval_rate"]
    assert event["experiment_id"] == "exp-2"
    assert "metrics" not in event
