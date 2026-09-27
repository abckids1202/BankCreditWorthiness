from credit_simulator.monitoring import drift_report
import pytest


def test_drift_report_flags_distribution_change():
    report = drift_report([{"x": 1}, {"x": 1}, {"x": 2}], [{"x": 100}, {"x": 100}, {"x": 100}], ["x"])
    assert report["metrics"]["x"]["psi"] > 0
    assert report["automatic_retraining"] is False


def test_drift_counts_values_outside_reference_range():
    report = drift_report([{"x": 0}, {"x": 1}, {"x": 2}], [{"x": 100}, {"x": 100}, {"x": 100}], ["x"])
    assert report["metrics"]["x"]["current_missing_rate"] == 0
    assert report["metrics"]["x"]["psi"] > 0.25
    assert "x" in report["critical_features"]


def test_drift_rejects_explicit_missing_feature():
    with pytest.raises(ValueError, match="missing"):
        drift_report([{"x": 1}], [{"x": 2}], ["unknown"])
