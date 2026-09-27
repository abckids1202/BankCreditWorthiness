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


def test_drift_flags_missingness_change_even_when_values_are_stable():
    report = drift_report(
        [{"x": 1}, {"x": 1}, {"x": 1}, {"x": 1}],
        [{"x": None}, {"x": None}, {"x": 1}, {"x": 1}],
        ["x"],
    )
    assert report["metrics"]["x"]["psi"] == 0
    assert report["metrics"]["x"]["level"] == "critical"
    assert report["metrics"]["x"]["level_reasons"] == ["missingness_critical"]


def test_drift_uses_custom_thresholds_and_reports_them():
    report = drift_report([{"x": 0}, {"x": 1}], [{"x": 2}, {"x": 2}], ["x"], psi_warning=0.01, psi_critical=0.02, missing_warning=0.01, missing_critical=0.02)
    assert report["thresholds"]["psi_critical"] == 0.02
    assert report["metrics"]["x"]["level"] == "critical"


def test_drift_rejects_invalid_threshold_order():
    with pytest.raises(ValueError, match="lower"):
        drift_report([{"x": 1}], [{"x": 2}], ["x"], psi_warning=0.3, psi_critical=0.2)
