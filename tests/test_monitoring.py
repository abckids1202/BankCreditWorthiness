from credit_simulator.monitoring import drift_report


def test_drift_report_flags_distribution_change():
    report = drift_report([{"x": 1}, {"x": 1}, {"x": 2}], [{"x": 100}, {"x": 100}, {"x": 100}], ["x"])
    assert report["metrics"]["x"]["psi"] > 0
    assert report["automatic_retraining"] is False
