import pytest

from credit_simulator.training import _approval_rate_report, _bootstrap_intervals, _candidate_specs, _data_quality, _plots


def test_approval_rate_report_measures_defaults_not_approved():
    rows = _approval_rate_report([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], target_rates=(0.5,))
    assert rows[0]["achieved_approval_rate"] == pytest.approx(0.5)
    assert rows[0]["default_rate_approved"] == 0
    assert rows[0]["default_recall_not_approved"] == 1


def test_approval_rate_report_returns_expected_operating_points():
    rows = _approval_rate_report([0, 1, 0, 1], [0.1, 0.2, 0.8, 0.9], target_rates=(0.5, 0.9))
    assert len(rows) == 2
    assert rows[0]["approved_count"] == 2
    assert rows[1]["target_approval_rate"] == pytest.approx(0.9)


def test_bootstrap_intervals_are_reproducible_and_bounded():
    y = [0, 1] * 20
    probabilities = [0.1, 0.9] * 20
    first = _bootstrap_intervals(y, probabilities, random_state=7, n_bootstrap=40)
    second = _bootstrap_intervals(y, probabilities, random_state=7, n_bootstrap=40)
    assert first == second
    assert first["successful_samples"] > 0
    assert 0 <= first["metrics"]["roc_auc"]["lower_95"] <= first["metrics"]["roc_auc"]["upper_95"] <= 1


def test_candidate_specs_cover_baseline_families_and_calibration_variants():
    specs = _candidate_specs()
    names = {spec["name"] for spec in specs}
    assert names == {
        "logistic_regression_uncalibrated",
        "logistic_regression_calibrated",
        "gradient_boosting_uncalibrated",
        "gradient_boosting_calibrated",
    }
    assert sum(bool(spec["calibrated"]) for spec in specs) == 2


def test_data_quality_report_includes_ids_targets_outliers_and_categorical_summary():
    frame = __import__("pandas").DataFrame({
        "ID": [1, 1, 3, 4],
        "LIMIT_BAL": [100, 100, 100, 10_000],
        "SEX": [1, 1, 2, 2],
        "default": [0, 1, 0, 0],
    })
    quality = _data_quality(frame, ["LIMIT_BAL"])
    assert quality["duplicate_id_rows"] == 1
    assert quality["target_values"] == [0, 1]
    assert quality["target_outside_binary_count"] == 0
    assert quality["numeric_outlier_counts_iqr"]["LIMIT_BAL"] == 1
    assert quality["categorical_summary"]["SEX"]["unique_values"] == 2


def test_feature_distribution_plots_cover_every_numeric_feature(tmp_path):
    frame = __import__("pandas").DataFrame({"LIMIT_BAL": [10, 20, 30, 40], "engineered/ratio": [0.1, 0.2, 0.3, 0.4], "default": [0, 1, 0, 1]})
    _plots(frame, frame["default"], __import__("numpy").array([0.1, 0.8, 0.2, 0.7]), tmp_path, features=["LIMIT_BAL", "engineered/ratio"])
    assert (tmp_path / "feature_distributions.png").exists()
    assert (tmp_path / "feature_distribution_LIMIT_BAL.png").exists()
    assert (tmp_path / "feature_distribution_engineered_ratio.png").exists()
