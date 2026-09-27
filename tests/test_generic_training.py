import pandas as pd
import pytest

from credit_simulator.datasets import DatasetBundle
from credit_simulator.generic_training import _split_id_overlap, train_tabular


def test_generic_trainer_handles_mixed_schema(tmp_path):
    frame = pd.DataFrame({"income": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], "purpose": ["car", "home"] * 5, "default": [0, 1] * 5})
    bundle = DatasetBundle("synthetic", frame, "default", [], ["income", "purpose"], {"source_url": "test"})
    metadata = train_tabular(bundle, tmp_path / "artifact")
    assert (tmp_path / "artifact" / "model.joblib").exists()
    assert (tmp_path / "artifact" / "versions" / metadata["artifact_fingerprint"] / "model.joblib").exists()
    assert (tmp_path / "artifact" / "versions" / metadata["artifact_fingerprint"] / "metadata.json").exists()
    assert (tmp_path / "artifact" / "training_report.json").exists()
    assert metadata["dataset_summary"]["default_rate"] == 0.5
    assert metadata["metrics_test"]["roc_auc"] >= 0
    assert metadata["metrics_validation"]["brier_score"] >= 0
    assert metadata["feature_summary"]["income"]["type"] == "numeric"
    assert metadata["feature_summary"]["purpose"]["rare_category_count"] == 0
    assert metadata["dataset_summary"]["target_definition"] == "default"
    assert "license" in metadata["dataset_summary"]
    assert metadata["training_timestamp"]
    assert len(metadata["training_config_sha256"]) == 64
    assert metadata["runtime"]["scikit_learn"]
    assert metadata["schema_version"] == "1.0"
    assert metadata["experiment_id"].startswith("exp-")
    assert len(metadata["threshold_analysis"]) == 19
    assert metadata["calibration"]["brier_score"] >= 0
    assert (tmp_path / "artifact" / "threshold_analysis.csv").exists()
    assert (tmp_path / "artifact" / "calibration.json").exists()
    assert (tmp_path / "artifact" / "fairness.json").exists()
    manifest = __import__("json").loads((tmp_path / "artifact" / "report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == metadata["experiment_id"]
    assert "training_report.json" in manifest["files"]


def test_generic_trainer_version_is_reproducible(tmp_path):
    frame = pd.DataFrame({"income": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], "purpose": ["car", "home"] * 5, "default": [0, 1] * 5})
    bundle = DatasetBundle("synthetic", frame, "default", [], ["income", "purpose"], {"source_url": "test"})
    first = train_tabular(bundle, tmp_path / "first")
    second = train_tabular(bundle, tmp_path / "second")
    assert first["model_version"] == second["model_version"]
    assert first["artifact_fingerprint"] == second["artifact_fingerprint"]


def test_generic_trainer_rejects_target_leakage(tmp_path):
    frame = pd.DataFrame({"income": [10, 20, 30, 40], "target": [0, 1, 0, 1], "default": [0, 1, 0, 1]})
    bundle = DatasetBundle("synthetic", frame, "default", [], ["income", "target"], {"source_url": "test"})
    with pytest.raises(ValueError, match="leakage"):
        train_tabular(bundle, tmp_path / "artifact")


def test_generic_trainer_rejects_non_binary_target(tmp_path):
    frame = pd.DataFrame({"income": [10, 20, 30, 40], "default": [0, 1, 2, 1]})
    bundle = DatasetBundle("synthetic", frame, "default", [], ["income"], {"source_url": "test"})
    with pytest.raises(ValueError, match="binary"):
        train_tabular(bundle, tmp_path / "artifact")


def test_generic_trainer_uses_temporal_split_when_adapter_declares_time_column(tmp_path):
    frame = pd.DataFrame({
        "application_date": pd.date_range("2024-01-01", periods=20, freq="D"),
        "income": list(range(20)),
        "default": [0, 1] * 10,
    })
    bundle = DatasetBundle("temporal", frame, "default", [], ["income"], {"source_url": "test", "time_column": "application_date"})
    metadata = train_tabular(bundle, tmp_path / "artifact")
    assert metadata["split_strategy"] == "temporal"
    assert metadata["dataset_summary"]["validation_rows"] == 4
    assert metadata["dataset_summary"]["test_rows"] == 4
    assert metadata["metrics_validation"]["roc_auc"] >= 0
    assert metadata["split_comparison"]["serving_split"] == "temporal"
    assert metadata["split_comparison"]["random_split"]["roc_auc"] >= 0


def test_split_id_overlap_reports_cross_split_duplicates_and_ignores_missing_ids():
    train = pd.DataFrame({"application_id": [1, 2, None]})
    validation = pd.DataFrame({"application_id": [2, 3, None]})
    test = pd.DataFrame({"application_id": [4, 1, None]})
    assert _split_id_overlap(train, validation, test, "application_id") == {"train_validation": 1, "train_test": 1, "validation_test": 0}
