import pandas as pd
import pytest

from credit_simulator.datasets import DatasetBundle
from credit_simulator.generic_training import train_tabular


def test_generic_trainer_handles_mixed_schema(tmp_path):
    frame = pd.DataFrame({"income": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], "purpose": ["car", "home"] * 5, "default": [0, 1] * 5})
    bundle = DatasetBundle("synthetic", frame, "default", [], ["income", "purpose"], {"source_url": "test"})
    metadata = train_tabular(bundle, tmp_path / "artifact")
    assert (tmp_path / "artifact" / "model.joblib").exists()
    assert (tmp_path / "artifact" / "versions" / metadata["artifact_fingerprint"] / "model.joblib").exists()
    assert (tmp_path / "artifact" / "versions" / metadata["artifact_fingerprint"] / "metadata.json").exists()
    assert metadata["metrics_test"]["roc_auc"] >= 0


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
