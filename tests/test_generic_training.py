import pandas as pd

from credit_simulator.datasets import DatasetBundle
from credit_simulator.generic_training import train_tabular


def test_generic_trainer_handles_mixed_schema(tmp_path):
    frame = pd.DataFrame({"income": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], "purpose": ["car", "home"] * 5, "default": [0, 1] * 5})
    bundle = DatasetBundle("synthetic", frame, "default", [], ["income", "purpose"], {"source_url": "test"})
    metadata = train_tabular(bundle, tmp_path / "artifact")
    assert (tmp_path / "artifact" / "model.joblib").exists()
    assert metadata["metrics_test"]["roc_auc"] >= 0
