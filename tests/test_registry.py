import json

from credit_simulator.registry import list_models, register_model


def test_registry_registers_and_deduplicates(tmp_path):
    path = tmp_path / "registry.json"
    metadata = {"dataset": "test", "model_version": "1", "metrics_test": {"roc_auc": 0.8}}
    register_model(metadata, tmp_path / "artifact", path)
    register_model(metadata, tmp_path / "artifact", path)
    assert len(list_models(path)) == 1
    assert json.loads(path.read_text(encoding="utf-8"))[0]["status"] == "available"
