import json

from credit_simulator.registry import list_models, register_model


def test_registry_registers_and_deduplicates(tmp_path):
    path = tmp_path / "registry.json"
    metadata = {"dataset": "test", "model_version": "1", "metrics_test": {"roc_auc": 0.8}}
    register_model(metadata, tmp_path / "artifact", path)
    register_model(metadata, tmp_path / "artifact", path)
    assert len(list_models(path)) == 1
    assert json.loads(path.read_text(encoding="utf-8"))[0]["status"] == "available"


def test_registry_keeps_fingerprints_and_handles_corrupt_file(tmp_path):
    path = tmp_path / "registry.json"
    metadata = {"dataset": {"name": "test", "sha256": "dataset-hash"}, "model_version": "1+abc", "artifact_fingerprint": "abc", "metrics_test": {}}
    entry = register_model(metadata, tmp_path / "artifact", path)
    assert entry["artifact_fingerprint"] == "abc"
    assert entry["dataset_sha256"] == "dataset-hash"
    path.write_text("not-json", encoding="utf-8")
    assert list_models(path) == []
