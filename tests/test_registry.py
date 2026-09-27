import json
import hashlib

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


def test_registry_reports_artifact_integrity(tmp_path):
    path = tmp_path / "registry.json"
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    model = artifact / "model.joblib"
    model.write_bytes(b"model")
    checksum = hashlib.sha256(b"model").hexdigest()
    (artifact / "metadata.json").write_text(json.dumps({"model_sha256": checksum}), encoding="utf-8")
    register_model({"dataset": "test", "model_version": "2", "model_sha256": checksum}, artifact, path)
    listed = list_models(path)[0]
    assert listed["model_sha256"] == checksum
    assert listed["artifact_available"] is True
    assert listed["checksum_valid"] is True
    model.write_bytes(b"changed")
    assert list_models(path)[0]["checksum_valid"] is False
