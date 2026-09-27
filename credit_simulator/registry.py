from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _write_registry(path: Path, entries: list[dict]) -> None:
    """Replace registry JSON atomically within the same filesystem."""
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary_path = handle.name
            json.dump(entries, handle, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def register_model(metadata: dict, artifact_dir: str | Path, registry_path: str | Path = "artifacts/model_registry.json") -> dict:
    registry_path = Path(registry_path); registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        try:
            entries = json.loads(registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            entries = []
        if not isinstance(entries, list):
            entries = []
    else:
        entries = []
    artifact = str(Path(artifact_dir).as_posix())
    dataset = metadata.get("dataset", "unknown")
    dataset_name = dataset.get("name", "unknown") if isinstance(dataset, dict) else dataset
    entry = {"dataset": dataset_name, "model_version": metadata.get("model_version", "unknown"), "policy_version": metadata.get("policy_version", "unknown"), "training_config_sha256": metadata.get("training_config_sha256", "unknown"), "artifact_fingerprint": metadata.get("artifact_fingerprint"), "model_sha256": metadata.get("model_sha256"), "dataset_sha256": metadata.get("dataset_sha256", dataset.get("sha256") if isinstance(dataset, dict) else None), "artifact_dir": artifact, "selected_model": metadata.get("selected_model", "generic"), "metrics_test": metadata.get("metrics_test", {}), "registered_at": datetime.now(timezone.utc).isoformat(), "status": "available"}
    entries = [item for item in entries if not (item.get("artifact_dir") == artifact and item.get("model_version") == entry["model_version"])]
    entries.append(entry)
    _write_registry(registry_path, entries)
    return entry


def list_models(registry_path: str | Path = "artifacts/model_registry.json") -> list[dict]:
    path = Path(registry_path)
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(entries, list):
        return []
    enriched = []
    for entry in entries:
        item = dict(entry)
        item.setdefault("policy_version", "unknown")
        item.setdefault("training_config_sha256", "unknown")
        artifact_dir = Path(item.get("artifact_dir", ""))
        model_path, metadata_path = artifact_dir / "model.joblib", artifact_dir / "metadata.json"
        item["artifact_available"] = model_path.exists() and metadata_path.exists()
        item["checksum_valid"] = False
        if item["artifact_available"]:
            try:
                item["checksum_valid"] = bool(item.get("model_sha256")) and item["model_sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
            except (OSError, ValueError, json.JSONDecodeError):
                item["checksum_valid"] = False
        enriched.append(item)
    return enriched


def promote_model(model_version: str, serving_dir: str | Path, registry_path: str | Path = "artifacts/model_registry.json", dataset: str | None = None) -> dict:
    """Promote one verified immutable snapshot into the serving directory.

    This is deliberately an explicit operator action. Promoting an older
    version is the rollback mechanism; training never calls this function.
    """
    registry_path = Path(registry_path)
    try:
        entries = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("model registry is unavailable or invalid") from exc
    if not isinstance(entries, list):
        raise ValueError("model registry must contain a list")
    matches = [entry for entry in entries if entry.get("model_version") == model_version and (dataset is None or entry.get("dataset") == dataset)]
    if len(matches) != 1:
        raise ValueError("model version is not uniquely registered; specify a dataset or check the registry")
    selected = matches[0]
    source_dir = Path(selected.get("artifact_dir", ""))
    source_model, source_metadata = source_dir / "model.joblib", source_dir / "metadata.json"
    if not source_model.exists() or not source_metadata.exists() or not selected.get("model_sha256"):
        raise ValueError("selected model snapshot is unavailable or missing a checksum")
    if hashlib.sha256(source_model.read_bytes()).hexdigest() != selected["model_sha256"]:
        raise ValueError("selected model snapshot failed checksum validation")
    serving = Path(serving_dir); serving.mkdir(parents=True, exist_ok=True)
    for source, destination in ((source_model, serving / "model.joblib"), (source_metadata, serving / "metadata.json")):
        temporary = destination.with_suffix(destination.suffix + ".promoting")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    for entry in entries:
        if dataset is None or entry.get("dataset") == dataset:
            entry["status"] = "active" if entry is selected else "retired"
    _write_registry(registry_path, entries)
    return selected
