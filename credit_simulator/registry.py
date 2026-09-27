from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


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
    entry = {"dataset": dataset_name, "model_version": metadata.get("model_version", "unknown"), "artifact_fingerprint": metadata.get("artifact_fingerprint"), "dataset_sha256": metadata.get("dataset_sha256", dataset.get("sha256") if isinstance(dataset, dict) else None), "artifact_dir": artifact, "selected_model": metadata.get("selected_model", "generic"), "metrics_test": metadata.get("metrics_test", {}), "registered_at": datetime.now(timezone.utc).isoformat(), "status": "available"}
    entries = [item for item in entries if not (item.get("artifact_dir") == artifact and item.get("model_version") == entry["model_version"])]
    entries.append(entry)
    registry_path.write_text(json.dumps(entries, indent=2, default=str), encoding="utf-8")
    return entry


def list_models(registry_path: str | Path = "artifacts/model_registry.json") -> list[dict]:
    path = Path(registry_path)
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return entries if isinstance(entries, list) else []
