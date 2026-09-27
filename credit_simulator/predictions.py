from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class PredictionEventStore:
    def __init__(self, path: str | Path = "artifacts/predictions.db"):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS prediction_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
                request_id TEXT, dataset TEXT NOT NULL, model_version TEXT NOT NULL,
                experiment_id TEXT NOT NULL DEFAULT 'unknown',
                policy_version TEXT NOT NULL DEFAULT 'unknown',
                dataset_version TEXT NOT NULL DEFAULT 'unknown',
                risk_probability REAL NOT NULL, credit_score INTEGER NOT NULL,
                risk_band TEXT NOT NULL, decision TEXT NOT NULL
            )""")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(prediction_events)").fetchall()}
            if "policy_version" not in columns:
                connection.execute("ALTER TABLE prediction_events ADD COLUMN policy_version TEXT NOT NULL DEFAULT 'unknown'")
            if "experiment_id" not in columns:
                connection.execute("ALTER TABLE prediction_events ADD COLUMN experiment_id TEXT NOT NULL DEFAULT 'unknown'")
            if "dataset_version" not in columns:
                connection.execute("ALTER TABLE prediction_events ADD COLUMN dataset_version TEXT NOT NULL DEFAULT 'unknown'")

    def record(self, dataset: str, model_version: str, risk_probability: float, credit_score: int, risk_band: str, decision: str, request_id: str | None = None, policy_version: str = "unknown", dataset_version: str = "unknown", experiment_id: str = "unknown") -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute("INSERT INTO prediction_events (created_at, request_id, dataset, model_version, experiment_id, policy_version, dataset_version, risk_probability, credit_score, risk_band, decision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (datetime.now(timezone.utc).isoformat(), request_id, dataset, model_version, experiment_id, policy_version, dataset_version, risk_probability, credit_score, risk_band, decision))

    def summary(self) -> dict:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute("SELECT dataset, model_version, experiment_id, policy_version, dataset_version, decision, COUNT(*) FROM prediction_events GROUP BY dataset, model_version, experiment_id, policy_version, dataset_version, decision ORDER BY dataset, model_version, experiment_id, policy_version, dataset_version, decision").fetchall()
            total = connection.execute("SELECT COUNT(*) FROM prediction_events").fetchone()[0]
        return {"total_predictions": total, "groups": [{"dataset": row[0], "model_version": row[1], "experiment_id": row[2], "policy_version": row[3], "dataset_version": row[4], "decision": row[5], "count": row[6]} for row in rows], "raw_inputs_stored": False}
