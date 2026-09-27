from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class DriftEventStore:
    """Persist aggregate drift diagnostics without storing applicant records."""

    def __init__(self, path: str | Path = "artifacts/drift_events.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS drift_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
                request_id TEXT, reference_rows INTEGER NOT NULL, current_rows INTEGER NOT NULL,
                features_checked_json TEXT NOT NULL, warning_features_json TEXT NOT NULL,
                critical_features_json TEXT NOT NULL, thresholds_json TEXT NOT NULL,
                recommended_action TEXT NOT NULL, model_version TEXT NOT NULL DEFAULT 'unknown',
                policy_version TEXT NOT NULL DEFAULT 'unknown', experiment_id TEXT NOT NULL DEFAULT 'unknown'
            )""")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(drift_events)").fetchall()}
            if "model_version" not in columns:
                connection.execute("ALTER TABLE drift_events ADD COLUMN model_version TEXT NOT NULL DEFAULT 'unknown'")
            if "policy_version" not in columns:
                connection.execute("ALTER TABLE drift_events ADD COLUMN policy_version TEXT NOT NULL DEFAULT 'unknown'")
            if "experiment_id" not in columns:
                connection.execute("ALTER TABLE drift_events ADD COLUMN experiment_id TEXT NOT NULL DEFAULT 'unknown'")
            connection.execute("""CREATE TABLE IF NOT EXISTS fairness_drift_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, request_id TEXT,
                metric_count INTEGER NOT NULL, warning_metrics_json TEXT NOT NULL, critical_metrics_json TEXT NOT NULL,
                missing_metrics_json TEXT NOT NULL, new_metrics_json TEXT NOT NULL, thresholds_json TEXT NOT NULL,
                recommended_action TEXT NOT NULL, model_version TEXT NOT NULL DEFAULT 'unknown',
                policy_version TEXT NOT NULL DEFAULT 'unknown', experiment_id TEXT NOT NULL DEFAULT 'unknown'
            )""")
            fairness_columns = {row[1] for row in connection.execute("PRAGMA table_info(fairness_drift_events)").fetchall()}
            if "experiment_id" not in fairness_columns:
                connection.execute("ALTER TABLE fairness_drift_events ADD COLUMN experiment_id TEXT NOT NULL DEFAULT 'unknown'")

    def record(self, report: dict, request_id: str | None = None, model_version: str = "unknown", policy_version: str = "unknown", experiment_id: str = "unknown") -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """INSERT INTO drift_events (
                    created_at, request_id, reference_rows, current_rows, features_checked_json,
                    warning_features_json, critical_features_json, thresholds_json, recommended_action,
                    model_version, policy_version, experiment_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(), request_id, report["reference_rows"], report["current_rows"],
                    json.dumps(report["features_checked"]), json.dumps(report["warning_features"]),
                    json.dumps(report["critical_features"]), json.dumps(report["thresholds"]), report["recommended_action"], model_version, policy_version, experiment_id,
                ),
            )

    def list(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                """SELECT event_id, created_at, request_id, reference_rows, current_rows,
                   features_checked_json, warning_features_json, critical_features_json,
                   thresholds_json, recommended_action, model_version, policy_version, experiment_id FROM drift_events
                   ORDER BY created_at DESC LIMIT ?""",
                (min(max(limit, 1), 200),),
            ).fetchall()
        columns = ("event_id", "created_at", "request_id", "reference_rows", "current_rows", "features_checked_json", "warning_features_json", "critical_features_json", "thresholds_json", "recommended_action", "model_version", "policy_version", "experiment_id")
        result = []
        for row in rows:
            item = dict(zip(columns, row))
            for key in ("features_checked", "warning_features", "critical_features", "thresholds"):
                item[key] = json.loads(item.pop(f"{key}_json"))
            result.append(item)
        return result

    def record_fairness(self, report: dict, request_id: str | None = None, model_version: str = "unknown", policy_version: str = "unknown", experiment_id: str = "unknown") -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """INSERT INTO fairness_drift_events (
                    created_at, request_id, metric_count, warning_metrics_json, critical_metrics_json,
                    missing_metrics_json, new_metrics_json, thresholds_json, recommended_action,
                    model_version, policy_version, experiment_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now(timezone.utc).isoformat(), request_id, len(report["metrics"]), json.dumps(report["warning_metrics"]), json.dumps(report["critical_metrics"]), json.dumps(report["missing_from_current"]), json.dumps(report["new_in_current"]), json.dumps(report["thresholds"]), report["recommended_action"], model_version, policy_version, experiment_id),
            )

    def list_fairness(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute("SELECT event_id, created_at, request_id, metric_count, warning_metrics_json, critical_metrics_json, missing_metrics_json, new_metrics_json, thresholds_json, recommended_action, model_version, policy_version, experiment_id FROM fairness_drift_events ORDER BY created_at DESC LIMIT ?", (min(max(limit, 1), 200),)).fetchall()
        columns = ("event_id", "created_at", "request_id", "metric_count", "warning_metrics_json", "critical_metrics_json", "missing_metrics_json", "new_metrics_json", "thresholds_json", "recommended_action", "model_version", "policy_version", "experiment_id")
        result = []
        for row in rows:
            item = dict(zip(columns, row))
            for key in ("warning_metrics", "critical_metrics", "missing_from_current", "new_in_current", "thresholds"):
                source = "missing_metrics_json" if key == "missing_from_current" else "new_metrics_json" if key == "new_in_current" else f"{key}_json"
                item[key] = json.loads(item.pop(source))
            result.append(item)
        return result
