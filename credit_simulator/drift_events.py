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
                recommended_action TEXT NOT NULL
            )""")

    def record(self, report: dict, request_id: str | None = None) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """INSERT INTO drift_events (
                    created_at, request_id, reference_rows, current_rows, features_checked_json,
                    warning_features_json, critical_features_json, thresholds_json, recommended_action
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(), request_id, report["reference_rows"], report["current_rows"],
                    json.dumps(report["features_checked"]), json.dumps(report["warning_features"]),
                    json.dumps(report["critical_features"]), json.dumps(report["thresholds"]), report["recommended_action"],
                ),
            )

    def list(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                """SELECT event_id, created_at, request_id, reference_rows, current_rows,
                   features_checked_json, warning_features_json, critical_features_json,
                   thresholds_json, recommended_action FROM drift_events
                   ORDER BY created_at DESC LIMIT ?""",
                (min(max(limit, 1), 200),),
            ).fetchall()
        columns = ("event_id", "created_at", "request_id", "reference_rows", "current_rows", "features_checked_json", "warning_features_json", "critical_features_json", "thresholds_json", "recommended_action")
        result = []
        for row in rows:
            item = dict(zip(columns, row))
            for key in ("features_checked", "warning_features", "critical_features", "thresholds"):
                item[key] = json.loads(item.pop(f"{key}_json"))
            result.append(item)
        return result
