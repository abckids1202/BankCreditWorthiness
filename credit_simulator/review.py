from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


class ReviewStore:
    def __init__(self, path: str | Path = "artifacts/reviews.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS review_cases (
                case_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                applicant_json TEXT NOT NULL, model_version TEXT NOT NULL, policy_version TEXT NOT NULL DEFAULT 'unknown',
                risk_probability REAL NOT NULL,
                credit_score INTEGER NOT NULL, automatic_decision TEXT NOT NULL, reason_codes_json TEXT NOT NULL,
                warnings_json TEXT NOT NULL, reviewer_decision TEXT, reviewer_note TEXT, reviewer_id TEXT, reviewed_at TEXT
            )""")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(review_cases)").fetchall()}
            if "policy_version" not in columns:
                connection.execute("ALTER TABLE review_cases ADD COLUMN policy_version TEXT NOT NULL DEFAULT 'unknown'")
            if "reviewer_id" not in columns:
                connection.execute("ALTER TABLE review_cases ADD COLUMN reviewer_id TEXT")
            connection.execute("""CREATE TABLE IF NOT EXISTS review_events (
                event_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, event_type TEXT NOT NULL,
                created_at TEXT NOT NULL, reviewer_decision TEXT, reviewer_note TEXT, reviewer_id TEXT
            )""")
            event_columns = {row[1] for row in connection.execute("PRAGMA table_info(review_events)").fetchall()}
            if "reviewer_id" not in event_columns:
                connection.execute("ALTER TABLE review_events ADD COLUMN reviewer_id TEXT")
            legacy_cases = connection.execute(
                """SELECT c.case_id, c.created_at FROM review_cases AS c
                   LEFT JOIN review_events AS e ON e.case_id = c.case_id
                   WHERE e.case_id IS NULL"""
            ).fetchall()
            for case in legacy_cases:
                connection.execute(
                    "INSERT INTO review_events (event_id, case_id, event_type, created_at, reviewer_decision, reviewer_note, reviewer_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), case[0], "created", case[1], None, None, None),
                )

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _serialize(row: sqlite3.Row) -> dict:
        result = dict(row)
        for key in ("applicant_json", "reason_codes_json", "warnings_json"):
            result[key.removesuffix("_json")] = json.loads(result.pop(key))
        return result

    def create(self, applicant: dict, prediction: dict) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        case_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO review_cases (
                    case_id, created_at, updated_at, applicant_json, model_version, policy_version,
                    risk_probability, credit_score, automatic_decision, reason_codes_json,
                    warnings_json, reviewer_decision, reviewer_note, reviewer_id, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    case_id, now, now, json.dumps(applicant), prediction["model_version"],
                    prediction.get("policy_version", "unknown"), prediction["risk_probability"],
                    prediction["credit_score"], prediction["decision"], json.dumps(prediction["reason_codes"]),
                    json.dumps(prediction.get("warnings", [])), None, None, None, None,
                ),
            )
            connection.execute(
                "INSERT INTO review_events (event_id, case_id, event_type, created_at, reviewer_decision, reviewer_note, reviewer_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), case_id, "created", now, None, None, None),
            )
        return self.get(case_id)

    def get(self, case_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM review_cases WHERE case_id = ?", (case_id,)).fetchone()
        return self._serialize(row) if row else None

    def list(self, limit: int = 50) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM review_cases ORDER BY created_at DESC LIMIT ?", (min(max(limit, 1), 200),)).fetchall()
        return [self._serialize(row) for row in rows]

    def history(self, case_id: str) -> list[dict] | None:
        with self._connect() as connection:
            exists = connection.execute("SELECT 1 FROM review_cases WHERE case_id = ?", (case_id,)).fetchone()
            if not exists:
                return None
            rows = connection.execute(
                "SELECT event_id, case_id, event_type, created_at, reviewer_decision, reviewer_note, reviewer_id FROM review_events WHERE case_id = ? ORDER BY created_at ASC, event_id ASC",
                (case_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update(self, case_id: str, reviewer_decision: str | None, reviewer_note: str | None, reviewer_id: str | None = None) -> dict | None:
        allowed = {"approved", "declined", "needs_more_information", "escalated"}
        if reviewer_decision is not None and reviewer_decision not in allowed:
            raise ValueError(f"reviewer_decision must be one of {sorted(allowed)}")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            exists = connection.execute("SELECT 1 FROM review_cases WHERE case_id = ?", (case_id,)).fetchone()
            if not exists:
                return None
            has_review_update = reviewer_decision is not None or reviewer_note is not None or reviewer_id is not None
            connection.execute("UPDATE review_cases SET reviewer_decision = COALESCE(?, reviewer_decision), reviewer_note = COALESCE(?, reviewer_note), reviewer_id = COALESCE(?, reviewer_id), reviewed_at = COALESCE(reviewed_at, ?), updated_at = ? WHERE case_id = ?", (reviewer_decision, reviewer_note, reviewer_id, now if has_review_update else None, now, case_id))
            if has_review_update:
                connection.execute(
                    "INSERT INTO review_events (event_id, case_id, event_type, created_at, reviewer_decision, reviewer_note, reviewer_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), case_id, "updated", now, reviewer_decision, reviewer_note, reviewer_id),
                )
        return self.get(case_id)
