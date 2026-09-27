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
                applicant_json TEXT NOT NULL, model_version TEXT NOT NULL, risk_probability REAL NOT NULL,
                credit_score INTEGER NOT NULL, automatic_decision TEXT NOT NULL, reason_codes_json TEXT NOT NULL,
                warnings_json TEXT NOT NULL, reviewer_decision TEXT, reviewer_note TEXT, reviewed_at TEXT
            )""")

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
            connection.execute("INSERT INTO review_cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (case_id, now, now, json.dumps(applicant), prediction["model_version"], prediction["risk_probability"], prediction["credit_score"], prediction["decision"], json.dumps(prediction["reason_codes"]), json.dumps(prediction.get("warnings", [])), None, None, None))
        return self.get(case_id)

    def get(self, case_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM review_cases WHERE case_id = ?", (case_id,)).fetchone()
        return self._serialize(row) if row else None

    def list(self, limit: int = 50) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM review_cases ORDER BY created_at DESC LIMIT ?", (min(max(limit, 1), 200),)).fetchall()
        return [self._serialize(row) for row in rows]

    def update(self, case_id: str, reviewer_decision: str | None, reviewer_note: str | None) -> dict | None:
        allowed = {"approved", "declined", "needs_more_information", "escalated"}
        if reviewer_decision is not None and reviewer_decision not in allowed:
            raise ValueError(f"reviewer_decision must be one of {sorted(allowed)}")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("UPDATE review_cases SET reviewer_decision = COALESCE(?, reviewer_decision), reviewer_note = COALESCE(?, reviewer_note), reviewed_at = ?, updated_at = ? WHERE case_id = ?", (reviewer_decision, reviewer_note, now if reviewer_decision else None, now, case_id))
        return self.get(case_id)
