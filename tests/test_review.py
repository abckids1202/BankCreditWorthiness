from credit_simulator.review import ReviewStore
from credit_simulator.api import ReviewUpdate
import pytest


def test_review_case_preserves_model_decision(tmp_path):
    store = ReviewStore(tmp_path / "reviews.db")
    case = store.create({"LIMIT_BAL": 1000}, {"model_version": "test", "policy_version": "policy-0.1.0", "risk_probability": 0.3, "credit_score": 600, "decision": "manual_review", "reason_codes": ["test"], "warnings": []})
    assert case["automatic_decision"] == "manual_review"
    assert case["policy_version"] == "policy-0.1.0"
    assert store.history(case["case_id"])[0]["event_type"] == "created"
    updated = store.update(case["case_id"], "approved", "Reviewed for prototype", "reviewer-demo")
    assert updated["automatic_decision"] == "manual_review"
    assert updated["reviewer_decision"] == "approved"
    assert updated["reviewer_id"] == "reviewer-demo"
    first_reviewed_at = updated["reviewed_at"]
    assert first_reviewed_at
    note_only = store.update(case["case_id"], None, "Additional context", "reviewer-demo")
    assert note_only["reviewed_at"] == first_reviewed_at
    assert note_only["reviewer_decision"] == "approved"
    events = store.history(case["case_id"])
    assert len(events) == 3
    assert events[1]["reviewer_decision"] == "approved"
    assert events[1]["reviewer_note"] == "Reviewed for prototype"
    assert events[1]["reviewer_id"] == "reviewer-demo"
    assert events[2]["reviewer_note"] == "Additional context"


def test_review_history_missing_case_returns_none(tmp_path):
    store = ReviewStore(tmp_path / "reviews.db")
    assert store.history("missing") is None
    assert store.update("missing", "approved", "note") is None


def test_review_store_migrates_existing_schema(tmp_path):
    import sqlite3

    database = tmp_path / "legacy_reviews.db"
    with sqlite3.connect(database) as connection:
        connection.execute("""CREATE TABLE review_cases (
            case_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            applicant_json TEXT NOT NULL, model_version TEXT NOT NULL, risk_probability REAL NOT NULL,
            credit_score INTEGER NOT NULL, automatic_decision TEXT NOT NULL, reason_codes_json TEXT NOT NULL,
            warnings_json TEXT NOT NULL, reviewer_decision TEXT, reviewer_note TEXT, reviewed_at TEXT
        )""")
        connection.execute(
            "INSERT INTO review_cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("legacy-case", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00", "{}", "legacy", 0.2, 650, "approve", "[]", "[]", None, None, None),
        )

    store = ReviewStore(database)
    assert store.history("legacy-case")[0]["event_type"] == "created"
    case = store.create({"LIMIT_BAL": 1000}, {"model_version": "legacy-compatible", "risk_probability": 0.2, "credit_score": 650, "decision": "approve", "reason_codes": [], "warnings": []})
    assert case["policy_version"] == "unknown"


def test_review_update_schema_rejects_unknown_decision():
    with pytest.raises(ValueError):
        ReviewUpdate.model_validate({"reviewer_decision": "maybe"})


def test_review_update_schema_limits_notes():
    with pytest.raises(ValueError):
        ReviewUpdate.model_validate({"reviewer_note": "x" * 5001})


def test_review_update_schema_validates_reviewer_id():
    with pytest.raises(ValueError):
        ReviewUpdate.model_validate({"reviewer_id": ""})

