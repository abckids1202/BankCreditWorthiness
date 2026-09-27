import sqlite3

from credit_simulator.predictions import PredictionEventStore


def test_prediction_store_records_policy_version_and_summary(tmp_path):
    store = PredictionEventStore(tmp_path / "predictions.db")
    store.record("uci_default", "model-1", 0.2, 700, "low", "approve", "request-1", "policy-2")
    summary = store.summary()
    assert summary["groups"] == [{"dataset": "uci_default", "model_version": "model-1", "policy_version": "policy-2", "decision": "approve", "count": 1}]


def test_prediction_store_migrates_legacy_database(tmp_path):
    database = tmp_path / "legacy.db"
    with sqlite3.connect(database) as connection:
        connection.execute("""CREATE TABLE prediction_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            request_id TEXT, dataset TEXT NOT NULL, model_version TEXT NOT NULL,
            risk_probability REAL NOT NULL, credit_score INTEGER NOT NULL,
            risk_band TEXT NOT NULL, decision TEXT NOT NULL
        )""")
    store = PredictionEventStore(database)
    store.record("legacy", "model-legacy", 0.5, 600, "moderate", "manual_review")
    assert store.summary()["groups"][0]["policy_version"] == "unknown"
