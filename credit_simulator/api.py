from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .explain import reason_codes
from .features import engineer_features
from .policy import decide
from .scoring import probability_to_score, risk_band


ARTIFACT_DIR = Path("artifacts")
app = FastAPI(title="Explainable Credit Approval Simulator", version="0.1.0", description="Educational prototype only; not for real lending decisions.")


class Applicant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    LIMIT_BAL: float = Field(gt=0)
    PAY_0: float; PAY_2: float; PAY_3: float; PAY_4: float; PAY_5: float; PAY_6: float
    BILL_AMT1: float; BILL_AMT2: float; BILL_AMT3: float; BILL_AMT4: float; BILL_AMT5: float; BILL_AMT6: float
    PAY_AMT1: float; PAY_AMT2: float; PAY_AMT3: float; PAY_AMT4: float; PAY_AMT5: float; PAY_AMT6: float


class Prediction(BaseModel):
    risk_probability: float
    credit_score: int
    risk_band: str
    decision: str
    rationale: str
    decision_thresholds: dict[str, float]
    model_version: str
    reason_codes: list[str]
    warnings: list[str]
    educational_disclaimer: str


def _artifacts():
    model_path, metadata_path = ARTIFACT_DIR / "model.joblib", ARTIFACT_DIR / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise HTTPException(503, "Model artifacts are unavailable. Run: python scripts/train.py")
    return joblib.load(model_path), json.loads(metadata_path.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": (ARTIFACT_DIR / "model.joblib").exists()}


@app.get("/model-info")
def model_info():
    _, metadata = _artifacts()
    return metadata


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant):
    model, metadata = _artifacts()
    payload = applicant.model_dump()
    raw_features = metadata.get("raw_feature_names", metadata["feature_names"])
    raw_frame = pd.DataFrame([{name: payload[name] for name in raw_features}])
    frame = engineer_features(raw_frame)
    features = metadata["feature_names"]
    try:
        probability = float(np.clip(model.predict_proba(frame[features])[:, 1][0], 1e-6, 1 - 1e-6))
        outlier = bool((frame.abs() > 1e9).any(axis=None))
        decision = decide(probability, out_of_distribution=outlier, **{key: metadata["thresholds"][key] for key in ("approve_max_risk", "decline_min_risk")})
    except (KeyError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    warnings = ["Input contains an extreme value and was routed to review"] if outlier else []
    return Prediction(risk_probability=probability, credit_score=probability_to_score(probability, metadata["score"]), risk_band=risk_band(probability, metadata["risk_bands"]), decision=decision.decision, rationale=decision.rationale, decision_thresholds={key: metadata["thresholds"][key] for key in ("approve_max_risk", "decline_min_risk")}, model_version=metadata["model_version"], reason_codes=reason_codes(model, frame, features), warnings=warnings, educational_disclaimer=metadata["disclaimer"])

