from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .explain import reason_codes, structured_reasons
from .features import engineer_features
from .policy import decide, simulate_thresholds
from .scoring import probability_to_score, risk_band
from .review import ReviewStore
from .datasets import ADAPTERS
from .monitoring import drift_report


ARTIFACT_DIR = Path("artifacts")
review_store = ReviewStore()
app = FastAPI(title="Explainable Credit Approval Simulator", version="0.1.0", description="Educational prototype only; not for real lending decisions.")
logger = logging.getLogger("credit_simulator.api")


@app.middleware("http")
async def request_context(request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info("request_id=%s method=%s path=%s status=%s duration_ms=%.2f", request_id, request.method, request.url.path, response.status_code, (time.perf_counter() - started) * 1000)
    return response


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
    explanations: list[dict]
    warnings: list[str]
    educational_disclaimer: str


class DatasetPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, Any]


class DatasetPrediction(BaseModel):
    dataset: str
    risk_probability: float
    credit_score: int
    risk_band: str
    decision: str
    model_version: str
    warnings: list[str]
    educational_disclaimer: str


class ThresholdSimulationRequest(BaseModel):
    probabilities: list[float] = Field(min_length=1, max_length=100000)
    approve_max_risk: float = Field(ge=0, le=1)
    decline_min_risk: float = Field(ge=0, le=1)
    actual_defaults: list[int] | None = None


class DriftRequest(BaseModel):
    reference_records: list[dict[str, Any]] = Field(min_length=1, max_length=100000)
    current_records: list[dict[str, Any]] = Field(min_length=1, max_length=100000)
    features: list[str] | None = None


def _artifacts():
    model_path, metadata_path = ARTIFACT_DIR / "model.joblib", ARTIFACT_DIR / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise HTTPException(503, "Model artifacts are unavailable. Run: python scripts/train.py")
    return joblib.load(model_path), json.loads(metadata_path.read_text(encoding="utf-8"))


def _dataset_artifacts(dataset: str):
    if dataset not in ADAPTERS or dataset == "uci_default":
        raise HTTPException(404, "Unknown alternate dataset")
    model_path, metadata_path = ARTIFACT_DIR / dataset / "model.joblib", ARTIFACT_DIR / dataset / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise HTTPException(503, f"Artifacts for {dataset} are unavailable. Run: python scripts/train.py --dataset {dataset}")
    return joblib.load(model_path), json.loads(metadata_path.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": (ARTIFACT_DIR / "model.joblib").exists()}


@app.get("/ready")
def ready():
    _, metadata = _artifacts()
    return {"status": "ready", "model_version": metadata["model_version"]}


@app.post("/policy/simulate")
def policy_simulation(request: ThresholdSimulationRequest):
    try:
        return simulate_thresholds(request.probabilities, request.approve_max_risk, request.decline_min_risk, request.actual_defaults)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/monitoring/drift")
def monitoring_drift(request: DriftRequest):
    try:
        return drift_report(request.reference_records, request.current_records, request.features)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


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
    explanations = structured_reasons(model, frame, features, metadata.get("feature_descriptions"))
    return Prediction(risk_probability=probability, credit_score=probability_to_score(probability, metadata["score"]), risk_band=risk_band(probability, metadata["risk_bands"]), decision=decision.decision, rationale=decision.rationale, decision_thresholds={key: metadata["thresholds"][key] for key in ("approve_max_risk", "decline_min_risk")}, model_version=metadata["model_version"], reason_codes=reason_codes(model, frame, features), explanations=explanations, warnings=warnings, educational_disclaimer=metadata["disclaimer"])


@app.post("/predict/{dataset}", response_model=DatasetPrediction)
def predict_alternate(dataset: str, request: DatasetPredictionRequest):
    model, metadata = _dataset_artifacts(dataset)
    expected = set(metadata["feature_names"]); received = set(request.features)
    missing, unknown = sorted(expected - received), sorted(received - expected)
    if missing or unknown:
        raise HTTPException(422, {"missing_features": missing, "unknown_features": unknown})
    frame = pd.DataFrame([{name: request.features[name] for name in metadata["feature_names"]}])
    try:
        probability = float(np.clip(model.predict_proba(frame)[:, 1][0], 1e-6, 1 - 1e-6))
        decision = decide(probability)
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return DatasetPrediction(dataset=dataset, risk_probability=probability, credit_score=probability_to_score(probability), risk_band=risk_band(probability), decision=decision.decision, model_version=metadata["model_version"], warnings=["Alternate dataset model; explanations and thresholds are dataset-specific research outputs"], educational_disclaimer=metadata["disclaimer"])


@app.post("/review-cases")
def create_review_case(applicant: Applicant):
    prediction = predict(applicant).model_dump()
    return review_store.create(applicant.model_dump(), prediction)


@app.get("/review-cases")
def list_review_cases(limit: int = 50):
    return review_store.list(limit)


@app.get("/review-cases/{case_id}")
def get_review_case(case_id: str):
    case = review_store.get(case_id)
    if not case:
        raise HTTPException(404, "Review case not found")
    return case


class ReviewUpdate(BaseModel):
    reviewer_decision: str | None = None
    reviewer_note: str | None = None


@app.patch("/review-cases/{case_id}")
def update_review_case(case_id: str, update: ReviewUpdate):
    try:
        case = review_store.update(case_id, update.reviewer_decision, update.reviewer_note)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not case:
        raise HTTPException(404, "Review case not found")
    return case

