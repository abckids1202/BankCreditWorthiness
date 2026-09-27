from __future__ import annotations

import json
import hashlib
import hmac
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from .explain import reason_codes, structured_reasons
from .features import engineer_features
from .policy import decide, simulate_thresholds
from .scoring import probability_to_score, risk_band
from .review import ReviewStore
from .datasets import ADAPTERS
from .monitoring import drift_report
from .registry import list_models
from .predictions import PredictionEventStore
from .drift_events import DriftEventStore
from .rate_limit import RateLimiter


ARTIFACT_DIR = Path("artifacts")
review_store = ReviewStore()
prediction_store = PredictionEventStore()
drift_store = DriftEventStore()
rate_limiter = RateLimiter()
app = FastAPI(title="Explainable Credit Approval Simulator", version="0.1.0", description="Educational prototype only; not for real lending decisions.")
logger = logging.getLogger("credit_simulator.api")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


@app.middleware("http")
async def request_context(request, call_next):
    supplied_request_id = request.headers.get("X-Request-ID")
    request_id = supplied_request_id if supplied_request_id and _REQUEST_ID_PATTERN.fullmatch(supplied_request_id) else str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    configured_key = os.getenv("CREDIT_API_KEY")
    public_paths = {"/health", "/ready", "/docs", "/openapi.json", "/redoc"}
    supplied_key = request.headers.get("X-API-Key", "")
    if configured_key and request.url.path not in public_paths and not hmac.compare_digest(supplied_key, configured_key):
        response = JSONResponse(status_code=401, content={"detail": "Missing or invalid X-API-Key"})
    else:
        rate_limit_raw = os.getenv("CREDIT_RATE_LIMIT_PER_MINUTE")
        if rate_limit_raw and request.url.path not in public_paths:
            try:
                rate_limit = int(rate_limit_raw)
                if rate_limit <= 0:
                    raise ValueError
            except ValueError:
                response = JSONResponse(status_code=500, content={"detail": "CREDIT_RATE_LIMIT_PER_MINUTE must be a positive integer"})
            else:
                key = request.headers.get("X-API-Key") or (request.client.host if request.client else "unknown")
                allowed, retry_after, remaining = rate_limiter.check_with_remaining(key, rate_limit)
                if not allowed:
                    response = JSONResponse(status_code=429, content={"detail": "Rate limit exceeded; retry later"}, headers={"Retry-After": str(retry_after), "X-RateLimit-Limit": str(rate_limit), "X-RateLimit-Remaining": "0"})
                else:
                    response = await call_next(request)
                    response.headers["X-RateLimit-Limit"] = str(rate_limit)
                    response.headers["X-RateLimit-Remaining"] = str(remaining)
        else:
            response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info("request_id=%s method=%s path=%s status=%s duration_ms=%.2f", request_id, request.method, request.url.path, response.status_code, (time.perf_counter() - started) * 1000)
    return response


class Applicant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    LIMIT_BAL: FiniteFloat = Field(gt=0)
    PAY_0: FiniteFloat = Field(ge=-2, le=8); PAY_2: FiniteFloat = Field(ge=-2, le=8); PAY_3: FiniteFloat = Field(ge=-2, le=8)
    PAY_4: FiniteFloat = Field(ge=-2, le=8); PAY_5: FiniteFloat = Field(ge=-2, le=8); PAY_6: FiniteFloat = Field(ge=-2, le=8)
    BILL_AMT1: FiniteFloat; BILL_AMT2: FiniteFloat; BILL_AMT3: FiniteFloat; BILL_AMT4: FiniteFloat; BILL_AMT5: FiniteFloat; BILL_AMT6: FiniteFloat
    PAY_AMT1: FiniteFloat = Field(ge=0); PAY_AMT2: FiniteFloat = Field(ge=0); PAY_AMT3: FiniteFloat = Field(ge=0)
    PAY_AMT4: FiniteFloat = Field(ge=0); PAY_AMT5: FiniteFloat = Field(ge=0); PAY_AMT6: FiniteFloat = Field(ge=0)


class Prediction(BaseModel):
    risk_probability: FiniteFloat = Field(ge=0, le=1)
    default_probability: FiniteFloat = Field(ge=0, le=1)
    credit_score: int = Field(ge=300, le=850)
    risk_band: str
    decision: str
    rationale: str
    decision_thresholds: dict[str, float]
    model_version: str
    policy_version: str
    dataset_version: str = "unknown"
    reason_codes: list[str]
    explanations: list[dict]
    warnings: list[str]
    educational_disclaimer: str


class BatchPredictionRequest(BaseModel):
    applicants: list[Applicant] = Field(min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[Prediction]


class DatasetPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, Any]


class DatasetPrediction(BaseModel):
    dataset: str
    risk_probability: FiniteFloat = Field(ge=0, le=1)
    default_probability: FiniteFloat = Field(ge=0, le=1)
    credit_score: int = Field(ge=300, le=850)
    risk_band: str
    decision: str
    model_version: str
    policy_version: str
    dataset_version: str = "unknown"
    warnings: list[str]
    educational_disclaimer: str


ReviewerDecision = Literal["approved", "declined", "needs_more_information", "escalated"]


class ReviewCase(BaseModel):
    case_id: str
    created_at: str
    updated_at: str
    applicant: dict[str, Any]
    model_version: str
    policy_version: str
    decision_thresholds: dict[str, float]
    risk_probability: FiniteFloat = Field(ge=0, le=1)
    credit_score: int = Field(ge=300, le=850)
    automatic_decision: Literal["approve", "manual_review", "decline"]
    reason_codes: list[str]
    warnings: list[str]
    reviewer_decision: ReviewerDecision | None = None
    reviewer_note: str | None = None
    reviewer_id: str | None = None
    reviewed_at: str | None = None


class ReviewHistoryEvent(BaseModel):
    event_id: str
    case_id: str
    event_type: Literal["created", "updated"]
    created_at: str
    reviewer_decision: ReviewerDecision | None = None
    reviewer_note: str | None = None
    reviewer_id: str | None = None


class ReviewHistoryResponse(BaseModel):
    case_id: str
    events: list[ReviewHistoryEvent]


class ThresholdSimulationRequest(BaseModel):
    probabilities: list[float] = Field(min_length=1, max_length=100000)
    approve_max_risk: float = Field(ge=0, le=1)
    decline_min_risk: float = Field(ge=0, le=1)
    actual_defaults: list[int] | None = None


class CurrentPolicy(BaseModel):
    policy_version: str
    decision_thresholds: dict[str, FiniteFloat]
    risk_bands: dict[str, FiniteFloat]


class DriftRequest(BaseModel):
    reference_records: list[dict[str, Any]] = Field(min_length=1, max_length=100000)
    current_records: list[dict[str, Any]] = Field(min_length=1, max_length=100000)
    features: list[str] | None = None
    psi_warning: float = Field(default=0.10, ge=0, le=1)
    psi_critical: float = Field(default=0.25, ge=0, le=1)
    missing_warning: float = Field(default=0.05, ge=0, le=1)
    missing_critical: float = Field(default=0.15, ge=0, le=1)


class DriftHistoryEvent(BaseModel):
    event_id: int
    created_at: str
    request_id: str | None = None
    reference_rows: int
    current_rows: int
    features_checked: list[str]
    warning_features: list[str]
    critical_features: list[str]
    thresholds: dict[str, float]
    recommended_action: str
    model_version: str
    policy_version: str


def _artifacts():
    model_path, metadata_path = ARTIFACT_DIR / "model.joblib", ARTIFACT_DIR / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise HTTPException(503, "Model artifacts are unavailable. Run: python scripts/train.py")
    try:
        model = joblib.load(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        required = {"model_version", "policy_version", "artifact_fingerprint", "training_config_sha256", "feature_names", "raw_feature_names", "thresholds", "score", "risk_bands", "feature_stats", "disclaimer", "model_sha256"}
        checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if not isinstance(metadata, dict) or not required.issubset(metadata) or metadata["model_sha256"] != checksum or not hasattr(model, "predict_proba"):
            raise ValueError("missing required model metadata or prediction interface")
        return model, metadata
    except Exception as exc:
        raise HTTPException(503, "Model artifacts could not be loaded; retrain or restore a valid artifact") from exc


def _dataset_artifacts(dataset: str):
    if dataset not in ADAPTERS or dataset == "uci_default":
        raise HTTPException(404, "Unknown alternate dataset")
    model_path, metadata_path = ARTIFACT_DIR / dataset / "model.joblib", ARTIFACT_DIR / dataset / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise HTTPException(503, f"Artifacts for {dataset} are unavailable. Run: python scripts/train.py --dataset {dataset}")
    try:
        model = joblib.load(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
        required = {"model_sha256", "model_version", "feature_names", "disclaimer", "dataset", "policy_version"}
        if not isinstance(metadata, dict) or not required.issubset(metadata) or metadata["model_sha256"] != checksum or not hasattr(model, "predict_proba"):
            raise ValueError("missing required alternate model metadata or prediction interface")
        return model, metadata
    except Exception as exc:
        raise HTTPException(503, f"Artifacts for {dataset} failed integrity validation; retrain the dataset model") from exc


def _dataset_version(metadata: dict) -> str:
    dataset = metadata.get("dataset")
    if isinstance(dataset, dict):
        return str(dataset.get("version") or dataset.get("sha256") or "unknown")
    return str(metadata.get("dataset_sha256") or "unknown")


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": (ARTIFACT_DIR / "model.joblib").exists()}


@app.get("/ready")
def ready():
    _, metadata = _artifacts()
    return {"status": "ready", "model_version": metadata["model_version"], "policy_version": metadata.get("policy_version", "unknown"), "artifact_fingerprint": metadata.get("artifact_fingerprint", "unknown"), "training_config_sha256": metadata.get("training_config_sha256", "unknown")}


@app.post("/policy/simulate")
def policy_simulation(request: ThresholdSimulationRequest):
    try:
        return simulate_thresholds(request.probabilities, request.approve_max_risk, request.decline_min_risk, request.actual_defaults)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/policy/current", response_model=CurrentPolicy)
def current_policy():
    _, metadata = _artifacts()
    return CurrentPolicy(policy_version=metadata.get("policy_version", "unknown"), decision_thresholds={key: metadata["thresholds"][key] for key in ("approve_max_risk", "decline_min_risk")}, risk_bands=metadata["risk_bands"])


@app.post("/monitoring/drift")
def monitoring_drift(request: DriftRequest, http_request: Request):
    try:
        report = drift_report(request.reference_records, request.current_records, request.features, request.psi_warning, request.psi_critical, request.missing_warning, request.missing_critical)
        model_version = policy_version = "unknown"
        metadata_path = ARTIFACT_DIR / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(metadata, dict):
                model_version = metadata.get("model_version", "unknown")
                policy_version = metadata.get("policy_version", "unknown")
        except (OSError, json.JSONDecodeError):
            pass
        report["model_version"] = model_version
        report["policy_version"] = policy_version
        drift_store.record(report, getattr(http_request.state, "request_id", None), model_version, policy_version)
        return report
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/monitoring/drift/history", response_model=list[DriftHistoryEvent])
def monitoring_drift_history(limit: int = 50):
    return drift_store.list(limit)


@app.get("/model-info")
def model_info():
    _, metadata = _artifacts()
    return metadata


@app.get("/models")
def models():
    return {"models": list_models()}


@app.get("/prediction-stats")
def prediction_stats():
    return prediction_store.summary()


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant, request: Request):
    model, metadata = _artifacts()
    payload = applicant.model_dump()
    raw_features = metadata.get("raw_feature_names", metadata["feature_names"])
    raw_frame = pd.DataFrame([{name: payload[name] for name in raw_features}])
    frame = engineer_features(raw_frame)
    features = metadata["feature_names"]
    try:
        probability = float(np.clip(model.predict_proba(frame[features])[:, 1][0], 1e-6, 1 - 1e-6))
        z_values = []
        for feature, stats in metadata.get("feature_stats", {}).items():
            if feature in frame and np.isfinite(frame[feature].iloc[0]):
                z_values.append(abs(float(frame[feature].iloc[0]) - stats["mean"]) / max(stats["std"], 1e-9))
        outlier = bool(z_values and max(z_values) > metadata["thresholds"].get("out_of_distribution_z", 5.0)) or bool((frame.abs() > 1e9).any(axis=None))
        decision = decide(probability, out_of_distribution=outlier, **{key: metadata["thresholds"][key] for key in ("approve_max_risk", "decline_min_risk")})
    except (KeyError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    warnings = ["Input is outside the training distribution and was routed to review"] if outlier else []
    explanations = structured_reasons(model, frame, features, metadata.get("feature_descriptions"))
    score = probability_to_score(probability, metadata["score"]); band = risk_band(probability, metadata["risk_bands"])
    prediction_store.record("uci_default", metadata["model_version"], probability, score, band, decision.decision, getattr(request.state, "request_id", None), metadata.get("policy_version", "unknown"))
    return Prediction(risk_probability=probability, default_probability=probability, credit_score=score, risk_band=band, decision=decision.decision, rationale=decision.rationale, decision_thresholds={key: metadata["thresholds"][key] for key in ("approve_max_risk", "decline_min_risk")}, model_version=metadata["model_version"], dataset_version=_dataset_version(metadata), policy_version=metadata.get("policy_version", "unknown"), reason_codes=reason_codes(model, frame, features), explanations=explanations, warnings=warnings, educational_disclaimer=metadata["disclaimer"])


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchPredictionRequest, request: Request):
    predictions = [predict(applicant, request) for applicant in batch.applicants]
    return BatchPredictionResponse(count=len(predictions), predictions=predictions)


@app.post("/predict/{dataset}", response_model=DatasetPrediction)
def predict_alternate(dataset: str, request: DatasetPredictionRequest, http_request: Request):
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
    policy_version = metadata.get("policy_version", "alternate-experiment-0.1.0")
    prediction_store.record(dataset, metadata["model_version"], probability, probability_to_score(probability), risk_band(probability), decision.decision, getattr(http_request.state, "request_id", None), policy_version)
    return DatasetPrediction(dataset=dataset, risk_probability=probability, default_probability=probability, credit_score=probability_to_score(probability), risk_band=risk_band(probability), decision=decision.decision, model_version=metadata["model_version"], dataset_version=_dataset_version(metadata), policy_version=policy_version, warnings=["Alternate dataset model; explanations and thresholds are dataset-specific research outputs"], educational_disclaimer=metadata["disclaimer"])


@app.post("/review-cases", response_model=ReviewCase)
def create_review_case(applicant: Applicant, request: Request):
    prediction_result = predict(applicant, request)
    if prediction_result.decision != "manual_review":
        raise HTTPException(422, "Only manual_review recommendations can be sent to the human-review queue")
    prediction = prediction_result.model_dump()
    return review_store.create(applicant.model_dump(), prediction)


@app.get("/review-cases", response_model=list[ReviewCase])
def list_review_cases(limit: int = 50):
    return review_store.list(limit)


@app.get("/review-cases/{case_id}", response_model=ReviewCase)
def get_review_case(case_id: str):
    case = review_store.get(case_id)
    if not case:
        raise HTTPException(404, "Review case not found")
    return case


@app.get("/review-cases/{case_id}/history", response_model=ReviewHistoryResponse)
def get_review_case_history(case_id: str):
    history = review_store.history(case_id)
    if history is None:
        raise HTTPException(404, "Review case not found")
    return {"case_id": case_id, "events": history}


class ReviewUpdate(BaseModel):
    reviewer_decision: ReviewerDecision | None = None
    reviewer_note: str | None = Field(default=None, max_length=5000)
    reviewer_id: str | None = Field(default=None, min_length=1, max_length=100)


@app.patch("/review-cases/{case_id}", response_model=ReviewCase)
def update_review_case(case_id: str, update: ReviewUpdate):
    try:
        case = review_store.update(case_id, update.reviewer_decision, update.reviewer_note, update.reviewer_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not case:
        raise HTTPException(404, "Review case not found")
    return case

