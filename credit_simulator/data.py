from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
from zipfile import ZipFile

import pandas as pd
import numpy as np
import requests


DATA_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
TARGET = "default"
PROTECTED = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{month}" for month in range(1, 7)]
PAYMENT_COLUMNS = [f"PAY_AMT{month}" for month in range(1, 7)]


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    rename = {"default.payment.next.month": TARGET, "default payment next month": TARGET}
    frame = frame.rename(columns=rename)
    if TARGET not in frame.columns:
        raise ValueError("Dataset must contain the default.payment.next.month target column")
    return frame


def load_uci_data(raw_dir: str | Path = "data/raw", download: bool = True) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    archive_path = raw_path / "default_of_credit_card_clients.zip"
    if download and not archive_path.exists():
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
        archive_path.write_bytes(response.content)
    if not archive_path.exists():
        raise FileNotFoundError(f"Dataset archive not found at {archive_path}. Run training with network access.")
    with ZipFile(archive_path) as archive:
        excel_name = next((name for name in archive.namelist() if name.lower().endswith((".xls", ".xlsx"))), None)
        if not excel_name:
            raise ValueError("UCI archive does not contain an Excel dataset")
        with archive.open(excel_name) as excel_file:
            frame = pd.read_excel(BytesIO(excel_file.read()), header=1)
    return _clean_columns(frame)


def validate_frame(frame: pd.DataFrame) -> None:
    required = {TARGET, *PROTECTED, "LIMIT_BAL"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame[TARGET].isna().any() or not set(frame[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target must be binary and non-null")
    domain_errors = []
    for column, allowed in {"SEX": {1, 2}, "EDUCATION": set(range(0, 7)), "MARRIAGE": set(range(0, 4))}.items():
        if column in frame and not frame[column].dropna().isin(allowed).all():
            domain_errors.append(column)
    for column in STATUS_COLUMNS:
        if column in frame and not frame[column].dropna().between(-2, 8).all():
            domain_errors.append(column)
    if "AGE" in frame and not frame["AGE"].dropna().between(18, 100).all():
        domain_errors.append("AGE")
    for column in ["LIMIT_BAL", *BILL_COLUMNS, *PAYMENT_COLUMNS]:
        if column in frame:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric[frame[column].notna()].isna().any():
                domain_errors.append(column)
            if numeric.notna().any() and not np.isfinite(numeric.dropna()).all():
                domain_errors.append(column)
            if column == "LIMIT_BAL" and not numeric.dropna().ge(0).all():
                domain_errors.append(column)
            if column in PAYMENT_COLUMNS and not numeric.dropna().ge(0).all():
                domain_errors.append(column)
    if domain_errors:
        raise ValueError(f"Feature values are outside the accepted UCI domains: {sorted(set(domain_errors))}")
    if len(frame) < 100:
        raise ValueError("Dataset is too small for a reliable train/test split")


def suspicious_leakage_columns(features: list[str]) -> list[str]:
    markers = ("target", "label", "outcome", "default", "future", "postapproval", "post_approval")
    suspicious = []
    for name in features:
        normalized = re.sub(r"[^a-z0-9]+", "", str(name).lower())
        if any(marker in normalized for marker in markers):
            suspicious.append(name)
    return suspicious


def validate_no_leakage(features: list[str]) -> None:
    suspicious = suspicious_leakage_columns(features)
    if suspicious:
        raise ValueError(f"Potential target leakage in model features: {suspicious}")


def validate_missingness(frame: pd.DataFrame, max_fraction: float) -> None:
    bad = frame.columns[frame.isna().mean() > max_fraction].tolist()
    if bad:
        raise ValueError(f"Columns exceed the allowed missingness fraction: {bad}")


def model_features(frame: pd.DataFrame) -> list[str]:
    excluded = {TARGET, "ID", *PROTECTED}
    return [column for column in frame.columns if column not in excluded]
