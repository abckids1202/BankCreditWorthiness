from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests


DATA_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
TARGET = "default"
PROTECTED = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]


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
    if len(frame) < 100:
        raise ValueError("Dataset is too small for a reliable train/test split")


def validate_no_leakage(features: list[str]) -> None:
    suspicious = [name for name in features if "default.payment" in name.lower() or name.lower() in {"target", "label"}]
    if suspicious:
        raise ValueError(f"Potential target leakage in model features: {suspicious}")


def validate_missingness(frame: pd.DataFrame, max_fraction: float) -> None:
    bad = frame.columns[frame.isna().mean() > max_fraction].tolist()
    if bad:
        raise ValueError(f"Columns exceed the allowed missingness fraction: {bad}")


def model_features(frame: pd.DataFrame) -> list[str]:
    excluded = {TARGET, "ID", *PROTECTED}
    return [column for column in frame.columns if column not in excluded]
