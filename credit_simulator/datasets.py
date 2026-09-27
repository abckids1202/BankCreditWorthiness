from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests

from .data import load_uci_data


@dataclass
class DatasetBundle:
    name: str
    frame: pd.DataFrame
    target: str
    protected_attributes: list[str]
    feature_columns: list[str]
    metadata: dict
    split_strategy: str = "stratified_random"


class DatasetAdapter:
    name = ""

    def load(self, raw_dir: str | Path = "data/raw") -> DatasetBundle:
        raise NotImplementedError

    def validate(self, bundle: DatasetBundle) -> None:
        if bundle.target not in bundle.frame:
            raise ValueError(f"{self.name}: target column {bundle.target!r} is missing")
        if bundle.frame[bundle.target].isna().any():
            raise ValueError(f"{self.name}: target contains missing values")
        if len(bundle.frame) < 100:
            raise ValueError(f"{self.name}: dataset has fewer than 100 rows")
        if not set(bundle.frame[bundle.target].unique()).issubset({0, 1}):
            raise ValueError(f"{self.name}: target must be encoded as 0/1")
        missing = set(bundle.feature_columns) - set(bundle.frame.columns)
        if missing:
            raise ValueError(f"{self.name}: missing feature columns {sorted(missing)}")


class UCIDefaultAdapter(DatasetAdapter):
    name = "uci_default"

    def load(self, raw_dir: str | Path = "data/raw") -> DatasetBundle:
        frame = load_uci_data(raw_dir)
        bundle = DatasetBundle(name=self.name, frame=frame, target="default", protected_attributes=["SEX", "EDUCATION", "MARRIAGE", "AGE"], feature_columns=[column for column in frame.columns if column not in {"default", "ID", "SEX", "EDUCATION", "MARRIAGE", "AGE"}], metadata={"source_url": "https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients", "license": "CC BY 4.0", "target_definition": "Default payment in the following month", "row_definition": "Existing credit-card client"})
        self.validate(bundle); return bundle


class GermanCreditAdapter(DatasetAdapter):
    name = "german_credit"
    URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
    COLUMNS = ["checking_status", "duration_months", "credit_history", "purpose", "credit_amount", "savings_status", "employment_since", "installment_rate", "personal_status_sex", "other_debtors", "residence_since", "property_status", "age", "other_installment_plans", "housing", "existing_credits", "job", "dependents", "telephone", "foreign_worker", "credit_risk"]

    @classmethod
    def parse(cls, path: str | Path) -> pd.DataFrame:
        frame = pd.read_csv(path, sep=r"\s+", header=None, names=cls.COLUMNS)
        frame["credit_risk"] = (frame["credit_risk"] == 2).astype(int)
        return frame

    def load(self, raw_dir: str | Path = "data/raw") -> DatasetBundle:
        raw_path = Path(raw_dir); raw_path.mkdir(parents=True, exist_ok=True); data_path = raw_path / "german.data"
        if not data_path.exists():
            archive_path = raw_path / "german_credit.zip"; response = requests.get(self.URL, timeout=60); response.raise_for_status(); archive_path.write_bytes(response.content)
            with ZipFile(archive_path) as archive:
                member = next(name for name in archive.namelist() if name.endswith("german.data")); data_path.write_bytes(archive.read(member))
        frame = self.parse(data_path)
        features = [column for column in frame.columns if column != "credit_risk"]
        bundle = DatasetBundle(name=self.name, frame=frame, target="credit_risk", protected_attributes=["personal_status_sex", "age"], feature_columns=features, metadata={"source_url": "https://archive.ics.uci.edu/dataset/144/statlog%2Bgerman%2Bcredit%2Bdata", "license": "CC BY 4.0", "target_definition": "Bad credit risk according to the dataset label", "row_definition": "Credit application", "cost_matrix": "Misclassifying bad credit as good has higher cost"})
        self.validate(bundle); return bundle


class GiveMeSomeCreditAdapter(DatasetAdapter):
    name = "give_me_some_credit"
    URL = "https://www.kaggle.com/c/GiveMeSomeCredit/data"

    def load(self, raw_dir: str | Path = "data/raw") -> DatasetBundle:
        path = Path(raw_dir) / "cs-training.csv"
        if not path.exists():
            raise FileNotFoundError(f"Download cs-training.csv from {self.URL} and place it at {path}")
        frame = pd.read_csv(path)
        if "SeriousDlqin2yrs" not in frame:
            raise ValueError("Give Me Some Credit file must contain SeriousDlqin2yrs")
        frame = frame.rename(columns={"SeriousDlqin2yrs": "default"})
        features = [column for column in frame.columns if column not in {"default", "Unnamed: 0"}]
        bundle = DatasetBundle(name=self.name, frame=frame, target="default", protected_attributes=["age"], feature_columns=features, metadata={"source_url": self.URL, "license": "Kaggle competition terms", "target_definition": "Serious delinquency 90 days or worse within two years", "row_definition": "Applicant record"})
        self.validate(bundle); return bundle


ADAPTERS = {adapter.name: adapter for adapter in (UCIDefaultAdapter(), GermanCreditAdapter(), GiveMeSomeCreditAdapter())}


def get_dataset(name: str, raw_dir: str | Path = "data/raw") -> DatasetBundle:
    try:
        return ADAPTERS[name].load(raw_dir)
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(ADAPTERS)}") from exc
