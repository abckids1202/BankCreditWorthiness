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
        protected_overlap = set(bundle.protected_attributes).intersection(bundle.feature_columns)
        if protected_overlap:
            raise ValueError(f"{self.name}: protected attributes cannot be model features: {sorted(protected_overlap)}")
        if len(bundle.feature_columns) != len(set(bundle.feature_columns)):
            raise ValueError(f"{self.name}: feature schema contains duplicate columns")

    def target_definition(self, bundle: DatasetBundle) -> str:
        return str(bundle.metadata.get("target_definition", bundle.target))

    def feature_schema(self, bundle: DatasetBundle) -> dict[str, str]:
        return {str(column): str(bundle.frame[column].dtype) for column in bundle.feature_columns}

    def protected_attributes(self, bundle: DatasetBundle) -> list[str]:
        return list(bundle.protected_attributes)

    def source_metadata(self, bundle: DatasetBundle) -> dict:
        return dict(bundle.metadata)

    def split_strategy(self, bundle: DatasetBundle) -> str:
        return bundle.split_strategy


class UCIDefaultAdapter(DatasetAdapter):
    name = "uci_default"
    SOURCE_METADATA = {
        "source_url": "https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients",
        "license": "CC BY 4.0",
        "citation": "Yeh and Lien (2009), The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients",
        "target_definition": "Default payment in the following month",
        "prediction_horizon": "One month after the six-month observation history",
        "row_definition": "Existing credit-card client",
        "missing_value_behavior": "Missing values are retained for quality checks and median-imputed inside the fitted pipeline",
        "known_limitations": "Historical Taiwanese credit-card customers; not a new-loan application population and not a causal fairness benchmark",
        "protected_attribute_notes": "SEX, EDUCATION, MARRIAGE, and AGE are retained for audit only",
    }

    def load(self, raw_dir: str | Path = "data/raw") -> DatasetBundle:
        frame = load_uci_data(raw_dir)
        bundle = DatasetBundle(name=self.name, frame=frame, target="default", protected_attributes=["SEX", "EDUCATION", "MARRIAGE", "AGE"], feature_columns=[column for column in frame.columns if column not in {"default", "ID", "SEX", "EDUCATION", "MARRIAGE", "AGE"}], metadata=dict(self.SOURCE_METADATA))
        self.validate(bundle); return bundle


class GermanCreditAdapter(DatasetAdapter):
    name = "german_credit"
    URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
    SOURCE_METADATA = {
        "source_url": "https://archive.ics.uci.edu/dataset/144/statlog%2Bgerman%2Bcredit%2Bdata",
        "license": "CC BY 4.0",
        "citation": "UCI Statlog (German Credit) dataset",
        "target_definition": "Bad credit risk according to the dataset label",
        "prediction_horizon": "Not specified by the source dataset",
        "row_definition": "Credit application",
        "missing_value_behavior": "The source file is encoded without blank fields; the pipeline still validates and imputes missing values if introduced",
        "known_limitations": "Small historical application sample, encoded categories, and an ambiguous target-cost context; not a modern lending population",
        "protected_attribute_notes": "personal_status_sex and age are retained for audit only",
        "cost_matrix": "Misclassifying bad credit as good has higher cost",
    }
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
        protected = ["personal_status_sex", "age"]
        features = [column for column in frame.columns if column not in {"credit_risk", *protected}]
        bundle = DatasetBundle(name=self.name, frame=frame, target="credit_risk", protected_attributes=protected, feature_columns=features, metadata=dict(self.SOURCE_METADATA))
        self.validate(bundle); return bundle


class GiveMeSomeCreditAdapter(DatasetAdapter):
    name = "give_me_some_credit"
    URL = "https://www.kaggle.com/c/GiveMeSomeCredit/data"
    SOURCE_METADATA = {
        "source_url": URL,
        "license": "Kaggle competition terms; verify current terms before reuse",
        "citation": "Give Me Some Credit Kaggle competition dataset",
        "target_definition": "Serious delinquency 90 days or worse within two years",
        "prediction_horizon": "Two years after the applicant record",
        "row_definition": "Applicant record",
        "missing_value_behavior": "MonthlyIncome and NumberOfDependents may be missing; missingness is reported and imputed inside the fitted pipeline",
        "known_limitations": "Competition data provenance and selection process are limited; age and other variables may encode protected or proxy information",
        "protected_attribute_notes": "age is retained for audit only",
    }

    def load(self, raw_dir: str | Path = "data/raw") -> DatasetBundle:
        path = Path(raw_dir) / "cs-training.csv"
        if not path.exists():
            raise FileNotFoundError(f"Download cs-training.csv from {self.URL} and place it at {path}")
        frame = pd.read_csv(path)
        if "SeriousDlqin2yrs" not in frame:
            raise ValueError("Give Me Some Credit file must contain SeriousDlqin2yrs")
        frame = frame.rename(columns={"SeriousDlqin2yrs": "default"})
        protected = ["age"]
        features = [column for column in frame.columns if column not in {"default", "Unnamed: 0", *protected}]
        bundle = DatasetBundle(name=self.name, frame=frame, target="default", protected_attributes=protected, feature_columns=features, metadata=dict(self.SOURCE_METADATA))
        self.validate(bundle); return bundle


ADAPTERS = {adapter.name: adapter for adapter in (UCIDefaultAdapter(), GermanCreditAdapter(), GiveMeSomeCreditAdapter())}


def get_dataset(name: str, raw_dir: str | Path = "data/raw") -> DatasetBundle:
    try:
        return ADAPTERS[name].load(raw_dir)
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(ADAPTERS)}") from exc
