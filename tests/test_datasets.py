import pandas as pd
import pytest

from credit_simulator.datasets import ADAPTERS, DatasetBundle, DatasetAdapter, GermanCreditAdapter


def test_dataset_registry_contains_three_adapters():
    assert set(ADAPTERS) == {"uci_default", "german_credit", "give_me_some_credit"}


def test_german_parser_normalizes_target(tmp_path):
    path = tmp_path / "german.data"
    path.write_text("".join(["A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A151 2 A173 1 A192 A201 1\n", "A12 48 A32 A46 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2\n"]), encoding="utf-8")
    frame = GermanCreditAdapter.parse(path)
    assert frame["credit_risk"].tolist() == [0, 1]


def test_dataset_adapter_contract_exposes_schema_metadata_and_split_strategy():
    bundle = DatasetBundle(
        name="demo",
        frame=pd.DataFrame({"feature": [0, 1] * 50, "audit_group": [1, 2] * 50, "target": [0, 1] * 50}),
        target="target",
        protected_attributes=["audit_group"],
        feature_columns=["feature"],
        metadata={"target_definition": "Demo binary outcome", "source_url": "https://example.test"},
    )
    adapter = DatasetAdapter()
    adapter.validate(bundle)
    assert adapter.target_definition(bundle) == "Demo binary outcome"
    assert adapter.feature_schema(bundle) == {"feature": "int64"}
    assert adapter.protected_attributes(bundle) == ["audit_group"]
    assert adapter.source_metadata(bundle)["source_url"] == "https://example.test"
    assert adapter.split_strategy(bundle) == "stratified_random"


def test_dataset_adapter_rejects_protected_model_feature():
    bundle = DatasetBundle(
        name="demo",
        frame=pd.DataFrame({"feature": [0] * 100, "target": [0, 1] * 50}),
        target="target",
        protected_attributes=["feature"],
        feature_columns=["feature"],
        metadata={},
    )
    with pytest.raises(ValueError, match="protected attributes"):
        DatasetAdapter().validate(bundle)
