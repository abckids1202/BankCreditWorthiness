from credit_simulator.datasets import ADAPTERS, GermanCreditAdapter


def test_dataset_registry_contains_three_adapters():
    assert set(ADAPTERS) == {"uci_default", "german_credit", "give_me_some_credit"}


def test_german_parser_normalizes_target(tmp_path):
    path = tmp_path / "german.data"
    path.write_text("".join(["A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A151 2 A173 1 A192 A201 1\n", "A12 48 A32 A46 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2\n"]), encoding="utf-8")
    frame = GermanCreditAdapter.parse(path)
    assert frame["credit_risk"].tolist() == [0, 1]
