import json
from pathlib import Path

from credit_simulator.api import Applicant


def test_synthetic_example_matches_api_schema():
    payload = json.loads((Path(__file__).parents[1] / "examples" / "sample_applicant.json").read_text(encoding="utf-8"))
    applicant = Applicant.model_validate(payload)
    assert applicant.LIMIT_BAL == 50000
