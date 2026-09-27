from pathlib import Path


def test_compose_forwards_api_key_to_api_and_dashboard():
    compose = Path(__file__).parents[1].joinpath("docker-compose.yml").read_text(encoding="utf-8")
    assert compose.count('CREDIT_API_KEY: "${CREDIT_API_KEY:-}"') == 2
    assert 'X-API-Key' in Path(__file__).parents[1].joinpath("dashboard.py").read_text(encoding="utf-8")
