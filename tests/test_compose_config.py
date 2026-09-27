from pathlib import Path

import yaml


def test_compose_forwards_api_key_to_api_and_dashboard():
    compose = Path(__file__).parents[1].joinpath("docker-compose.yml").read_text(encoding="utf-8")
    assert compose.count('CREDIT_API_KEY: "${CREDIT_API_KEY:-}"') == 2
    assert 'X-API-Key' in Path(__file__).parents[1].joinpath("dashboard.py").read_text(encoding="utf-8")


def test_compose_has_healthy_api_dependency():
    root = Path(__file__).parents[1]
    config = yaml.safe_load(root.joinpath("docker-compose.yml").read_text(encoding="utf-8"))
    assert set(config["services"]) == {"api", "dashboard"}
    assert config["services"]["dashboard"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert "/ready" in " ".join(config["services"]["api"]["healthcheck"]["test"])
