import argparse

from credit_simulator.registry import promote_model


parser = argparse.ArgumentParser(description="Explicitly promote or roll back a verified model snapshot")
parser.add_argument("--version", required=True, help="Exact model_version from GET /models")
parser.add_argument("--dataset", default=None, help="Optional exact dataset name when versions are ambiguous")
parser.add_argument("--serving-dir", default="artifacts", help="Directory used by the API for the serving artifact")
parser.add_argument("--registry", default="artifacts/model_registry.json")
args = parser.parse_args()

entry = promote_model(args.version, args.serving_dir, args.registry, args.dataset)
print(f"Promoted {entry['model_version']} from {entry['artifact_dir']} to {args.serving_dir}")
