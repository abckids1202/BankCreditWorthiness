import argparse
import json

from credit_simulator.datasets import get_dataset
from credit_simulator.generic_training import train_tabular
from credit_simulator.training import train


parser = argparse.ArgumentParser(description="Train the educational credit-risk models")
parser.add_argument("--dataset", choices=["uci_default", "german_credit", "give_me_some_credit"], default="uci_default")
parser.add_argument("--raw-dir", default="data/raw")
parser.add_argument("--output-dir", default=None)
parser.add_argument("--config", default=None, help="Optional YAML configuration for primary UCI training")
args = parser.parse_args()


if __name__ == "__main__":
    if args.dataset == "uci_default":
        output_dir = args.output_dir or "artifacts"
        metadata = train(raw_dir=args.raw_dir, output_dir=output_dir, config_path=args.config)
        print(json.dumps({"status": "trained", "dataset": args.dataset, "model_version": metadata["model_version"], "policy_version": metadata["policy_version"], "artifact_fingerprint": metadata["artifact_fingerprint"], "artifact_dir": output_dir, "metrics_test": metadata["metrics_test"]}, sort_keys=True, default=str))
    else:
        bundle = get_dataset(args.dataset, args.raw_dir)
        output_dir = args.output_dir or f"artifacts/{args.dataset}"
        metadata = train_tabular(bundle, output_dir)
        print(json.dumps({"status": "trained", "dataset": args.dataset, "model_version": metadata["model_version"], "policy_version": metadata["policy_version"], "artifact_fingerprint": metadata["artifact_fingerprint"], "artifact_dir": output_dir, "metrics_test": metadata["metrics_test"]}, sort_keys=True, default=str))

