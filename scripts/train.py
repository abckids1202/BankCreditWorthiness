import argparse

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
        train(raw_dir=args.raw_dir, output_dir=args.output_dir or "artifacts", config_path=args.config)
    else:
        bundle = get_dataset(args.dataset, args.raw_dir)
        metadata = train_tabular(bundle, args.output_dir or f"artifacts/{args.dataset}")
        print(f"Trained {args.dataset}: {metadata['metrics_test']}")

