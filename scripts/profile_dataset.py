import argparse
import json

from credit_simulator.datasets import get_dataset


parser = argparse.ArgumentParser(description="Load and profile a supported credit-risk dataset")
parser.add_argument("--dataset", choices=["uci_default", "german_credit", "give_me_some_credit"], default="uci_default")
parser.add_argument("--raw-dir", default="data/raw")
args = parser.parse_args()
bundle = get_dataset(args.dataset, args.raw_dir)
print(json.dumps({"name": bundle.name, "rows": len(bundle.frame), "columns": len(bundle.frame.columns), "target": bundle.target, "target_rate": float(bundle.frame[bundle.target].mean()), "protected_attributes": bundle.protected_attributes, "feature_columns": bundle.feature_columns, "metadata": bundle.metadata}, indent=2, default=str))
