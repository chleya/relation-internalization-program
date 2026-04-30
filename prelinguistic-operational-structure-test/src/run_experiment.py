from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from .data import generate_dataset
from .null_controls import evaluate_null_controls
from .ood import make_all_ood
from .scoring import merge_scores
from .train import train_model


def run_one(config_path: str, model_name: str, seed: int) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    train_dataset = generate_dataset(config, "train", seed)
    test_dataset = generate_dataset(config, "test", seed + 10000)
    ood_datasets = make_all_ood(config, seed + 20000)
    model = train_model(model_name, train_dataset, config)

    from .evaluate import evaluate_all

    results = evaluate_all(model, {"test": test_dataset, "ood": ood_datasets}, config)
    metrics = merge_scores(results["behavior"], results["structure"], results["ood"], config["gates"])
    null_controls = evaluate_null_controls(model, config, seed + 30000)
    return {
        "model": model_name,
        "seed": seed,
        **metrics,
        **null_controls,
    }


def write_single_outputs(row: dict[str, Any]) -> None:
    Path("results").mkdir(exist_ok=True)
    path = Path("results/records.csv")
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default="schema_model")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    row = run_one(args.config, args.model, args.seed)
    write_single_outputs(row)
    print(f"{row['model']}: plos_candidate_score={row['plos_candidate_score']:.3f}")


if __name__ == "__main__":
    main()
