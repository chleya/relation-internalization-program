from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from .evaluate import evaluate_all
from .train import train_model


def run(config_path: str, model_name: str, seed: int) -> dict[str, float | str | int]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model = train_model(model_name, seed, config)
    metrics = evaluate_all(model, config, seed, model_name)
    row = {"model": model_name, "seed": seed, **metrics}
    Path("results").mkdir(exist_ok=True)
    with Path("results/records.csv").open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default="edit_pressure_training")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    row = run(args.config, args.model, args.seed)
    print(f"{row['model']} seed={row['seed']} gated_internalization_score={row['gated_internalization_score']:.3f}")


if __name__ == "__main__":
    main()
