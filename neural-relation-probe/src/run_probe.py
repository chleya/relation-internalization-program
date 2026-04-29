from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import make_dataset
from .model import accuracy, hidden_states, train_mlp
from .probe import (
    intervention_drop,
    probe_accuracy,
    random_intervention_drop,
    random_label_control,
    subspace_intervention_drop,
    top_probe_dimensions,
    train_probe,
)


def gated_score(metrics: dict[str, float]) -> float:
    gates = [
        metrics["ood_accuracy"] >= 0.8,
        metrics["spurious_attack_accuracy"] >= 0.8,
        metrics["probe_selectivity"] >= 0.2,
        metrics["relation_subspace_drop"] >= metrics["nuisance_subspace_drop"] + 0.1,
    ]
    if not all(gates):
        return 0.0
    return float(
        0.25 * metrics["ood_accuracy"]
        + 0.25 * metrics["spurious_attack_accuracy"]
        + 0.25 * metrics["probe_selectivity"]
        + 0.25 * min(1.0, metrics["relation_subspace_drop"])
    )


def run(
    seed: int = 0,
    output: str = "results/metrics.json",
    train_mode: str = "base",
    train_size: int = 1200,
) -> dict[str, float]:
    train = make_dataset(train_size, seed=seed, mode=train_mode)
    test = make_dataset(300, seed=seed + 1, mode="base")
    ood = make_dataset(300, seed=seed + 2, mode="ood")
    attack = make_dataset(300, seed=seed + 3, mode="spurious_attack")

    model = train_mlp(train.x, train.y_resource, seed=seed)
    h_train = hidden_states(model, train.x)
    h_test = hidden_states(model, test.x)

    relation_probe = train_probe(h_train, train.y_relation, seed=seed)
    nuisance_probe = train_probe(h_train, train.y_nuisance, seed=seed + 1)
    probe_acc = probe_accuracy(relation_probe, h_test, test.y_relation)
    random_probe_acc = random_label_control(h_train, train.y_relation, seed=seed + 10)
    dims = top_probe_dimensions(relation_probe, k=4)

    metrics = {
        "train_mode": train_mode,
        "train_size": float(train_size),
        "train_accuracy": accuracy(model, train.x, train.y_resource),
        "test_accuracy": accuracy(model, test.x, test.y_resource),
        "ood_accuracy": accuracy(model, ood.x, ood.y_resource),
        "spurious_attack_accuracy": accuracy(model, attack.x, attack.y_resource),
        "relation_probe_accuracy": probe_acc,
        "random_label_probe_accuracy": random_probe_acc,
        "probe_selectivity": max(0.0, probe_acc - random_probe_acc),
        "relation_dimension_zero_drop": intervention_drop(model, test.x, test.y_resource, dims),
        "random_dimension_zero_drop": random_intervention_drop(
            model, test.x, test.y_resource, k=len(dims), seed=seed + 20
        ),
        "relation_subspace_drop": subspace_intervention_drop(
            model, test.x, test.y_resource, relation_probe
        ),
        "nuisance_subspace_drop": subspace_intervention_drop(
            model, test.x, test.y_resource, nuisance_probe
        ),
    }
    metrics["gated_neural_relation_score"] = gated_score(metrics)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/metrics.json")
    parser.add_argument("--train-mode", default="base", choices=["base", "shortcut", "ood", "spurious_attack"])
    parser.add_argument("--train-size", type=int, default=1200)
    args = parser.parse_args()
    metrics = run(
        seed=args.seed,
        output=args.output,
        train_mode=args.train_mode,
        train_size=args.train_size,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
