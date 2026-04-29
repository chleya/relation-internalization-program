from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import make_dataset
from .model import accuracy, hidden_states, train_mlp
from .probe import train_probe
from .relation_bridge import extract_relation_table, relation_alignment, table_resource_accuracy


def gated_extraction_score(metrics: dict[str, object]) -> float:
    table_ood = float(metrics["table_ood_accuracy"])
    table_attack = float(metrics["table_spurious_attack_accuracy"])
    alignment = float(metrics["table_relation_alignment"])
    edit_ok = bool(metrics["edit_success"])
    gates = [
        table_ood >= 0.9,
        table_attack >= 0.9,
        alignment >= 0.9,
        edit_ok,
    ]
    if not all(gates):
        return 0.0
    return float(0.3 * table_ood + 0.3 * table_attack + 0.3 * alignment + 0.1)


def run(seed: int = 0, train_mode: str = "base", output: str = "results/extracted_seed0.json") -> dict[str, object]:
    train = make_dataset(1200, seed=seed, mode=train_mode)
    ood = make_dataset(300, seed=seed + 1, mode="ood")
    attack = make_dataset(300, seed=seed + 2, mode="spurious_attack")

    model = train_mlp(train.x, train.y_resource, seed=seed)
    relation_probe = train_probe(hidden_states(model, train.x), train.y_relation, seed=seed)
    table = extract_relation_table(model, relation_probe)

    target_context = {"texture": "A", "wet": "dry", "color": "blue", "odor": "weak"}
    rules_before_edit = table.describe_relations()
    table_ood_accuracy = table_resource_accuracy(table, ood.contexts, ood.y_resource)
    table_attack_accuracy = table_resource_accuracy(table, attack.contexts, attack.y_resource)
    table_alignment = relation_alignment(table)
    before = table.act(target_context)
    table.edit_rule({"texture": "A", "wet": "dry"}, "poison")
    after = table.act(target_context)

    metrics: dict[str, object] = {
        "seed": seed,
        "train_mode": train_mode,
        "model_ood_accuracy": accuracy(model, ood.x, ood.y_resource),
        "model_spurious_attack_accuracy": accuracy(model, attack.x, attack.y_resource),
        "table_ood_accuracy": table_ood_accuracy,
        "table_spurious_attack_accuracy": table_attack_accuracy,
        "table_relation_alignment": table_alignment,
        "edit_before_action": before,
        "edit_after_action": after,
        "edit_success": before == "eat" and after == "avoid",
        "rules_before_edit": rules_before_edit,
        "rules_after_edit": table.describe_relations(),
    }
    metrics["gated_extraction_score"] = gated_extraction_score(metrics)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-mode", default="base", choices=["base", "shortcut", "ood", "spurious_attack"])
    parser.add_argument("--output", default="results/extracted_seed0.json")
    args = parser.parse_args()
    print(json.dumps(run(seed=args.seed, train_mode=args.train_mode, output=args.output), indent=2))


if __name__ == "__main__":
    main()
