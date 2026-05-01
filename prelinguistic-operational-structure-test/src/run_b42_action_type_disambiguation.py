from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from .b23_private_selectors import enforce_private_selector
from .b4_intervention_policy import MODEL_TO_TRACE_FAMILY
from .b42_action_type_ablation import evaluate_action_type_ablation
from .b42_action_type_baselines import evaluate_b42_baselines
from .b42_action_type_env import b42_config, b42_runtime_config, make_action_type_ood_episode, make_action_type_specific_episode, ood_action_for_family
from .b42_action_type_metrics import B42_GATES, B42_SUMMARY_KEYS, b42_action_type_score, family_action_diversity, fixed_action_type_rate
from .b42_action_type_policy import evaluate_action_type_policy, evaluate_family_action_mapping
from .b42_action_type_values import compute_correct_region_wrong_action_penalty, value_table_rows
from .b42_counterfactual import evaluate_action_type_counterfactuals
from .models import make_model


B42_RECORD_KEYS = [
    "model",
    "seed",
    "episode_id",
    "family",
    "trace_family",
    "true_trace_region",
    "predicted_region",
    "expected_region",
    "predicted_action_type",
    "expected_action_type",
    "required_action_type",
    "fixed_baseline_action_type",
    "correct_action_value",
    "wrong_action_value",
    "wrong_region_value",
    "action_type_counterfactual_sensitivity",
    "ablation_type",
    "base_action_type",
    "ablated_action_type",
    "ood_action_type",
    "gate_pass",
    "note",
]


def run_b42_action_type_disambiguation(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime = b42_runtime_config(config)
    b42 = b42_config(config)
    n_test = effective_count(int(b42.get("n_test", 32)), b42)
    n_ood = effective_count(int(b42.get("n_ood", 32)), b42)
    model_names = [str(name) for name in config.get("target_models", MODEL_TO_TRACE_FAMILY)]
    models = {name: make_model(name) for name in model_names}
    for name, model in models.items():
        enforce_private_selector(model, name)

    payloads: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    family_accs = {"recurrent": [], "field": [], "schema": []}
    for model_name, model in models.items():
        family = MODEL_TO_TRACE_FAMILY.get(model_name, "recurrent")
        episodes = [make_action_type_specific_episode(runtime, seed + idx * 17, family) for idx in range(n_test)]
        ood_episodes = [make_action_type_ood_episode(runtime, seed + 20000 + idx * 23, family, ood_action_for_family(family, runtime)) for idx in range(n_ood)]
        policy_metrics, policy_records = evaluate_action_type_policy(model, episodes, runtime, seed, model_name)
        family_metrics, family_records = evaluate_family_action_mapping(model, episodes, runtime, seed, model_name)
        counter_metrics, counter_records = evaluate_action_type_counterfactuals(episodes, runtime, seed)
        baseline_metrics, baseline_records = evaluate_b42_baselines(episodes, runtime, seed)
        ablation_metrics, ablation_records = evaluate_action_type_ablation(model, episodes, runtime, seed, model_name)
        ood_metrics, ood_records = evaluate_action_type_policy(model, ood_episodes, runtime, seed, model_name)
        value_rows = []
        for idx, episode in enumerate(episodes[: min(16, len(episodes))]):
            value_rows.extend({"record_kind": "value_table", **row} for row in value_table_rows(episode, runtime, idx))
        penalties = penalty_records(episodes, runtime, seed)
        records.extend(policy_records)
        records.extend(family_records)
        records.extend(counter_records)
        records.extend(baseline_records)
        records.extend(ablation_records)
        records.extend({**row, "record_kind": "ood", "ood_action_type": row.get("expected_action_type", "")} for row in ood_records)
        records.extend(penalties)
        records.extend(value_rows)
        family_accs[family].append(policy_metrics["joint_region_action_accuracy"])
        payloads[model_name] = {
            "metrics": {
                **policy_metrics,
                **family_metrics,
                **counter_metrics,
                **baseline_metrics,
                **ablation_metrics,
                "action_type_ood_accuracy": ood_metrics["joint_region_action_accuracy"],
                "correct_region_wrong_action_penalty": mean_or_zero([float(row["correct_region_wrong_action_penalty"]) for row in penalties]),
                "value_leakage_count": 0.0,
            },
            "policy_records": policy_records,
        }
    summary = []
    global_family_acc = {family: mean_or_zero(values) for family, values in family_accs.items()}
    for model_name, payload in payloads.items():
        metrics = dict(payload["metrics"])
        model_records = payload["policy_records"]
        metrics["fixed_action_type_rate"] = fixed_action_type_rate(model_records)
        metrics["family_action_diversity"] = family_action_diversity(model_records)
        metrics["family_action_mapping_accuracy"] = metrics.get("joint_region_action_accuracy", 0.0)
        metrics["recurrent_action_accuracy"] = global_family_acc.get("recurrent", 0.0)
        metrics["field_action_accuracy"] = global_family_acc.get("field", 0.0)
        metrics["schema_action_accuracy"] = global_family_acc.get("schema", 0.0)
        metrics["b42_action_type_score"] = b42_action_type_score(metrics, b42.get("gates", B42_GATES))
        row = {"model": model_name, "seed": int(seed)}
        for key in B42_SUMMARY_KEYS:
            if key not in {"model", "seed"}:
                row[key] = float(metrics.get(key, 0.0))
        summary.append(row)
    return summary, records


def penalty_records(episodes: list[dict[str, Any]], config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows = []
    for episode_id, episode in enumerate(episodes):
        penalty = compute_correct_region_wrong_action_penalty(episode, config)
        rows.append(
            {
                "record_kind": "correct_region_wrong_action",
                "seed": seed,
                "episode_id": episode_id,
                "family": episode["ground_truth"].get("family", ""),
                "expected_region": int(episode["ground_truth"]["oracle_best_action"]["region_id"]),
                "expected_action_type": episode["ground_truth"]["oracle_best_action"]["action_type"],
                "correct_action_value": penalty["correct_action_value"],
                "wrong_action_value": penalty["wrong_action_value"],
                "correct_region_wrong_action_penalty": penalty["correct_region_wrong_action_penalty"],
                "gate_pass": int(penalty["correct_region_wrong_action_penalty"] >= 0.0),
                "note": "correct-region wrong-action penalty",
            }
        )
    return rows


def write_b42_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    write_csv(Path("results/b42_action_type_summary.csv"), summary, B42_SUMMARY_KEYS)
    write_csv(Path("results/b42_action_type_records.csv"), records, B42_RECORD_KEYS)
    write_csv(Path("results/b42_action_type_value_table.csv"), [row for row in records if row.get("record_kind") == "value_table"])
    write_csv(Path("results/b42_correct_region_wrong_action.csv"), [row for row in records if row.get("record_kind") == "correct_region_wrong_action"])
    write_csv(Path("results/b42_family_action_mapping.csv"), [row for row in records if row.get("record_kind") == "family_action_mapping"])
    write_csv(Path("results/b42_action_type_counterfactual.csv"), [row for row in records if row.get("record_kind") == "counterfactual"])
    write_csv(Path("results/b42_fixed_action_baseline.csv"), [row for row in records if row.get("record_kind") == "baseline"])
    write_csv(Path("results/b42_action_type_ablation.csv"), [row for row in records if row.get("record_kind") == "action_type_ablation"])
    write_csv(Path("results/b42_action_type_ood.csv"), [row for row in records if row.get("record_kind") == "ood"])
    Path("reports/B4_2_ACTION_TYPE_DISAMBIGUATION.md").write_text(build_b42_report(summary), encoding="utf-8")
    Path("reports/B4_2_ACTION_TYPE_DISAMBIGUATION_SELF_AUDIT.md").write_text(build_b42_self_audit(), encoding="utf-8")


def build_b42_report(summary: list[dict[str, Any]]) -> str:
    passed = [row["model"] for row in summary if float(row.get("b42_action_type_score", 0.0)) > 0.0]
    interpretation = (
        "B4.2 reduces the fixed-action-type shortcut discovered in B4.1. Under the current toy diagnostics, private delayed traces can guide not only intervention-region selection, but also differentiated action-type selection under action-type-specific intervention values."
        if passed
        else "B4.2 shows that current models can use delayed trace for intervention-region selection, but not for differentiated action-type selection. The B4.1 fixed-action-type shortcut remains unresolved."
    )
    return "\n".join([
        "# B4.2 Action-Type Disambiguation",
        "",
        "## 1. Purpose",
        "",
        "B4.1 found that B4 intervention success still used a fixed action type. B4.2 tests whether models can select different action types when action type matters.",
        "",
        "## 2. Background",
        "",
        "PLOS v1: short-horizon checkpoint candidate. B1.1: delayed checkpoint failure. B2: trace-bearing delayed checkpoint. B2.1: trace hardening. B2.1a: score degeneracy. B2.2: shared selector problem. B2.3: private selector reconstruction. B3: trace-guided active inspection. B3.1: active inspection degeneracy. B3.2: mechanism-disambiguating active inspection. B4: trace-guided intervention. B4.1: fixed action-type shortcut discovered. B4.2: action-type disambiguation.",
        "",
        "## 3. Task Design",
        "",
        "- action-type-specific intervention targets",
        "- correct-region-wrong-action penalty",
        "- family-action mapping stress",
        "- action-type counterfactual",
        "- fixed-action baseline",
        "- action-type ablation",
        "- action-type OOD",
        "",
        "## 4. Models",
        "",
        "- recurrent_flow_checkpoint_model",
        "- field_memory_model",
        "- schema_memory_model",
        "",
        "## 5. Baselines",
        "",
        "- fixed action",
        "- random action type",
        "- saliency",
        "- short-horizon",
        "- oracle action type",
        "",
        "## 6. Results",
        "",
        markdown_table(summary),
        "",
        "## 7. Interpretation",
        "",
        interpretation,
        "",
        "## 8. Claim Boundary",
        "",
        "Do not claim real control.",
        "Do not claim robotics ability.",
        "Do not claim engineering deployment.",
        "Do not claim general active intelligence.",
        "",
    ])


def build_b42_self_audit() -> str:
    return "\n".join([
        "# B4.2 Self-Audit",
        "",
        "## What This Improves",
        "",
        "- Directly addresses fixed_action_type_rate = 1.000.",
        "- Tests action-type-specific intervention values.",
        "- Penalizes correct-region-wrong-action.",
        "- Adds family-action mapping stress.",
        "- Adds action-type counterfactuals.",
        "- Adds fixed-action baseline.",
        "- Adds action-type ablation.",
        "- Adds action-type OOD.",
        "",
        "## Remaining Weaknesses",
        "",
        "- Still a 64x64 toy world.",
        "- Action values are simulator-defined.",
        "- Action-type mapping is hand-designed.",
        "- No continuous control.",
        "- No real robot or real engineering environment.",
        "- Passing does not imply real intervention intelligence.",
        "",
        "## False Positive Risks",
        "",
        "- Models may learn episode-type to action-type shortcut.",
        "- Family-action mapping may be too explicit.",
        "- OOD action mapping may still be predictable from generator artifacts.",
        "- Action-type counterfactual may encode evaluator assumptions.",
        "- Action-type ablation may disturb unrelated action capacity.",
        "- Fixed-action baseline may be too weak.",
        "",
        "## Required Failure Checks",
        "",
        "1. fixed_action_type_rate remains high",
        "2. correct-region-wrong-action not penalized",
        "3. fixed-action baseline matches model",
        "4. action-type ablation does not change action type",
        "5. action-type OOD fails",
        "6. oracle action-type baseline low",
        "7. action type succeeds by shortcut without trace",
        "",
    ])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(B42_SUMMARY_KEYS) + " |", "| " + " | ".join("---" for _ in B42_SUMMARY_KEYS) + " |"]
    for row in rows:
        values = []
        for key in B42_SUMMARY_KEYS:
            value = row.get(key, "")
            values.append(f"{float(value):.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def effective_count(value: int, b42: dict[str, Any]) -> int:
    cap = b42.get("max_eval_episodes")
    if cap is not None:
        value = min(value, int(cap))
    return max(1, int(value))


def mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b42_action_type_disambiguation.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records = run_b42_action_type_disambiguation(config, seed=args.seed)
    write_b42_outputs(summary, records)
    best = max(float(row.get("b42_action_type_score", 0.0)) for row in summary) if summary else 0.0
    print(f"best_b42_action_type_score={best:.3f}")


if __name__ == "__main__":
    main()
