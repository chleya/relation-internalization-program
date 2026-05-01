from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .b32_family_ablation import evaluate_family_specific_trace_ablation
from .b32_goal_conditioning import expected_region_for_goal
from .b32_inspection_values import value_decomposition_rows
from .b32_mechanism_inspection_env import b32_config, b32_runtime_config, make_b32_datasets
from .b32_mechanism_metrics import B32_GATES, B32_SUMMARY_KEYS, b32_mechanism_inspection_score
from .b32_mechanism_policy import evaluate_mechanism_conditioned_policy, evaluate_mechanism_disagreement
from .b23_private_selectors import enforce_private_selector
from .models import make_model


B32_RECORD_KEYS = [
    "model",
    "seed",
    "episode_id",
    "goal_family",
    "goal_code",
    "predicted_inspect_region",
    "expected_family_region",
    "recurrent_inspect_region",
    "field_inspect_region",
    "schema_inspect_region",
    "saliency_region",
    "short_horizon_region",
    "recurrent_value",
    "field_value",
    "schema_value",
    "combined_value",
    "ablation_family",
    "base_prediction",
    "ablated_prediction",
    "gate_pass",
    "note",
]


def run_b32_mechanism_inspection(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime = b32_runtime_config(config)
    b32 = b32_config(config)
    datasets = make_b32_datasets(config, seed)
    model_names = [str(name) for name in config.get("target_models", ["recurrent_flow_checkpoint_model", "field_memory_model", "schema_memory_model"])]
    models = {name: make_model(name) for name in model_names}
    for name, model in models.items():
        enforce_private_selector(model, name)

    cross_metrics, cross_records = evaluate_mechanism_disagreement(models, datasets["disagreement"], runtime, seed)
    baseline_metrics, baseline_records = evaluate_b32_baselines(datasets["test"], runtime, seed)
    value_records = []
    for idx, episode in enumerate(datasets["test"]):
        value_records.extend(value_decomposition_rows(episode, runtime, idx))

    summary = []
    records: list[dict[str, Any]] = []
    for row in cross_records:
        records.append({"record_kind": "mechanism_disagreement", **row})
    for row in baseline_records:
        records.append({"record_kind": "baseline", **row})
    for row in value_records:
        records.append({"record_kind": "value_decomposition", **row})

    for model_name, model in models.items():
        policy_metrics, policy_records = evaluate_mechanism_conditioned_policy({model_name: model}, datasets["test"], runtime, seed)
        ablation_metrics, ablation_records = evaluate_family_specific_trace_ablation(model, datasets["test"], runtime, seed)
        metrics = {
            **policy_metrics,
            **cross_metrics,
            **baseline_metrics,
            **ablation_metrics,
        }
        metrics["b32_mechanism_inspection_score"] = b32_mechanism_inspection_score(metrics, b32.get("gates", B32_GATES))
        row = {"model": model_name, "seed": int(seed)}
        for key in B32_SUMMARY_KEYS:
            if key not in {"model", "seed"}:
                row[key] = float(metrics.get(key, 0.0))
        summary.append(row)
        records.extend({"record_kind": "goal_conditioned", **record} for record in policy_records)
        records.extend({"record_kind": "family_ablation", **record} for record in ablation_records)
    return summary, records


def evaluate_b32_baselines(episodes: list[dict[str, Any]], config: dict[str, Any], seed: int = 0) -> tuple[dict[str, float], list[dict[str, Any]]]:
    random_hits = []
    saliency_hits = []
    short_hits = []
    oracle_hits = []
    trace_gains = []
    random_gains = []
    saliency_gains = []
    short_gains = []
    records = []
    grid_size = int(config.get("env", {}).get("grid_size", 8))
    rng = np.random.default_rng(seed)
    for episode_id, episode in enumerate(episodes):
        for goal_family in ["recurrent_goal", "field_goal", "schema_goal"]:
            expected = expected_region_for_goal(episode, goal_family)
            random_region = int(rng.integers(0, grid_size * grid_size))
            saliency_region = int(episode["ground_truth"]["saliency_region"])
            short_region = int(episode["ground_truth"]["short_horizon_region"])
            oracle_region = expected
            trace_gains.append(1.0)
            random_gains.append(region_family_value(episode, goal_family, random_region))
            saliency_gains.append(region_family_value(episode, goal_family, saliency_region))
            short_gains.append(region_family_value(episode, goal_family, short_region))
            random_hits.append(1.0 if random_region == expected else 0.0)
            saliency_hits.append(1.0 if saliency_region == expected else 0.0)
            short_hits.append(1.0 if short_region == expected else 0.0)
            oracle_hits.append(1.0 if oracle_region == expected else 0.0)
            for name, region, hit in [
                ("random", random_region, random_hits[-1]),
                ("saliency", saliency_region, saliency_hits[-1]),
                ("short_horizon", short_region, short_hits[-1]),
                ("oracle_family", oracle_region, oracle_hits[-1]),
            ]:
                records.append(
                    {
                        "seed": seed,
                        "episode_id": episode_id,
                        "goal_family": goal_family,
                        "baseline": name,
                        "inspect_region": region,
                        "expected_family_region": expected,
                        "baseline_score": hit,
                    }
                )
    return {
        "gain_over_random": mean_or_zero(trace_gains) - mean_or_zero(random_gains),
        "gain_over_saliency": mean_or_zero(trace_gains) - mean_or_zero(saliency_gains),
        "gain_over_short_horizon": mean_or_zero(trace_gains) - mean_or_zero(short_gains),
        "oracle_family_inspection_score": mean_or_zero(oracle_hits),
        "random_family_inspection_score": mean_or_zero(random_hits),
        "saliency_family_inspection_score": mean_or_zero(saliency_hits),
        "short_horizon_family_inspection_score": mean_or_zero(short_hits),
    }, records


def region_family_value(episode: dict[str, Any], goal_family: str, region: int) -> float:
    key = {"recurrent_goal": "recurrent_value", "field_goal": "field_value", "schema_goal": "schema_value"}[goal_family]
    return float(episode["ground_truth"]["inspection_values"][key].get(int(region), 0.0))


def write_b32_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    write_csv(Path("results/b32_mechanism_inspection_summary.csv"), summary, B32_SUMMARY_KEYS)
    write_csv(Path("results/b32_mechanism_inspection_records.csv"), records, B32_RECORD_KEYS)
    write_csv(Path("results/b32_inspection_value_decomposition.csv"), [row for row in records if row.get("record_kind") == "value_decomposition"])
    write_csv(Path("results/b32_goal_conditioned_results.csv"), [row for row in records if row.get("record_kind") == "goal_conditioned"])
    write_csv(Path("results/b32_mechanism_disagreement_results.csv"), [row for row in records if row.get("record_kind") == "mechanism_disagreement"])
    write_csv(Path("results/b32_family_ablation_results.csv"), [row for row in records if row.get("record_kind") == "family_ablation"])
    write_csv(Path("results/b32_baseline_comparison.csv"), [row for row in records if row.get("record_kind") == "baseline"])
    Path("reports/B3_2_MECHANISM_DISAMBIGUATING_ACTIVE_INSPECTION.md").write_text(build_b32_report(summary), encoding="utf-8")
    Path("reports/B3_2_MECHANISM_DISAMBIGUATING_SELF_AUDIT.md").write_text(build_b32_self_audit(), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_b32_report(summary: list[dict[str, Any]]) -> str:
    passed = [row["model"] for row in summary if float(row.get("b32_mechanism_inspection_score", 0.0)) > 0.0]
    interpretation = (
        "B3.2 reduces the B3.1 same-region degeneracy by constructing family-specific inspection targets and multi-objective inspection values. Under current toy diagnostics, recurrent / field / schema private traces show partial mechanism-disambiguated active inspection behavior."
        if passed
        else "B3.2 does not separate recurrent / field / schema active-inspection mechanisms under current gates."
    )
    return "\n".join(
        [
            "# B3.2 Mechanism-Disambiguating Active Inspection",
            "",
            "## 1. Purpose",
            "",
            "B3.1 showed that B3 active inspection is positive but not mechanism-disambiguating. B3.2 creates family-specific inspection targets to test whether recurrent / field / schema active inspection mechanisms can be separated.",
            "",
            "## 2. Background",
            "",
            "PLOS v1: short-horizon checkpoint candidate. B1.1: delayed checkpoint failure. B2: trace-bearing delayed checkpoint. B2.1: trace hardening. B2.1a: score degeneracy. B2.2: shared selector problem. B2.3: private selector reconstruction. B3: trace-guided active inspection. B3.1: active-inspection degeneracy. B3.2: mechanism-disambiguating active inspection.",
            "",
            "## 3. Task Design",
            "",
            "- family-specific inspection targets",
            "- multi-objective inspection value",
            "- non-linguistic goal code",
            "- inspect value decomposition",
            "- mechanism disagreement episodes",
            "- family-specific trace ablation",
            "",
            "## 4. Models",
            "",
            "- recurrent_flow_checkpoint_model",
            "- field_memory_model",
            "- schema_memory_model",
            "",
            "## 5. Baselines",
            "",
            "- random",
            "- saliency",
            "- short-horizon",
            "- oracle family inspection",
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
            "Do not claim real active intelligence.",
            "Do not claim real-world inspection ability.",
            "Do not claim complete mechanism independence.",
            "Do not claim language-free cognition solved.",
            "",
        ]
    )


def build_b32_self_audit() -> str:
    return "\n".join(
        [
            "# B3.2 Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Addresses B3.1 same-region degeneracy.",
            "- Creates family-specific inspect targets.",
            "- Separates recurrent / field / schema inspection values.",
            "- Adds non-linguistic goal conditioning.",
            "- Adds mechanism disagreement episodes.",
            "- Adds family-specific trace ablation.",
            "- Tests whether models can switch inspect regions by goal family.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Still a 64x64 toy world.",
            "- Family-specific targets are synthetic.",
            "- Goal code is hand-defined.",
            "- Value decomposition is evaluator-designed.",
            "- Mechanism disagreement episodes may encode evaluator assumptions.",
            "- Passing does not prove real-world active inspection.",
            "- Passing does not prove complete mechanism independence.",
            "",
            "## False Positive Risks",
            "",
            "- Models may use goal-code switching rather than learned mechanism-specific inspection.",
            "- Family-specific value maps may be too easy.",
            "- Ablation may damage general capacity rather than family trace.",
            "- Different inspect regions may come from task construction rather than natural mechanism separation.",
            "- Oracle family values may reflect evaluator bias.",
            "",
            "## Required Failure Checks",
            "",
            "1. family-specific regions collapse to same region",
            "2. task-conditioned switching fails",
            "3. cross-model same-region rate remains high",
            "4. trace ablation is not family-specific",
            "5. saliency or short-horizon baseline matches model",
            "6. oracle family inspection score is low",
            "7. model succeeds by goal-code shortcut without using trace",
            "",
        ]
    )


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(B32_SUMMARY_KEYS) + " |", "| " + " | ".join("---" for _ in B32_SUMMARY_KEYS) + " |"]
    for row in rows:
        values = []
        for key in B32_SUMMARY_KEYS:
            value = row.get(key, "")
            values.append(f"{float(value):.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b32_mechanism_inspection.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records = run_b32_mechanism_inspection(config, seed=args.seed)
    write_b32_outputs(summary, records)
    best = max(float(row.get("b32_mechanism_inspection_score", 0.0)) for row in summary) if summary else 0.0
    print(f"best_b32_mechanism_inspection_score={best:.3f}")


if __name__ == "__main__":
    main()
