from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .b3_active_inspection_baselines import evaluate_b3_baselines
from .b3_active_inspection_env import b3_config, b3_runtime_config, make_b3_datasets
from .b3_active_inspection_metrics import B3_GATES, B3_SUMMARY_KEYS, b3_active_inspection_score
from .b3_information_gain import compute_information_gain_after_inspection
from .b3_inspection_policy import evaluate_trace_guided_policy, trace_guided_inspection_policy
from .b3_trace_ablation_eval import evaluate_inspection_after_trace_ablation
from .b23_private_selectors import enforce_private_selector
from .models import make_model


B3_RECORD_KEYS = [
    "model",
    "seed",
    "episode_id",
    "episode_type",
    "delay",
    "true_trace_region",
    "saliency_region",
    "oracle_best_inspect_region",
    "predicted_inspect_region",
    "policy_source",
    "trace_family",
    "pre_inspection_error",
    "post_inspection_error",
    "absolute_gain",
    "relative_gain",
    "baseline_random_gain",
    "baseline_saliency_gain",
    "baseline_short_horizon_gain",
    "trace_ablated_region",
    "trace_ablated_gain",
    "gate_pass",
    "note",
]


def run_b3_active_inspection(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime = b3_runtime_config(config)
    datasets = make_b3_datasets(config, seed)
    b3 = b3_config(config)
    gates = b3.get("gates", B3_GATES)
    target_models = [str(name) for name in config.get("target_models", ["recurrent_flow_checkpoint_model", "field_memory_model", "schema_memory_model"])]
    baseline_metrics, baseline_records = evaluate_b3_baselines(datasets["test"], runtime, seed)
    summary = []
    records: list[dict[str, Any]] = []
    for baseline_record in baseline_records:
        records.append({"record_kind": "baseline", "seed": seed, **baseline_record})

    for model_name in target_models:
        model = make_model(model_name)
        enforce_private_selector(model, model_name)
        policy_metrics, policy_records = evaluate_trace_guided_policy(model, datasets["test"], runtime)
        conflict_metrics, _ = evaluate_trace_guided_policy(model, datasets["conflict"], runtime)
        ood_metrics, _ = evaluate_trace_guided_policy(model, datasets["ood"], runtime)
        info_metrics, info_records = evaluate_information_gain(model, datasets["test"], runtime, baseline_metrics, seed)
        ablation_metrics, ablation_records = evaluate_inspection_after_trace_ablation(model, datasets["test"], runtime)
        metrics = {
            **policy_metrics,
            "trace_vs_saliency_rejection": conflict_metrics["trace_vs_saliency_rejection"],
            "delay_ood_inspection_accuracy": ood_metrics["trace_guided_inspection_accuracy"],
            **info_metrics,
            **baseline_metrics,
            **ablation_metrics,
        }
        metrics["inspection_value_gain_over_random"] = metrics["delayed_information_gain"] - baseline_metrics["random_information_gain"]
        metrics["inspection_value_gain_over_saliency"] = metrics["delayed_information_gain"] - baseline_metrics["saliency_information_gain"]
        metrics["inspection_value_gain_over_short_horizon"] = metrics["delayed_information_gain"] - baseline_metrics["short_horizon_information_gain"]
        metrics["b3_active_inspection_score"] = b3_active_inspection_score(metrics, gates)
        row = {"model": model_name, "seed": int(seed)}
        for key in B3_SUMMARY_KEYS:
            if key not in {"model", "seed"}:
                row[key] = float(metrics.get(key, 0.0))
        summary.append(row)

        ablation_by_episode = {int(row["episode_id"]): row for row in ablation_records}
        for record in info_records:
            episode_id = int(record["episode_id"])
            ablated = ablation_by_episode.get(episode_id, {})
            records.append(
                {
                    "record_kind": "main",
                    "model": model_name,
                    "seed": seed,
                    **record,
                    "trace_ablated_region": ablated.get("trace_ablated_region", ""),
                    "trace_ablated_gain": "",
                    "gate_pass": int(record.get("predicted_inspect_region") == record.get("oracle_best_inspect_region")),
                }
            )
        for row_ablation in ablation_records:
            records.append({"record_kind": "ablation", "model": model_name, "seed": seed, **row_ablation})
    return summary, records


def evaluate_information_gain(
    model: Any,
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    baseline_metrics: dict[str, float],
    seed: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    gains = []
    relative = []
    records = []
    for idx, episode in enumerate(episodes):
        policy = trace_guided_inspection_policy(model, episode, config)
        region = int(policy["inspect_region"])
        gain = compute_information_gain_after_inspection(model, episode, region, config)
        gains.append(gain["absolute_gain"])
        relative.append(gain["relative_gain"])
        gt = episode["ground_truth"]
        records.append(
            {
                "episode_id": idx,
                "episode_type": gt.get("episode_type", ""),
                "delay": gt.get("delay", ""),
                "true_trace_region": int(gt["true_trace_region"]),
                "saliency_region": int(gt["saliency_region"]),
                "oracle_best_inspect_region": int(gt["oracle_best_inspect_region"]),
                "predicted_inspect_region": region,
                "policy_source": policy["policy_source"],
                "trace_family": policy["trace_family"],
                "pre_inspection_error": gain["pre_inspection_error"],
                "post_inspection_error": gain["post_inspection_error"],
                "absolute_gain": gain["absolute_gain"],
                "relative_gain": gain["relative_gain"],
                "baseline_random_gain": baseline_metrics["random_information_gain"],
                "baseline_saliency_gain": baseline_metrics["saliency_information_gain"],
                "baseline_short_horizon_gain": baseline_metrics["short_horizon_information_gain"],
                "note": "trace-guided inspection",
            }
        )
    return {
        "delayed_information_gain": mean_or_zero(gains),
        "relative_information_gain": mean_or_zero(relative),
    }, records


def write_b3_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    with Path("results/b3_active_inspection_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=B3_SUMMARY_KEYS)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row.get(key, "") for key in B3_SUMMARY_KEYS})
    with Path("results/b3_active_inspection_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=B3_RECORD_KEYS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in B3_RECORD_KEYS})
    write_csv(Path("results/b3_baseline_comparison.csv"), [row for row in records if row.get("record_kind") == "baseline"])
    write_csv(Path("results/b3_trace_ablation_results.csv"), [row for row in records if row.get("record_kind") == "ablation"])
    write_csv(Path("results/b3_information_gain_records.csv"), [row for row in records if row.get("record_kind") == "main"])
    Path("reports/B3_DELAYED_TRACE_GUIDED_ACTIVE_INSPECTION.md").write_text(build_b3_report(summary), encoding="utf-8")
    Path("reports/B3_ACTIVE_INSPECTION_SELF_AUDIT.md").write_text(build_b3_self_audit(), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_b3_report(summary: list[dict[str, Any]]) -> str:
    passed = [row["model"] for row in summary if float(row.get("b3_active_inspection_score", 0.0)) > 0.0]
    interpretation = (
        "Private delayed trace can guide budgeted active inspection in the toy PLOS world."
        if passed
        else "Private delayed trace representation is not yet sufficient for active inspection under current gates."
    )
    return "\n".join(
        [
            "# B3 Delayed Trace-Guided Active Inspection",
            "",
            "## 1. Purpose",
            "",
            "B3 tests whether B2.3 private delayed traces can guide active inspection under budget.",
            "",
            "## 2. Background",
            "",
            "PLOS v1: short-horizon checkpoint candidate. B1.1: delayed checkpoint failure. B2: trace-bearing path solves delayed checkpoint. B2.1: trace hardening. B2.1a: identical-score degeneracy. B2.2: shared selector problem. B2.3: private selector reconstruction. B3: trace-guided active inspection.",
            "",
            "## 3. Task",
            "",
            "The model has limited inspection budget and must choose one region. Correct inspection should follow delayed trace, not saliency or short-horizon checkpoint.",
            "",
            "## 4. Models",
            "",
            "- recurrent_flow_checkpoint_model",
            "- field_memory_model",
            "- schema_memory_model",
            "",
            "## 5. Baselines",
            "",
            "- random inspection",
            "- saliency inspection",
            "- short-horizon checkpoint inspection",
            "- oracle inspection",
            "",
            "## 6. Metrics",
            "",
            "- trace-guided inspection accuracy",
            "- trace-vs-saliency rejection",
            "- delayed information gain",
            "- gain over random",
            "- gain over saliency",
            "- gain over short-horizon",
            "- trace ablation inspection drop",
            "- delay OOD inspection",
            "",
            "## 7. Results",
            "",
            markdown_table(summary),
            "",
            "## 8. Interpretation",
            "",
            interpretation,
            "",
            "B3 does not re-audit mechanism separation. Identical B3 scores across the private-selector models mean that all three solve the current active-inspection task under the same inspection-value protocol; they are not additional evidence of fully independent mechanisms.",
            "",
            "## 9. Claim Boundary",
            "",
            "Do not claim general active intelligence.",
            "Do not claim real-world robot inspection.",
            "Do not claim human-like attention.",
            "Do not claim language-free cognition solved.",
            "",
        ]
    )


def build_b3_self_audit() -> str:
    return "\n".join(
        [
            "# B3 Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Moves from delayed trace representation to trace-guided active inspection.",
            "- Tests budgeted inspect/action.",
            "- Adds trace-vs-saliency conflict.",
            "- Adds delayed information gain.",
            "- Compares random, saliency, short-horizon, and oracle baselines.",
            "- Tests inspection after private trace ablation.",
            "- Keeps trace mechanism fixed from B2.3.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Still a 64x64 toy world.",
            "- Inspect action is simplified.",
            "- Oracle inspection value comes from simulator.",
            "- Active inspection is not full control.",
            "- Trace-guided inspection may still exploit generator regularities.",
            "- B3 does not prove real-world engineering inspection.",
            "- Passing B3 does not prove general active intelligence.",
            "- Identical B3 scores do not prove independent mechanisms.",
            "",
            "## False Positive Risks",
            "",
            "- Model may select trace region without actual information gain.",
            "- Saliency baseline may be too weak.",
            "- Short-horizon baseline may be poorly implemented.",
            "- Oracle information gain may encode evaluator bias.",
            "- Trace ablation may damage unrelated model capacity.",
            "- Inspect patch may reveal too much information.",
            "",
            "## Required Failure Checks",
            "",
            "1. random baseline passes",
            "2. saliency baseline matches trace model",
            "3. short-horizon baseline matches trace model",
            "4. trace ablation does not reduce inspection performance",
            "5. selected trace region does not improve delayed prediction",
            "6. OOD delay inspection collapses",
            "7. oracle score is low",
            "",
        ]
    )


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(B3_SUMMARY_KEYS) + " |", "| " + " | ".join("---" for _ in B3_SUMMARY_KEYS) + " |"]
    for row in rows:
        values = []
        for key in B3_SUMMARY_KEYS:
            value = row.get(key, "")
            values.append(f"{float(value):.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b3_active_inspection.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records = run_b3_active_inspection(config, seed=args.seed)
    write_b3_outputs(summary, records)
    best = max(float(row.get("b3_active_inspection_score", 0.0)) for row in summary) if summary else 0.0
    print(f"best_b3_active_inspection_score={best:.3f}")


if __name__ == "__main__":
    main()
