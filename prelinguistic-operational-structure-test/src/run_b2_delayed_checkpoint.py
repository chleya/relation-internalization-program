from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from .b2_delayed_env import make_b2_datasets, resolve_base_config
from .b2_delayed_interventions import evaluate_causal_trace_intervention_with_records
from .b2_delayed_metrics import B2_GATES, B2_SUMMARY_KEYS, b2_delayed_score, evaluate_b2_behavior
from .train import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b2_delayed_checkpoint.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records = run_b2(config, seed=args.seed)
    write_b2_outputs(summary, records)
    best = max(float(row["b2_delayed_score"]) for row in summary) if summary else 0.0
    print(f"best_b2_score={best:.3f}")


def run_b2(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Evaluate whether delayed operational checkpoints require memory/trace/schema substrates.
    """

    base_config = resolve_base_config(config)
    datasets = make_b2_datasets(config, seed)
    target_models = config.get(
        "target_models",
        ["flow_checkpoint_model", "recurrent_flow_checkpoint_model", "field_memory_model", "schema_memory_model"],
    )
    gates = config.get("b2", {}).get("gates", B2_GATES)
    summary: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for model_name in target_models:
        model = train_model(str(model_name), datasets["train"], base_config)
        behavior_metrics, behavior_records = evaluate_b2_behavior(model, datasets, base_config)
        trace_metrics, trace_records = evaluate_causal_trace_intervention_with_records(model, datasets["delayed"], base_config)
        metrics = {**behavior_metrics, **trace_metrics}
        metrics["b2_delayed_score"] = b2_delayed_score(metrics, gates)
        row = {"model": str(model_name), "seed": int(seed)}
        row.update({key: float(metrics.get(key, 0.0)) for key in B2_SUMMARY_KEYS if key not in {"model", "seed"}})
        summary.append(row)

        for record in behavior_records + trace_records:
            enriched = {"model": str(model_name), "seed": int(seed)}
            enriched.update(record)
            records.append(enriched)
    return summary, records


def write_b2_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    with Path("results/b2_delayed_checkpoint_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=B2_SUMMARY_KEYS)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row.get(key, "") for key in B2_SUMMARY_KEYS})

    record_keys = [
        "model",
        "seed",
        "episode_id",
        "episode_type",
        "delay",
        "true_delayed_checkpoint_region",
        "early_saliency_region",
        "predicted_region",
        "selected_delay",
        "correct",
        "ood_type",
        "intervention_type",
        "base_endpoint",
        "intervened_endpoint",
        "endpoint_shift",
    ]
    with Path("results/b2_delayed_checkpoint_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=record_keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in record_keys})

    Path("reports/B2_DELAYED_CHECKPOINT_REPORT.md").write_text(build_b2_report(summary), encoding="utf-8")
    Path("reports/B2_DELAYED_CHECKPOINT_SELF_AUDIT.md").write_text(build_b2_self_audit(), encoding="utf-8")


def build_b2_report(summary: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# B2 Delayed Operational Checkpoint Substrate",
            "",
            "## 1. Purpose",
            "",
            "B1.1 found that flow_checkpoint_model fails delayed_checkpoint_accuracy. B2 tests whether delayed operational checkpoint requires memory, field trace, or schema memory substrate.",
            "",
            "## 2. Models",
            "",
            "- flow_checkpoint_model",
            "- recurrent_flow_checkpoint_model",
            "- field_memory_model",
            "- schema_memory_model",
            "",
            "## 3. Tasks",
            "",
            "- delayed checkpoint",
            "- multi-delay checkpoint",
            "- early saliency rejection",
            "- causal trace intervention",
            "- delay OOD",
            "",
            "## 4. Results",
            "",
            markdown_table(summary),
            "",
            "## 5. Interpretation",
            "",
            interpretation(summary),
            "",
            "## 6. Failure Analysis",
            "",
            "If all fail: delayed checkpoint remains unresolved.",
            "If only recurrent_flow succeeds: temporal memory may be sufficient, with flow-checkpoint prior still discounted.",
            "If field_memory succeeds: delayed O may be field-trace form.",
            "If schema_memory succeeds: sparse delayed schema may be strongest candidate, with injected structure prior discounted.",
            "If flow_checkpoint still fails: this confirms the B1.1 short-horizon limitation.",
            "",
            "## 7. Claim Boundary",
            "",
            "Do not claim general physical reasoning.",
            "Do not claim blank-slate emergence.",
            "Do not claim B-line solved.",
            "",
        ]
    )


def build_b2_self_audit() -> str:
    return "\n".join(
        [
            "# B2 Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Directly targets B1.1 delayed checkpoint failure.",
            "- Tests delayed operational structure rather than short-horizon checkpoint saliency.",
            "- Adds multi-delay and heldout-delay OOD.",
            "- Adds causal trace intervention.",
            "- Compares recurrent memory, field memory, and schema memory substrates.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Still a 64x64 toy world.",
            "- Delayed episodes are hand-designed.",
            "- Ground truth delays come from simulator.",
            "- Memory architectures may inject delayed structure prior.",
            "- Trace interventions may not capture distributed memory.",
            "- Passing B2 does not prove natural emergence.",
            "",
            "## False Positive Risks",
            "",
            "- Model may infer delay from generator artifacts.",
            "- Early saliency decoy may be too weak.",
            "- Heldout delays may be interpolated rather than understood.",
            "- Trace intervention may create OOD hidden states.",
            "- Schema memory may pass because sparse delayed slots are built in.",
            "",
            "## Required Failure Checks",
            "",
            "1. flow_checkpoint remains short-horizon.",
            "2. recurrent model passes behavior but fails trace intervention.",
            "3. field memory passes OOD but fails causal trace.",
            "4. schema memory passes because of hard-coded slot priority.",
            "5. all models fail heldout delays.",
            "6. delayed score high but early saliency rejection low.",
            "",
        ]
    )


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = B2_SUMMARY_KEYS
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def interpretation(summary: list[dict[str, Any]]) -> str:
    passed = [str(row["model"]) for row in summary if float(row.get("b2_delayed_score", 0.0)) > 0.0]
    if not passed:
        return "Delayed operational checkpoint remains unresolved. B2 localizes the next hard problem: delayed causal trace."
    notes = []
    if "recurrent_flow_checkpoint_model" in passed:
        notes.append("Temporal memory may be sufficient to extend checkpoint substrate from short-horizon to delayed checkpoint.")
    if "field_memory_model" in passed:
        notes.append("Delayed operational structure may be field-trace form, supporting a non-object-centric delayed O candidate.")
    if "schema_memory_model" in passed:
        notes.append("Sparse schema memory may be a stronger substrate for delayed operational checkpoints.")
    if "flow_checkpoint_model" not in passed:
        notes.append("flow_checkpoint_model remains short-horizon, confirming the B1.1 diagnosis.")
    return " ".join(notes)


if __name__ == "__main__":
    main()
