from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .g1_1_env import CONDITIONS, make_g1_1_datasets
from .g1_1_generator import choose_pressure_action, fit_pressure_generator, pressure_ablation_artifacts
from .g1_baselines import BASELINE_NAMES, baseline_action
from .g1_metrics import RECORD_FIELDS, SUMMARY_FIELDS, score_episode, summarize


POLICIES = ["g1_1_generator", "g1_1_no_feedback", "g1_1_no_compression", *BASELINE_NAMES[1:]]


def run_g1_1(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    datasets = make_g1_1_datasets(config, seed)
    train_inputs = [episode["model_input"] for episodes in datasets.values() for episode in episodes]
    artifact = fit_pressure_generator(train_inputs, config)
    ablations = pressure_ablation_artifacts(artifact)
    records: list[dict[str, Any]] = []
    raw: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for condition, episodes in datasets.items():
        for policy_name in POLICIES:
            rows = []
            for episode in episodes:
                if policy_name == "g1_1_generator":
                    output = choose_pressure_action(episode["model_input"], artifact)
                elif policy_name == "g1_1_no_feedback":
                    output = choose_pressure_action(episode["model_input"], ablations["no_feedback"])
                elif policy_name == "g1_1_no_compression":
                    output = choose_pressure_action(episode["model_input"], ablations["no_compression"])
                else:
                    output = baseline_action(episode["model_input"], policy_name, seed=int(episode["episode_id"]), truth=episode["evaluator_ground_truth"])
                row = score_episode(episode, output, policy_name)
                rows.append(row)
                records.append(row)
            raw[(condition, policy_name)] = rows
    summary = []
    for condition in CONDITIONS:
        baseline_scores = {policy_name: mean(raw.get((condition, policy_name), []), "generator_score") for policy_name in POLICIES}
        for policy_name in POLICIES:
            summary.append(summarize(raw.get((condition, policy_name), []), baseline_scores))
    metrics = build_metrics(summary, records, artifact)
    return summary, records, metrics, artifact


def build_metrics(summary: list[dict[str, Any]], records: list[dict[str, Any]], artifact: dict[str, Any]) -> dict[str, Any]:
    rows = {row["condition"]: row for row in summary if row["policy_name"] == "g1_1_generator"}
    no_feedback = {row["condition"]: row for row in summary if row["policy_name"] == "g1_1_no_feedback"}
    no_compression = {row["condition"]: row for row in summary if row["policy_name"] == "g1_1_no_compression"}
    feedback_drop = avg([
        float(rows[c]["generator_score"]) - float(no_feedback[c]["generator_score"])
        for c in rows
    ])
    compression_drop = avg([
        float(rows[c]["generator_score"]) - float(no_compression[c]["generator_score"])
        for c in rows
    ])
    return {
        "summary_rows": len(summary),
        "record_rows": len(records),
        "conditions": sorted({row["condition"] for row in summary}),
        "g1_1_mean_score": mean(list(rows.values()), "generator_score"),
        "g1_1_ood_pressure_score": float(rows.get("ood_pressure_remap", {}).get("generator_score", 0.0)),
        "feedback_pressure_gain": feedback_drop,
        "compression_pressure_gain": compression_drop,
        "generated_rule_pressure_coverage": int(artifact["rule"].use_feedback) + int(artifact["rule"].use_compression),
        "g1_1_oracle_gap": mean(list(rows.values()), "oracle_gap"),
        "g1_1_mask_f1": mean(list(rows.values()), "mask_f1"),
        "invalid_metric_count_total": sum(int(row["invalid_metric_count"]) for row in summary),
        "generator_rule": artifact["rule"].__dict__,
        "submit_ready_as_generator_diagnostic": True,
    }


def write_g1_1_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    out_dir = Path("results/g1_1_pressure_hardening")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "summary.csv", summary, SUMMARY_FIELDS)
    write_csv(out_dir / "records.csv", records, RECORD_FIELDS)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/G1_1_PRESSURE_USE_HARDENING_RESULTS.md").write_text(build_report(metrics), encoding="utf-8")


def build_report(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# G1.1 Pressure-Use Hardening Results",
            "",
            "## Summary",
            "G1.1 creates targeted pressure splits to test whether feedback and compression channels become useful for generated operational structure.",
            "",
            "## Metrics",
            f"- g1_1_mean_score = {float(metrics['g1_1_mean_score']):.3f}",
            f"- g1_1_ood_pressure_score = {float(metrics['g1_1_ood_pressure_score']):.3f}",
            f"- feedback_pressure_gain = {float(metrics['feedback_pressure_gain']):.3f}",
            f"- compression_pressure_gain = {float(metrics['compression_pressure_gain']):.3f}",
            f"- generated_rule_pressure_coverage = {int(metrics['generated_rule_pressure_coverage'])}",
            f"- g1_1_oracle_gap = {float(metrics['g1_1_oracle_gap']):.3f}",
            f"- g1_1_mask_f1 = {float(metrics['g1_1_mask_f1']):.3f}",
            "",
            "## Interpretation",
            "G1.1 is useful if pressure ablations reduce performance and the selected rule uses more pressure channels than G1.",
            "",
            "## Claim Boundary",
            "G1.1 remains a toy generator diagnostic. It does not prove autonomous cognition, real-world risk intelligence, robotics capability, safety certification, construction-site autonomy, or deployable engineering control.",
            "",
        ]
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
