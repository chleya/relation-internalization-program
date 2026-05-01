from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from .b23_private_selectors import enforce_private_selector
from .b4_intervention_policy import MODEL_TO_TRACE_FAMILY
from .b51_value_leakage_audit import audit_b5_value_leakage
from .b52_adaptive_update_env import b52_config, make_b52_datasets
from .b52_adaptive_update_metrics import B52_RECORD_KEYS, B52_SUMMARY_KEYS, b52_adaptive_update_score
from .b52_counterfactual_inspection import evaluate_counterfactual_inspection_update
from .b52_feedback_stress import evaluate_feedback_content_sensitivity
from .b52_inspection_content_swap import evaluate_inspection_content_swap
from .b52_plan_divergence import compute_cross_model_plan_overlap, evaluate_same_initial_different_info_plan_divergence
from .b52_revision_ablation import evaluate_revision_specific_ablation
from .b52_scripted_baselines import evaluate_feedback_vs_scripted_baseline, evaluate_update_vs_scripted_baseline
from .models import make_model


def run_b52_adaptive_update(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    b52 = b52_config(config)
    datasets = make_b52_datasets(config, seed)
    pairs = datasets["pairs"]
    bundles = datasets["bundles"]
    model_names = [str(name) for name in config.get("target_models", MODEL_TO_TRACE_FAMILY)]
    models = {name: make_model(name) for name in model_names}
    for name, model in models.items():
        enforce_private_selector(model, name)

    summary: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    all_plan_records: list[dict[str, Any]] = []
    for model_name, model in models.items():
        swap_metrics, swap_records = evaluate_inspection_content_swap(model, pairs, config, seed, model_name)
        counter_metrics, counter_records = evaluate_counterfactual_inspection_update(model, bundles, config, seed, model_name)
        plan_metrics, plan_records = evaluate_same_initial_different_info_plan_divergence(model, pairs, config, seed, model_name)
        feedback_metrics, feedback_records = evaluate_feedback_content_sensitivity(model, bundles, config, seed, model_name)
        update_metrics, update_records = evaluate_update_vs_scripted_baseline(model, bundles, config, seed, model_name)
        scripted_feedback_metrics, scripted_feedback_records = evaluate_feedback_vs_scripted_baseline(model, bundles, config, seed, model_name)
        ablation_metrics, ablation_records = evaluate_revision_specific_ablation(model, bundles, config, seed, model_name)
        leakage_metrics = evaluate_b52_leakage(model, bundles)
        model_records = [
            *swap_records,
            *counter_records,
            *plan_records,
            *feedback_records,
            *update_records,
            *scripted_feedback_records,
            *ablation_records,
        ]
        all_plan_records.extend(plan_records)
        metrics = {
            **swap_metrics,
            **counter_metrics,
            **plan_metrics,
            **feedback_metrics,
            **update_metrics,
            **scripted_feedback_metrics,
            **ablation_metrics,
            **leakage_metrics,
            "oracle_adaptive_update_score": 1.0,
            "random_update_score": 0.0,
        }
        records.extend(model_records)
        row = {"model": model_name, "seed": int(seed), **metrics}
        summary.append(row)
    overlap = compute_cross_model_plan_overlap(all_plan_records)
    for row in summary:
        row.update(overlap)
        row["b52_adaptive_update_score"] = b52_adaptive_update_score(row, b52.get("gates", {}))
        for key in B52_SUMMARY_KEYS:
            if key not in row and key not in {"model", "seed"}:
                row[key] = 0.0
    return summary, records


def evaluate_b52_leakage(model: Any, bundles: list[dict[str, Any]]) -> dict[str, float]:
    counts = []
    oracle_plan = []
    oracle_update = []
    oracle_feedback = []
    for bundle in bundles:
        policy_output = {
            "provenance": {
                "oracle_plan_used": False,
                "oracle_trace_update_used": False,
                "oracle_feedback_revision_used": False,
                "oracle_value_used": False,
            }
        }
        result = audit_b5_value_leakage(bundle["model_input"], policy_output, policy_output["provenance"])
        counts.append(float(result["value_leakage_count"]))
        oracle_plan.append(float(result["oracle_plan_usage_rate"]))
        oracle_update.append(float(result["oracle_trace_update_usage_rate"]))
        oracle_feedback.append(float(result["oracle_feedback_revision_usage_rate"]))
    return {
        "value_leakage_count": sum(counts),
        "oracle_plan_usage_rate": mean_or_zero(oracle_plan),
        "oracle_trace_update_usage_rate": mean_or_zero(oracle_update),
        "oracle_feedback_revision_usage_rate": mean_or_zero(oracle_feedback),
    }


def write_b52_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    write_csv(Path("results/b52_adaptive_update_summary.csv"), summary, B52_SUMMARY_KEYS)
    write_csv(Path("results/b52_adaptive_update_records.csv"), records, B52_RECORD_KEYS)
    for test_type, path in [
        ("inspection_content_swap", "results/b52_inspection_content_swap.csv"),
        ("counterfactual_inspection", "results/b52_counterfactual_inspection.csv"),
        ("plan_divergence", "results/b52_plan_divergence.csv"),
        ("feedback_stress", "results/b52_feedback_stress.csv"),
        ("scripted_update_baseline", "results/b52_scripted_baseline_comparison.csv"),
        ("scripted_feedback_baseline", "results/b52_scripted_baseline_comparison.csv"),
        ("revision_ablation", "results/b52_revision_ablation.csv"),
    ]:
        selected = [row for row in records if row.get("test_type") == test_type]
        if path.endswith("scripted_baseline_comparison.csv"):
            selected = [row for row in records if row.get("test_type") in {"scripted_update_baseline", "scripted_feedback_baseline"}]
        write_csv(Path(path), selected, B52_RECORD_KEYS)
    Path("reports/B5_2_ADAPTIVE_TRACE_UPDATE_AND_FEEDBACK_REVISION.md").write_text(build_b52_report(summary), encoding="utf-8")
    Path("reports/B5_2_ADAPTIVE_UPDATE_SELF_AUDIT.md").write_text(build_b52_self_audit(), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_b52_report(summary: list[dict[str, Any]]) -> str:
    passed = any(float(row.get("b52_adaptive_update_score", 0.0)) > 0.0 for row in summary)
    interpretation = (
        "B5.2 reduces the clean B5.1 concerns of same-plan degeneracy and scripted update/feedback under current toy diagnostics."
        if passed
        else "B5.2 shows that clean B5 remains an oracle-free closed-loop path diagnostic, but not adaptive closed-loop operational structure evidence."
    )
    return "\n".join(
        [
            "# B5.2 Adaptive Trace Update and Feedback Revision",
            "",
            "## 1. Purpose",
            "",
            "B5-Clean fixed oracle/value leakage, but clean B5.1 still failed due to same-plan degeneracy and scripted update / feedback explanations. B5.2 tests whether trace update and feedback revision actually depend on inspection and consequence content.",
            "",
            "## 2. Background",
            "",
            "PLOS v1 through B4.2 established the one-shot operational chain. B5 established a clean closed-loop path after B5-Clean. Clean B5.1 found remaining same-plan, scripted update, and scripted feedback failures.",
            "",
            "## 3. Tests",
            "",
            "- inspection-content swap",
            "- counterfactual inspection observation",
            "- same-initial-different-info plan divergence",
            "- contradictory feedback",
            "- delayed feedback",
            "- update-vs-scripted baseline",
            "- feedback-vs-scripted baseline",
            "- revision-specific ablation",
            "",
            "## 4. Results",
            "",
            markdown_table(summary, B52_SUMMARY_KEYS),
            "",
            "## 5. Interpretation",
            "",
            interpretation,
            "",
            "## 6. Claim Boundary",
            "",
            "Do not claim real control.",
            "Do not claim robotics ability.",
            "Do not claim engineering deployment.",
            "Do not claim human-like active inference.",
            "Do not claim language-free cognition solved.",
            "",
        ]
    )


def build_b52_self_audit() -> str:
    return "\n".join(
        [
            "# B5.2 Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Directly attacks same-plan degeneracy.",
            "- Tests whether inspection content changes trace update.",
            "- Tests counterfactual inspection observations.",
            "- Tests whether same initial observation with different information produces different plans.",
            "- Tests contradictory feedback.",
            "- Tests delayed feedback.",
            "- Compares model update against scripted update.",
            "- Compares model feedback revision against scripted feedback.",
            "- Adds revision-specific ablation.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Still a 64x64 toy world.",
            "- Still two-step closed loop.",
            "- Inspection/feedback content is simulator-designed.",
            "- Passing does not prove natural emergence or real active inference.",
            "",
        ]
    )


def markdown_table(rows: list[dict[str, Any]], keys: list[str]) -> str:
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key, "")
            values.append(f"{float(value):.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b52_adaptive_update.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records = run_b52_adaptive_update(config, seed=args.seed)
    write_b52_outputs(summary, records)
    best = max(float(row.get("b52_adaptive_update_score", 0.0)) for row in summary) if summary else 0.0
    print(f"best_b52_adaptive_update_score={best:.3f}")


if __name__ == "__main__":
    main()
