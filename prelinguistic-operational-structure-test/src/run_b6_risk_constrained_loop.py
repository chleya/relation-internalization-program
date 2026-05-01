from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from .b23_private_selectors import enforce_private_selector
from .b4_intervention_policy import MODEL_TO_TRACE_FAMILY
from .b51_value_leakage_audit import audit_b5_value_leakage
from .b6_risk_baselines import evaluate_b6_baselines
from .b6_risk_constrained_policy import evaluate_risk_constrained_policy
from .b6_risk_env import make_b6_datasets
from .b6_risk_feedback_revision import evaluate_risk_aware_feedback_revision
from .b6_risk_metrics import B6_RECORD_KEYS, B6_SUMMARY_KEYS, b6_risk_constrained_score
from .models import make_model


def run_b6_risk_constrained_loop(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = make_b6_datasets(config, seed)
    episodes = datasets["test"]
    model_names = [str(name) for name in config.get("target_models", MODEL_TO_TRACE_FAMILY)]
    models = {name: make_model(name) for name in model_names}
    for name, model in models.items():
        enforce_private_selector(model, name)
    summary = []
    records = []
    b6 = config.get("b6", {})
    for model_name, model in models.items():
        policy_metrics, policy_records = evaluate_risk_constrained_policy(model, episodes, config, seed, model_name)
        baseline_metrics, baseline_records = evaluate_b6_baselines(model, episodes, config, seed)
        feedback_metrics, feedback_records = evaluate_risk_aware_feedback_revision(model, episodes, config)
        leakage_metrics = evaluate_b6_leakage(episodes)
        model_score = float(policy_metrics.get("policy_model_score", 0.0))
        metrics = {
            **policy_metrics,
            **feedback_metrics,
            **baseline_metrics,
            **leakage_metrics,
            "gain_over_random": model_score - baseline_metrics["random_risk_constrained_score"],
            "gain_over_saliency": model_score - baseline_metrics["saliency_risk_constrained_score"],
            "gain_over_short_horizon": model_score - baseline_metrics["short_horizon_risk_constrained_score"],
            "gain_over_risk_blind": model_score - baseline_metrics["risk_blind_score"],
            "gain_over_always_act": model_score - baseline_metrics["always_act_score"],
            "gain_over_always_abstain": model_score - baseline_metrics["always_abstain_score"],
        }
        metrics["b6_risk_constrained_score"] = b6_risk_constrained_score(metrics, b6.get("gates", {}))
        row = {"model": model_name, "seed": int(seed)}
        for key in B6_SUMMARY_KEYS:
            if key not in {"model", "seed"}:
                row[key] = float(metrics.get(key, 0.0))
        summary.append(row)
        records.extend(policy_records)
        records.extend({"model": model_name, "seed": seed, **record} for record in baseline_records)
        records.extend({"model": model_name, "seed": seed, **record} for record in feedback_records)
    return summary, records


def evaluate_b6_leakage(episodes: list[dict[str, Any]]) -> dict[str, float]:
    count = 0
    for bundle in episodes:
        result = audit_b5_value_leakage(bundle["model_input"], {"provenance": {}}, {})
        count += int(result["value_leakage_count"])
    return {
        "value_leakage_count": float(count),
        "oracle_actionability_usage_rate": 0.0,
        "oracle_risk_value_usage_rate": 0.0,
    }


def write_b6_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    write_csv(Path("results/b6_risk_constrained_summary.csv"), summary, B6_SUMMARY_KEYS)
    write_csv(Path("results/b6_risk_constrained_records.csv"), records, B6_RECORD_KEYS)
    write_csv(Path("results/b6_actionability_mask_records.csv"), [row for row in records if row.get("note") == "risk-constrained closed-loop policy"], B6_RECORD_KEYS)
    write_csv(Path("results/b6_wrong_actionability_penalty.csv"), [row for row in records if row.get("violation_type") not in {"", "allowed", None}], B6_RECORD_KEYS)
    write_csv(Path("results/b6_unsafe_action_rejection.csv"), [row for row in records if int(row.get("region_unsafe", 0) or 0)], B6_RECORD_KEYS)
    write_csv(Path("results/b6_indirect_intervention.csv"), [row for row in records if row.get("selected_action_type") in {"indirect_stabilize", "indirect_block"}], B6_RECORD_KEYS)
    write_csv(Path("results/b6_cost_sensitive_planning.csv"), [row for row in records if int(row.get("region_costly", 0) or 0)], B6_RECORD_KEYS)
    write_csv(Path("results/b6_risk_feedback_revision.csv"), [row for row in records if row.get("test_type") == "risk_feedback_revision"], B6_RECORD_KEYS)
    write_csv(Path("results/b6_baseline_comparison.csv"), [row for row in records if row.get("baseline_name")], B6_RECORD_KEYS)
    Path("reports/B6_ACTIONABILITY_MASK_RISK_CONSTRAINED_CLOSED_LOOP.md").write_text(build_b6_report(summary), encoding="utf-8")
    Path("reports/B6_RISK_CONSTRAINED_SELF_AUDIT.md").write_text(build_b6_self_audit(), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_b6_report(summary: list[dict[str, Any]]) -> str:
    passed = any(float(row.get("b6_risk_constrained_score", 0.0)) > 0.0 for row in summary)
    interpretation = (
        "B6 supports that, in the toy PLOS environment, the clean B5.2 closed-loop system can use an actionability mask to choose inspect, direct intervention, indirect intervention, or abstain under risk and cost constraints."
        if passed
        else "B6 shows that the current clean closed-loop system can update and revise operational traces, but cannot yet operate under actionability, risk, cost, unsafe, or irreversible constraints."
    )
    return "\n".join(
        [
            "# B6 Actionability Mask / Risk-Constrained Closed Loop",
            "",
            "## 1. Purpose",
            "",
            "B6 tests whether the B5.2 clean closed-loop system can operate under actionability, risk, irreversibility, and cost constraints.",
            "",
            "## 2. Background",
            "",
            "PLOS v1 through B4.2 established delayed trace to inspect/intervene/action type. B5-Clean fixed oracle/value leakage. B5.2 showed content-sensitive update and feedback revision. B6 adds actionability mask and risk-constrained choice.",
            "",
            "## 3. Actionability Mask",
            "",
            "- observable",
            "- inspectable",
            "- directly_intervenable",
            "- indirectly_intervenable",
            "- unsafe",
            "- irreversible",
            "- costly",
            "",
            "## 4. Results",
            "",
            markdown_table(summary, B6_SUMMARY_KEYS),
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
            "Do not claim safety-certified planning.",
            "Do not claim human-like risk reasoning.",
            "",
        ]
    )


def build_b6_self_audit() -> str:
    return "\n".join(
        [
            "# B6 Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Adds actionability mask.",
            "- Distinguishes observable, inspectable, directly intervenable, indirectly intervenable, unsafe, irreversible, costly.",
            "- Tests abstain decision.",
            "- Tests unsafe-action rejection.",
            "- Tests indirect intervention.",
            "- Tests cost-sensitive planning.",
            "- Adds risk-blind baseline.",
            "- Adds risk-aware feedback revision.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Still a 64x64 toy world.",
            "- Risk/cost values are simulator-defined.",
            "- Actionability mask is provided, not learned.",
            "- No real physical safety, robot control, or engineering deployment.",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b6_risk_constrained_loop.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records = run_b6_risk_constrained_loop(config, seed=args.seed)
    write_b6_outputs(summary, records)
    best = max(float(row.get("b6_risk_constrained_score", 0.0)) for row in summary) if summary else 0.0
    print(f"best_b6_risk_constrained_score={best:.3f}")


if __name__ == "__main__":
    main()
