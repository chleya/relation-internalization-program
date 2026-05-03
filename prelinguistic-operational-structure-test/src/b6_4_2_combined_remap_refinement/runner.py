from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .combined_metrics import RECORD_FIELDS, SUMMARY_FIELDS, score_output, summarize_policy
from .combined_policy import POLICY_NAMES, audit_policy_integrity, policy_for
from .combined_remap_env import CONDITIONS, make_b642_episode
from .failure_attribution import attribute_failures


def make_datasets(config: dict[str, Any], seed: int) -> dict[str, list[dict[str, Any]]]:
    section = config.get("b6_4_2", {})
    n = int(section.get("episodes_per_condition", 4))
    conditions = list(section.get("conditions", CONDITIONS))
    return {
        condition: [make_b642_episode(config, seed + cidx * 1000 + idx, condition) for idx in range(n)]
        for cidx, condition in enumerate(conditions)
    }


def run_b6_4_2_combined_refinement(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    datasets = make_datasets(config, seed)
    records: list[dict[str, Any]] = []
    raw: dict[tuple[str, str], list[dict[str, Any]]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for condition, episodes in datasets.items():
        audits[condition] = audit_policy_integrity(episodes[0], config) if episodes else {
            "forbidden_reference_count": 0,
            "poisoned_evaluator_invariance_pass": False,
            "policy_uses_model_input_only": False,
        }
        for policy_name in POLICY_NAMES:
            policy = policy_for(policy_name)
            rows = []
            for episode in episodes:
                out = policy(episode, config)
                scored = score_output(episode, out)
                row = {"condition": condition, "seed": seed, "policy_name": policy_name, **scored}
                rows.append(row)
                records.append(row)
            raw[(condition, policy_name)] = rows
    summary: list[dict[str, Any]] = []
    for condition in datasets:
        baselines = {policy_name: mean(raw.get((condition, policy_name), []), "combined_refinement_score") for policy_name in POLICY_NAMES}
        for policy_name in POLICY_NAMES:
            summary.append(
                summarize_policy(
                    raw.get((condition, policy_name), []),
                    {"condition": condition, "seed": seed, "policy_name": policy_name},
                    baselines,
                    audits[condition],
                )
            )
    metrics = build_metrics(summary, records)
    return summary, records, metrics


def build_metrics(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    policy_rows = [row for row in summary if row["policy_name"] == "b64_2_combined_policy"]
    attribution = attribute_failures(summary, records)
    pair_rows = [row for row in policy_rows if row["condition"].startswith("pair_")]
    triple_rows = [row for row in policy_rows if row["condition"].startswith("triple_")]
    combined = next((row for row in policy_rows if row["condition"] == "combined_remap_hard_reference"), {})
    ablation = {row["condition"]: row for row in policy_rows if row["condition"].startswith("combined_disable_")}
    baseline_gap = avg([float(row["combined_baseline_gap"]) for row in policy_rows])
    oracle_gap = avg([float(row["combined_oracle_gap"]) for row in policy_rows])
    pair_drop = avg([float(row["pairwise_transfer_drop"]) for row in pair_rows])
    triple_drop = avg([float(row["triple_transfer_drop"]) for row in triple_rows])
    confidence = float(attribution["failure_attribution_confidence"])
    integrity = 1.0 if max((int(row["forbidden_reference_count"]) for row in summary), default=0) == 0 and sum(int(row["invalid_metric_count"]) for row in summary) == 0 else 0.0
    score = 0.30 * max(0.0, baseline_gap) + 0.25 * confidence + 0.20 * (1.0 - min(1.0, oracle_gap)) + 0.15 * (1.0 - min(1.0, triple_drop)) + 0.10 * integrity
    combined_gap = float(combined.get("combined_oracle_gap", 0.0))
    gate = "STAY_IN_B6_REFINEMENT" if combined_gap > 0.15 else "PROCEED_TO_B7_ALLOWED"
    return {
        "summary_rows": len(summary),
        "record_rows": len(records),
        "conditions": sorted({row["condition"] for row in summary}),
        "combined_refinement_score": score,
        "combined_recovery_score": float(combined.get("combined_recovery_score", 0.0)),
        "pairwise_transfer_drop": pair_drop,
        "triple_transfer_drop": triple_drop,
        "mechanism_specific_oracle_gap": attribution["ablation_oracle_gaps"],
        "failure_attribution_confidence": confidence,
        "combined_failure_source": attribution["combined_failure_source"],
        "combined_oracle_gap": combined_gap,
        "combined_policy_score": float(combined.get("combined_policy_score", 0.0)),
        "combined_oracle_score": float(combined.get("combined_oracle_score", 0.0)),
        "combined_baseline_gap": float(combined.get("combined_baseline_gap", 0.0)),
        "pair_visual_risk_score": score_for(policy_rows, "pair_visual_risk"),
        "pair_visual_dynamics_score": score_for(policy_rows, "pair_visual_dynamics"),
        "pair_risk_mask_score": score_for(policy_rows, "pair_risk_mask"),
        "pair_dynamics_delay_score": score_for(policy_rows, "pair_dynamics_delay"),
        "pair_mask_indirect_score": score_for(policy_rows, "pair_mask_indirect"),
        "pair_delay_indirect_score": score_for(policy_rows, "pair_delay_indirect"),
        "triple_visual_risk_dynamics_score": score_for(policy_rows, "triple_visual_risk_dynamics"),
        "triple_risk_mask_delay_score": score_for(policy_rows, "triple_risk_mask_delay"),
        "triple_dynamics_delay_indirect_score": score_for(policy_rows, "triple_dynamics_delay_indirect"),
        "triple_visual_mask_indirect_score": score_for(policy_rows, "triple_visual_mask_indirect"),
        "triple_visual_risk_mask_score": score_for(policy_rows, "triple_visual_risk_mask"),
        "drop_under_combined_disable_trace_repair": float(ablation.get("combined_disable_trace_repair", {}).get("drop_under_combined_disable_trace_repair", 0.0)),
        "drop_under_combined_disable_fallback_risk": float(ablation.get("combined_disable_fallback_risk", {}).get("drop_under_combined_disable_fallback_risk", 0.0)),
        "drop_under_combined_disable_delayed_credit": float(ablation.get("combined_disable_delayed_credit", {}).get("drop_under_combined_disable_delayed_credit", 0.0)),
        "drop_under_combined_disable_candidate_search": float(ablation.get("combined_disable_candidate_search", {}).get("drop_under_combined_disable_candidate_search", 0.0)),
        "drop_under_combined_disable_inspection_recovery": float(ablation.get("combined_disable_inspection_recovery", {}).get("drop_under_combined_disable_inspection_recovery", 0.0)),
        "forbidden_reference_count_max": max((int(row["forbidden_reference_count"]) for row in summary), default=0),
        "invalid_metric_count_total": sum(int(row["invalid_metric_count"]) for row in summary),
        "no_sample_metric_count_total": sum(int(row["no_sample_metric_count"]) for row in summary),
        "combined_leakage_count_total": sum(int(row["combined_leakage_count"]) for row in summary),
        "shortcut_leakage_count_total": sum(int(row["shortcut_leakage_count"]) for row in summary),
        "poisoned_evaluator_invariance_all_pass": all(str(row["poisoned_evaluator_invariance_pass"]).lower() in {"true", "1"} for row in summary),
        "submit_ready_as_diagnostic": True,
        "gate_judgment": gate,
        "gate_reason": "combined_remap_hard oracle gap remains above threshold" if gate == "STAY_IN_B6_REFINEMENT" else "combined remap gate eligible for review",
    }


def write_b6_4_2_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    write_csv(Path("results/b6_4_2_combined_refinement_summary.csv"), summary, SUMMARY_FIELDS)
    write_csv(Path("results/b6_4_2_combined_refinement_records.csv"), records, RECORD_FIELDS)
    Path("results/b6_4_2_combined_refinement_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    Path("reports/B6_4_2_COMBINED_REMAP_REFINEMENT_REPORT.md").write_text(build_report(metrics), encoding="utf-8")


def build_report(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# B6.4.2 Combined Remap Refinement",
            "",
            "## Purpose",
            "B6.4.2 diagnoses the B6.4.1 combined_remap_hard oracle gap through pairwise, triple, and mechanism-ablation combined remaps.",
            "",
            "## Starting Blocker",
            "B6.4.1 combined_remap_hard had policy=0.675 and oracle_gap=0.325.",
            "",
            "## Results",
            f"- combined_refinement_score = {float(metrics['combined_refinement_score']):.3f}",
            f"- combined_policy_score = {float(metrics['combined_policy_score']):.3f}",
            f"- combined_oracle_gap = {float(metrics['combined_oracle_gap']):.3f}",
            f"- combined_failure_source = {metrics['combined_failure_source']}",
            f"- failure_attribution_confidence = {float(metrics['failure_attribution_confidence']):.3f}",
            "",
            "## Claim Boundary",
            "B6.4.2 supports only toy-to-toy combined-remap diagnostic evidence. It does not support real-world transfer, robotics, construction-site autonomy, safety certification, or deployable control.",
            "",
        ]
    )


def score_for(rows: list[dict[str, Any]], condition: str) -> float:
    row = next((item for item in rows if item["condition"] == condition), None)
    return float(row.get("combined_refinement_score", 0.0)) if row else 0.0


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
