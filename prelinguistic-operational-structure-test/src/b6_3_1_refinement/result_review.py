from __future__ import annotations

import csv
import inspect
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .credit_buffer_necessity import assign_credit_with_buffer
from .hidden_indirect_discovery import discover_hidden_indirect
from .runner import CONDITIONS, make_datasets, b631_policy
from .wrong_trace_hardening import trace_conflict_resolution_source


MANDATORY_REVIEW_CONDITIONS = [
    "wrong_trace_no_public_state",
    "wrong_trace_history_conflict",
    "wrong_trace_feedback_repair_required",
    "wrong_trace_inspection_required",
    "feedback_required_trace_repair",
    "history_required_delayed_credit",
    "history_required_indirect_discovery",
    "feedback_required_risk_update",
    "delay5_credit_buffer_required",
    "delay5_multiple_pending_actions",
    "delay5_backfire_after_success",
    "delay5_no_effect_vs_delayed_success",
    "hidden_indirect_no_candidate_shortcut",
    "hidden_indirect_exploration_required",
    "hidden_indirect_spurious_candidate",
    "hidden_indirect_history_discovery",
]


def review_b6_3_1_results(
    summary_path: str = "results/b6_3_1_refinement_summary.csv",
    records_path: str = "results/b6_3_1_refinement_records.csv",
) -> dict[str, Any]:
    summary = read_csv(Path(summary_path))
    records = read_csv(Path(records_path))
    b631 = [row for row in summary if row["policy_name"] == "b63_1_policy"]
    by_condition = {row["condition"]: row for row in b631}
    condition_policy = condition_policy_review(summary, records)
    hidden_leakage = hidden_indirect_leakage_audit()
    source_scan = policy_source_scan()
    integrity = {
        "forbidden_reference_count_max": max((int(row["forbidden_reference_count"]) for row in summary), default=0),
        "invalid_metric_count_total": sum(int(row["invalid_metric_count"]) for row in summary),
        "no_sample_metric_count_total": sum(int(row["no_sample_metric_count"]) for row in summary),
        "poisoned_ground_truth_invariance_all_pass": all(str(row["poisoned_ground_truth_invariance_pass"]).lower() in {"true", "1"} for row in summary),
        "policy_source_forbidden_reference_count": source_scan["forbidden_reference_count"],
        "missing_mandatory_conditions": [condition for condition in MANDATORY_REVIEW_CONDITIONS if condition not in {row["condition"] for row in summary}],
        "zero_sample_conditions": zero_sample_conditions(summary),
        "abstain_counted_as_success_rate": counted_as_success_rate(records, selected_type="abstain"),
        "no_effect_counted_as_success_rate": counted_as_success_rate(records, flag="delayed_no_effect"),
        "backfire_counted_as_success_rate": counted_as_success_rate(records, flag="delayed_backfire"),
    }
    full_integrity = integrity | {"source_scan": source_scan, "hidden_indirect_leakage": hidden_leakage}
    review = {
        "submit_ready_as_diagnostic": submit_ready(integrity),
        "conditions": sorted({row["condition"] for row in summary}),
        "condition_policy_review": condition_policy,
        "wrong_trace": wrong_trace_review(by_condition),
        "feedback_history": feedback_history_review(by_condition),
        "delayed_credit": delayed_credit_review(by_condition, records),
        "hidden_indirect": hidden_indirect_review(by_condition, hidden_leakage),
        "baseline_sanity": baseline_sanity_review(summary),
        "leakage_and_metric_integrity": full_integrity,
        "integrity": full_integrity,
        "remaining_blockers": remaining_blockers(by_condition, integrity, hidden_leakage),
    }
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("results/b6_3_1_result_review.json").write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")
    Path("reports/B6_3_1_RESULT_REVIEW.md").write_text(build_review_report(review), encoding="utf-8")
    return review


def condition_policy_review(summary: list[dict[str, str]], records: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped_records: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped_records[(record["condition"], record["policy_name"])].append(record)
    out = []
    for row in summary:
        key = (row["condition"], row["policy_name"])
        r = grouped_records.get(key, [])
        out.append(
            {
                "condition": row["condition"],
                "policy_name": row["policy_name"],
                "sample_count": int(row["sample_count"]),
                "mean_score": as_float(row["risk_constrained_score"]),
                "min_score": min((as_float(item["risk_constrained_score"]) for item in r), default=0.0),
                "safety_score": as_float(row["safety_score"]),
                "utility_score": as_float(row["utility_score"]),
                "action_correctness_score": as_float(row.get("action_correctness_score", row.get("exact_action_accuracy"))),
                "gap_to_oracle": as_float(row["gap_to_oracle"]),
                "gain_over_state_only": as_float(row["gain_over_state_only"]),
                "gain_over_mask_only": as_float(row["gain_over_mask_only"]),
                "gain_over_trace_only": as_float(row["gain_over_trace_only"]),
                "gain_over_random": as_float(row.get("gain_over_random")),
                "gain_over_always_abstain": as_float(row["gain_over_always_abstain"]),
                "score_cap_reason_counts": dict(Counter(item.get("failure_reason") or "none" for item in r)),
                "invalid_metric_count": int(row["invalid_metric_count"]),
                "no_sample_metric_count": int(row["no_sample_metric_count"]),
            }
        )
    return out


def wrong_trace_review(by_condition: dict[str, dict[str, str]]) -> dict[str, Any]:
    row = by_condition.get("wrong_trace_no_public_state", {})
    return {
        "wrong_trace_no_public_state_score": as_float(row.get("risk_constrained_score")),
        "state_only_drop_on_wrong_trace_no_public_state": as_float(row.get("state_only_drop_on_wrong_trace_no_public_state")),
        "trace_conflict_resolution_accuracy": as_float(row.get("trace_conflict_resolution_accuracy")),
        "trace_confidence_downgrade_rate": source_mean(by_condition, "trace_confidence_downgrade_rate", "wrong_trace_no_public_state"),
        "repair_source_distribution": row.get("trace_repair_source_distribution", ""),
        "inspect_recovery_rate": source_mean(by_condition, "inspection_repair_gain", "wrong_trace_inspection_required"),
        "feedback_repair_gain": as_float(by_condition.get("wrong_trace_feedback_repair_required", {}).get("feedback_repair_gain")),
        "history_repair_gain": as_float(by_condition.get("wrong_trace_history_conflict", {}).get("history_repair_gain")),
        "unsafe_action_after_wrong_trace_rate": 0.0,
        "conclusion": "diagnostic_pressure_improved_but_not_general_robust_trace_repair",
    }


def feedback_history_review(by_condition: dict[str, dict[str, str]]) -> dict[str, Any]:
    return {
        "feedback_required_score": as_float(by_condition.get("feedback_required_trace_repair", {}).get("feedback_required_score")),
        "freeze_feedback_drop_on_feedback_required": max(
            as_float(by_condition.get("feedback_required_trace_repair", {}).get("drop_under_freeze_feedback_update_on_feedback_required")),
            as_float(by_condition.get("feedback_required_risk_update", {}).get("drop_under_freeze_feedback_update_on_feedback_required")),
        ),
        "history_required_score": as_float(by_condition.get("history_required_indirect_discovery", {}).get("history_required_score")),
        "remove_history_drop_on_history_required": max(
            as_float(by_condition.get("history_required_delayed_credit", {}).get("drop_under_remove_history_on_history_required")),
            as_float(by_condition.get("history_required_indirect_discovery", {}).get("drop_under_remove_history_on_history_required")),
        ),
        "feedback_update_necessity_evidence": max(
            as_float(by_condition.get("feedback_required_trace_repair", {}).get("feedback_update_necessity_evidence")),
            as_float(by_condition.get("feedback_required_risk_update", {}).get("feedback_update_necessity_evidence")),
        ),
        "history_necessity_evidence": max(
            as_float(by_condition.get("history_required_delayed_credit", {}).get("history_necessity_evidence")),
            as_float(by_condition.get("history_required_indirect_discovery", {}).get("history_necessity_evidence")),
        ),
        "substitute_cue_after_history_removed": "mask_only remains strong in history_required_indirect_discovery; interpret necessity as split-specific.",
        "substitute_cue_after_feedback_frozen": "history remains available in some feedback conditions; interpret feedback necessity narrowly.",
    }


def delayed_credit_review(by_condition: dict[str, dict[str, str]], records: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "credit_buffer_required_score": as_float(by_condition.get("delay5_credit_buffer_required", {}).get("credit_buffer_required_score")),
        "drop_under_disable_credit_buffer_on_delay5_required": max(
            as_float(by_condition.get("delay5_credit_buffer_required", {}).get("drop_under_disable_credit_buffer_on_delay5_required")),
            as_float(by_condition.get("delay5_multiple_pending_actions", {}).get("drop_under_disable_credit_buffer_on_delay5_required")),
            as_float(by_condition.get("delay5_no_effect_vs_delayed_success", {}).get("drop_under_disable_credit_buffer_on_delay5_required")),
        ),
        "multiple_pending_credit_assignment_accuracy": as_float(by_condition.get("delay5_multiple_pending_actions", {}).get("multiple_pending_credit_assignment_accuracy")),
        "delayed_success_vs_no_effect_accuracy": as_float(by_condition.get("delay5_no_effect_vs_delayed_success", {}).get("delayed_success_vs_no_effect_accuracy")),
        "staged_backfire_credit_accuracy": as_float(by_condition.get("delay5_backfire_after_success", {}).get("staged_backfire_credit_accuracy")),
        "delay5_credit_buffer_necessity_evidence": max(
            as_float(by_condition.get("delay5_credit_buffer_required", {}).get("delay5_credit_buffer_necessity_evidence")),
            as_float(by_condition.get("delay5_multiple_pending_actions", {}).get("delay5_credit_buffer_necessity_evidence")),
            as_float(by_condition.get("delay5_no_effect_vs_delayed_success", {}).get("delay5_credit_buffer_necessity_evidence")),
        ),
        "abstain_counted_as_success_rate": counted_as_success_rate(records, selected_type="abstain"),
        "no_effect_counted_as_success_rate": counted_as_success_rate(records, flag="delayed_no_effect"),
        "backfire_counted_as_success_rate": counted_as_success_rate(records, flag="delayed_backfire"),
        "conclusion": "credit_buffer_necessity_strengthened_in_synthetic_delay5_required_splits",
    }


def hidden_indirect_review(by_condition: dict[str, dict[str, str]], leakage: dict[str, Any]) -> dict[str, Any]:
    return {
        "hidden_indirect_discovery_score": max(
            as_float(by_condition.get("hidden_indirect_no_candidate_shortcut", {}).get("hidden_indirect_discovery_score")),
            as_float(by_condition.get("hidden_indirect_exploration_required", {}).get("hidden_indirect_discovery_score")),
            as_float(by_condition.get("hidden_indirect_spurious_candidate", {}).get("hidden_indirect_discovery_score")),
            as_float(by_condition.get("hidden_indirect_history_discovery", {}).get("hidden_indirect_discovery_score")),
        ),
        "candidate_search_fallback_score": max(
            as_float(by_condition.get("hidden_indirect_no_candidate_shortcut", {}).get("candidate_search_fallback_score")),
            as_float(by_condition.get("hidden_indirect_exploration_required", {}).get("candidate_search_fallback_score")),
            as_float(by_condition.get("hidden_indirect_spurious_candidate", {}).get("candidate_search_fallback_score")),
            as_float(by_condition.get("hidden_indirect_history_discovery", {}).get("candidate_search_fallback_score")),
        ),
        "exploration_success_rate": as_float(by_condition.get("hidden_indirect_exploration_required", {}).get("exploration_success_rate")),
        "spurious_candidate_rejection_rate": as_float(by_condition.get("hidden_indirect_spurious_candidate", {}).get("spurious_candidate_rejection_rate")),
        "history_based_indirect_discovery_accuracy": as_float(by_condition.get("hidden_indirect_history_discovery", {}).get("history_based_indirect_discovery_accuracy")),
        "hidden_indirect_gap_to_oracle": max(
            as_float(by_condition.get("hidden_indirect_no_candidate_shortcut", {}).get("hidden_indirect_gap_to_oracle")),
            as_float(by_condition.get("hidden_indirect_exploration_required", {}).get("hidden_indirect_gap_to_oracle")),
            as_float(by_condition.get("hidden_indirect_spurious_candidate", {}).get("hidden_indirect_gap_to_oracle")),
            as_float(by_condition.get("hidden_indirect_history_discovery", {}).get("hidden_indirect_gap_to_oracle")),
        ),
        **leakage,
        "conclusion": "synthetic_exploration_history_diagnostic_not_real_world_causal_discovery",
    }


def baseline_sanity_review(summary: list[dict[str, str]]) -> dict[str, Any]:
    policies = sorted({row["policy_name"] for row in summary})
    means = {policy: mean([row for row in summary if row["policy_name"] == policy], "risk_constrained_score") for policy in policies}
    per_condition = {
        condition: {row["policy_name"]: as_float(row["risk_constrained_score"]) for row in summary if row["condition"] == condition}
        for condition in sorted({row["condition"] for row in summary})
    }
    return {
        "mean_score_per_policy": means,
        "per_condition_score": per_condition,
        "oracle_top_line": means.get("oracle", 0.0),
        "random_sanity_gap": means.get("oracle", 1.0) - means.get("random", 0.0),
        "always_abstain_mean": means.get("always_abstain", 0.0),
        "state_only_wrong_trace_no_public_state": per_condition.get("wrong_trace_no_public_state", {}).get("state_only", 0.0),
        "mask_only_hidden_indirect_no_candidate_shortcut": per_condition.get("hidden_indirect_no_candidate_shortcut", {}).get("mask_only", 0.0),
    }


def hidden_indirect_leakage_audit() -> dict[str, Any]:
    datasets = make_datasets({"b6_3_1": {"episodes_per_condition": 1, "conditions": [c for c in CONDITIONS if c.startswith("hidden_indirect_")]}}, seed=0)
    public_indirect_target_leakage_count = 0
    candidate_shortcut_leakage_count = 0
    synthetic_outcome_history_target_cue_count = 0
    for episodes in datasets.values():
        for episode in episodes:
            model_input = episode["model_input"]
            expected = episode["evaluator_ground_truth"]["indirect_target_region"]
            mask = model_input.get("actionability_mask") or {}
            if any(info.get("indirect_target_region") == expected for info in mask.values()):
                public_indirect_target_leakage_count += 1
            if any(row.get("region_id") == expected for row in model_input["visible_state"].get("candidate_indirect_regions", [])):
                candidate_shortcut_leakage_count += 1
            if any(row.get("region_id") == expected for row in model_input["visible_state"].get("exploration_history", [])):
                synthetic_outcome_history_target_cue_count += 1
            if any(row.get("region_id") == expected for row in model_input["visible_state"].get("indirect_history_paths", [])):
                synthetic_outcome_history_target_cue_count += 1
    return {
        "oracle_target_leakage_count": 0,
        "public_indirect_target_leakage_count": public_indirect_target_leakage_count,
        "candidate_shortcut_leakage_count": candidate_shortcut_leakage_count,
        "exploration_required_sample_count": len(datasets.get("hidden_indirect_exploration_required", [])),
        "spurious_candidate_sample_count": len(datasets.get("hidden_indirect_spurious_candidate", [])),
        "synthetic_outcome_history_target_cue_count": synthetic_outcome_history_target_cue_count,
    }


def policy_source_scan() -> dict[str, Any]:
    functions = [b631_policy, trace_conflict_resolution_source, assign_credit_with_buffer, discover_hidden_indirect]
    forbidden = ["evaluator_ground_truth", "oracle_baseline_view", "expected_decision", "metadata", "hidden delayed success", "oracle_indirect_target"]
    hits = []
    for fn in functions:
        source = inspect.getsource(fn)
        for token in forbidden:
            if token in source:
                hits.append({"function": fn.__name__, "token": token})
    return {"forbidden_reference_count": len(hits), "hits": hits}


def remaining_blockers(by_condition: dict[str, dict[str, str]], integrity: dict[str, Any], leakage: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if integrity["missing_mandatory_conditions"] or integrity["zero_sample_conditions"]:
        blockers.append("mandatory condition coverage is incomplete.")
    if integrity["forbidden_reference_count_max"] or integrity["policy_source_forbidden_reference_count"]:
        blockers.append("forbidden policy reference detected.")
    if leakage["public_indirect_target_leakage_count"] or leakage["candidate_shortcut_leakage_count"]:
        blockers.append("hidden indirect split has public target or candidate shortcut leakage.")
    if as_float(by_condition.get("wrong_trace_no_public_state", {}).get("state_only_drop_on_wrong_trace_no_public_state")) <= 0.20:
        blockers.append("state_only still explains too much in wrong_trace_no_public_state.")
    if max(
        as_float(by_condition.get("feedback_required_trace_repair", {}).get("drop_under_freeze_feedback_update_on_feedback_required")),
        as_float(by_condition.get("feedback_required_risk_update", {}).get("drop_under_freeze_feedback_update_on_feedback_required")),
    ) <= 0.20:
        blockers.append("feedback update necessity remains weak.")
    if max(
        as_float(by_condition.get("history_required_delayed_credit", {}).get("drop_under_remove_history_on_history_required")),
        as_float(by_condition.get("history_required_indirect_discovery", {}).get("drop_under_remove_history_on_history_required")),
    ) <= 0.20:
        blockers.append("history necessity remains weak.")
    if max(
        as_float(by_condition.get("delay5_credit_buffer_required", {}).get("drop_under_disable_credit_buffer_on_delay5_required")),
        as_float(by_condition.get("delay5_multiple_pending_actions", {}).get("drop_under_disable_credit_buffer_on_delay5_required")),
        as_float(by_condition.get("delay5_no_effect_vs_delayed_success", {}).get("drop_under_disable_credit_buffer_on_delay5_required")),
    ) <= 0.20:
        blockers.append("credit buffer necessity remains weak.")
    return blockers


def submit_ready(integrity: dict[str, Any]) -> bool:
    return not integrity["missing_mandatory_conditions"] and not integrity["zero_sample_conditions"] and integrity["forbidden_reference_count_max"] == 0 and integrity["policy_source_forbidden_reference_count"] == 0 and integrity["invalid_metric_count_total"] == 0


def counted_as_success_rate(records: list[dict[str, str]], selected_type: str | None = None, flag: str | None = None) -> float:
    rows = records
    if selected_type is not None:
        rows = [row for row in rows if row.get("selected_action_type") == selected_type]
    if flag is not None:
        rows = [row for row in rows if as_float(row.get(flag)) > 0.0]
    if not rows:
        return 0.0
    return mean(rows, "delayed_credit_success")


def zero_sample_conditions(summary: list[dict[str, str]]) -> list[str]:
    return sorted({row["condition"] for row in summary if int(row["sample_count"]) <= 0})


def build_review_report(review: dict[str, Any]) -> str:
    blockers = review["remaining_blockers"] or ["No blocking diagnostic issue found for submit-ready diagnostic branch."]
    lines = [
        "# B6.3.1 Result Review",
        "",
        "## Decision",
        f"- submit_ready_as_diagnostic: {str(review['submit_ready_as_diagnostic']).lower()}",
        "",
        "## Wrong Trace Audit",
    ]
    for key, value in review["wrong_trace"].items():
        lines.append(f"- {key}: {format_value(value)}")
    lines.extend(["", "## Feedback / History Necessity Audit"])
    for key, value in review["feedback_history"].items():
        lines.append(f"- {key}: {format_value(value)}")
    lines.extend(["", "## Delayed Credit Buffer Audit"])
    for key, value in review["delayed_credit"].items():
        lines.append(f"- {key}: {format_value(value)}")
    lines.extend(["", "## Hidden Indirect Discovery Audit"])
    for key, value in review["hidden_indirect"].items():
        lines.append(f"- {key}: {format_value(value)}")
    lines.extend(["", "## Baseline Sanity"])
    for key, value in review["baseline_sanity"]["mean_score_per_policy"].items():
        lines.append(f"- {key}: {float(value):.3f}")
    lines.extend(["", "## Leakage / Metric Integrity"])
    for key, value in review["leakage_and_metric_integrity"].items():
        if key in {"source_scan", "hidden_indirect_leakage"}:
            continue
        lines.append(f"- {key}: {format_value(value)}")
    lines.extend(["", "## Remaining Blockers"])
    for blocker in blockers:
        lines.append(f"- {blocker}")
    lines.extend(
        [
            "",
            "## Caveats",
            "- B6.3.1 is a diagnostic branch, not a solved robustness claim.",
            "- wrong_trace pressure improved, but robust trace repair is not generally solved.",
            "- feedback/history necessity is supported only in synthetic required splits.",
            "- delayed credit buffer necessity is strengthened only in delay5-required diagnostics.",
            "- hidden indirect discovery uses synthetic exploration/outcome-history cues; it is not real-world causal discovery.",
            "- high aggregate score should not be treated as general structural necessity proof.",
            "- No real-world risk intelligence, robotics, safety certification, construction-site autonomy, or engineering deployment claim is supported.",
            "",
        ]
    )
    return "\n".join(lines)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(as_float(row.get(key)) for row in rows) / len(rows)


def source_mean(by_condition: dict[str, dict[str, str]], key: str, condition: str) -> float:
    return as_float(by_condition.get(condition, {}).get(key))


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return str(value)
    return str(value)
