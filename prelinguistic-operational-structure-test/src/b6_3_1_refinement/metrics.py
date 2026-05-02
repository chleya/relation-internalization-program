from __future__ import annotations

from collections import Counter
from typing import Any


SUMMARY_FIELDS = [
    "condition",
    "seed",
    "policy_name",
    "sample_count",
    "no_sample_metric_count",
    "invalid_metric_count",
    "risk_constrained_score",
    "safety_score",
    "utility_score",
    "exact_action_accuracy",
    "action_correctness_score",
    "wrong_trace_no_public_state_score",
    "wrong_trace_history_conflict_score",
    "wrong_trace_feedback_repair_score",
    "wrong_trace_inspection_recovery_score",
    "state_only_drop_on_wrong_trace_no_public_state",
    "feedback_repair_gain",
    "history_repair_gain",
    "inspection_repair_gain",
    "trace_conflict_resolution_accuracy",
    "trace_repair_source_distribution",
    "public_state_dependency_score",
    "substitute_history_dependency_score",
    "substitute_feedback_dependency_score",
    "public_state_hidden_drop",
    "public_state_plus_history_hidden_drop",
    "public_state_plus_feedback_hidden_drop",
    "public_state_plus_mask_hidden_drop",
    "feedback_required_score",
    "history_required_score",
    "drop_under_freeze_feedback_update_on_feedback_required",
    "drop_under_remove_history_on_history_required",
    "feedback_update_necessity_evidence",
    "history_necessity_evidence",
    "credit_buffer_required_score",
    "drop_under_disable_credit_buffer_on_delay5_required",
    "multiple_pending_credit_assignment_accuracy",
    "delayed_success_vs_no_effect_accuracy",
    "staged_backfire_credit_accuracy",
    "delay5_credit_buffer_necessity_evidence",
    "hidden_indirect_discovery_score",
    "candidate_search_fallback_score",
    "exploration_success_rate",
    "spurious_candidate_rejection_rate",
    "history_based_indirect_discovery_accuracy",
    "hidden_indirect_gap_to_oracle",
    "gain_over_state_only",
    "gain_over_mask_only",
    "gain_over_trace_only",
    "gain_over_random",
    "gain_over_always_abstain",
    "gain_over_conservative",
    "gap_to_oracle",
    "forbidden_reference_count",
    "poisoned_ground_truth_invariance_pass",
    "policy_uses_model_input_only",
    "score_cap_reason",
]

RECORD_FIELDS = [
    *SUMMARY_FIELDS,
    "episode_id",
    "selected_action_type",
    "selected_region",
    "expected_action_type",
    "expected_region",
    "selected_inspect",
    "expected_inspect",
    "trace_conflict_detected",
    "trace_repaired",
    "trace_confidence_downgraded",
    "repair_source",
    "feedback_used",
    "history_used",
    "inspection_used",
    "credit_buffer_used",
    "candidate_search_used",
    "hidden_indirect_success",
    "delayed_credit_success",
    "delayed_no_effect",
    "delayed_backfire",
    "false_safe_commit",
    "unnecessary_abstain",
    "failure_reason",
]


def score_output(episode: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    evaluator = episode["evaluator_ground_truth"]
    expected = evaluator["expected_action"]
    action = output.get("action")
    selected_type = "abstain" if action is None else str(action["action_type"])
    selected_region = "" if action is None else int(action["region_id"])
    expected_type = "abstain" if expected is None else str(expected["action_type"])
    expected_region = "" if expected is None else int(expected["region_id"])
    selected_inspect = int(bool(output.get("inspect")))
    expected_inspect = int(bool(evaluator.get("expected_inspect", False)))
    exact = selected_type == expected_type and selected_region == expected_region and selected_inspect == expected_inspect
    false_safe = int(selected_type == "apply_local_damping" and evaluator.get("true_unsafe_target", False))
    unnecessary_abstain = int(action is None and expected is not None)
    delayed_required = bool(evaluator.get("required_credit_buffer", False))
    delayed = output.get("delayed_credit", {})
    delayed_success = int(bool(delayed.get("credit_assigned", False)))
    hidden_required = bool(evaluator.get("hidden_indirect_target", False))
    hidden_success = int(bool(output.get("hidden_indirect", {}).get("success", False)))

    safety = 0.0 if false_safe else 1.0
    utility = 1.0 if exact else (0.0 if false_safe else (0.20 if unnecessary_abstain else 0.50))
    score = 1.0 if exact else 0.45 * safety + 0.55 * utility
    if false_safe:
        score = min(score, 0.20)
    if unnecessary_abstain:
        score = min(score, 0.45)
    if delayed_required and expected is not None and not delayed_success:
        score = min(score, 0.50)
    if hidden_required and expected_type == "indirect_stabilize" and not hidden_success:
        score = min(score, 0.55)

    repair = output.get("trace_repair", {})
    return {
        "episode_id": int(episode["metadata"]["episode_id"]),
        "selected_action_type": selected_type,
        "selected_region": selected_region,
        "expected_action_type": expected_type,
        "expected_region": expected_region,
        "selected_inspect": selected_inspect,
        "expected_inspect": expected_inspect,
        "risk_constrained_score": max(0.0, min(1.0, score)),
        "safety_score": safety,
        "utility_score": utility,
        "exact_action_accuracy": int(exact),
        "action_correctness_score": int(exact),
        "trace_conflict_detected": int(bool(output.get("trace_conflict", {}).get("trace_conflict_detected", False))),
        "trace_repaired": int(bool(repair.get("trace_repaired", False))),
        "trace_confidence_downgraded": int(bool(repair.get("trace_confidence_downgraded", False))),
        "repair_source": str(repair.get("repair_source", "none")),
        "feedback_used": int(repair.get("repair_source") == "feedback"),
        "history_used": int(repair.get("repair_source") == "history"),
        "inspection_used": int(repair.get("repair_source") == "inspect"),
        "credit_buffer_used": int(bool(output.get("delayed_credit", {}).get("buffer_used", False))),
        "candidate_search_used": int(bool(output.get("hidden_indirect", {}).get("candidate_search_used", False))),
        "hidden_indirect_success": hidden_success,
        "delayed_credit_success": delayed_success,
        "delayed_no_effect": int(bool(delayed.get("no_effect", False))),
        "delayed_backfire": int(bool(delayed.get("backfire", False))),
        "false_safe_commit": false_safe,
        "unnecessary_abstain": unnecessary_abstain,
        "failure_reason": str(output.get("failure_reason", "")),
    }


def summarize_policy(
    records: list[dict[str, Any]],
    context: dict[str, Any],
    baseline_scores: dict[str, float],
    audit: dict[str, Any],
) -> dict[str, Any]:
    no_sample = 1 if not records else 0
    score = mean(records, "risk_constrained_score")
    condition = context["condition"]
    row = {
        **context,
        "sample_count": len(records),
        "no_sample_metric_count": no_sample,
        "invalid_metric_count": no_sample,
        "risk_constrained_score": score,
        "safety_score": mean(records, "safety_score"),
        "utility_score": mean(records, "utility_score"),
        "exact_action_accuracy": mean(records, "exact_action_accuracy"),
        "action_correctness_score": mean(records, "exact_action_accuracy"),
        "wrong_trace_no_public_state_score": score if condition == "wrong_trace_no_public_state" else 0.0,
        "wrong_trace_history_conflict_score": score if condition == "wrong_trace_history_conflict" else 0.0,
        "wrong_trace_feedback_repair_score": score if condition == "wrong_trace_feedback_repair_required" else 0.0,
        "wrong_trace_inspection_recovery_score": score if condition == "wrong_trace_inspection_required" else 0.0,
        "trace_conflict_resolution_accuracy": mean([r for r in records if int(r.get("trace_conflict_detected", 0))], "exact_action_accuracy"),
        "trace_repair_source_distribution": source_distribution(records),
        "feedback_required_score": score if condition in {"feedback_required_trace_repair", "feedback_required_risk_update"} else 0.0,
        "history_required_score": score if condition in {"history_required_delayed_credit", "history_required_indirect_discovery"} else 0.0,
        "credit_buffer_required_score": score if condition.startswith("delay5_") else 0.0,
        "multiple_pending_credit_assignment_accuracy": mean(records, "delayed_credit_success") if condition == "delay5_multiple_pending_actions" else 0.0,
        "delayed_success_vs_no_effect_accuracy": mean(records, "delayed_credit_success") if condition == "delay5_no_effect_vs_delayed_success" else 0.0,
        "staged_backfire_credit_accuracy": score if condition == "delay5_backfire_after_success" else 0.0,
        "hidden_indirect_discovery_score": mean(records, "hidden_indirect_success") if condition.startswith("hidden_indirect_") else 0.0,
        "candidate_search_fallback_score": mean(records, "candidate_search_used") if condition.startswith("hidden_indirect_") else 0.0,
        "exploration_success_rate": mean([r for r in records if condition in {"hidden_indirect_no_candidate_shortcut", "hidden_indirect_exploration_required"}], "hidden_indirect_success"),
        "spurious_candidate_rejection_rate": mean(records, "hidden_indirect_success") if condition == "hidden_indirect_spurious_candidate" else 0.0,
        "history_based_indirect_discovery_accuracy": mean(records, "hidden_indirect_success") if condition == "hidden_indirect_history_discovery" else 0.0,
        "hidden_indirect_gap_to_oracle": max(0.0, baseline_scores.get("oracle", 1.0) - score) if condition.startswith("hidden_indirect_") else 0.0,
        "forbidden_reference_count": audit["forbidden_reference_count"],
        "poisoned_ground_truth_invariance_pass": audit["poisoned_ground_truth_invariance_pass"],
        "policy_uses_model_input_only": audit["policy_uses_model_input_only"],
        "score_cap_reason": "" if records else "no_samples",
    }
    row["gain_over_state_only"] = score - baseline_scores.get("state_only", 0.0)
    row["gain_over_mask_only"] = score - baseline_scores.get("mask_only", 0.0)
    row["gain_over_trace_only"] = score - baseline_scores.get("trace_only", 0.0)
    row["gain_over_random"] = score - baseline_scores.get("random", 0.0)
    row["gain_over_always_abstain"] = score - baseline_scores.get("always_abstain", 0.0)
    row["gain_over_conservative"] = score - baseline_scores.get("conservative_uncertainty", 0.0)
    row["gap_to_oracle"] = baseline_scores.get("oracle", 1.0) - score
    row["state_only_drop_on_wrong_trace_no_public_state"] = max(0.0, baseline_scores.get("oracle", 1.0) - baseline_scores.get("state_only", 0.0)) if condition == "wrong_trace_no_public_state" else 0.0
    row["feedback_repair_gain"] = score - baseline_scores.get("b63_1_no_feedback_update", 0.0) if condition in {"wrong_trace_feedback_repair_required", "feedback_required_trace_repair", "feedback_required_risk_update"} else 0.0
    row["history_repair_gain"] = score - baseline_scores.get("b63_1_no_history", 0.0) if condition in {"wrong_trace_history_conflict", "history_required_delayed_credit", "history_required_indirect_discovery"} else 0.0
    row["inspection_repair_gain"] = score - baseline_scores.get("b63_1_no_inspection_recovery", 0.0) if condition == "wrong_trace_inspection_required" else 0.0
    row["public_state_dependency_score"] = public_state_dependency(condition, baseline_scores)
    row["substitute_history_dependency_score"] = score - baseline_scores.get("b63_1_no_history", 0.0) if "history" in condition else 0.0
    row["substitute_feedback_dependency_score"] = score - baseline_scores.get("b63_1_no_feedback_update", 0.0) if "feedback" in condition else 0.0
    row["public_state_hidden_drop"] = max(0.0, baseline_scores.get("oracle", 1.0) - score) if condition == "hide_public_state_cue" else 0.0
    row["public_state_plus_history_hidden_drop"] = max(0.0, baseline_scores.get("oracle", 1.0) - score) if condition == "hide_public_state_and_history" else 0.0
    row["public_state_plus_feedback_hidden_drop"] = max(0.0, baseline_scores.get("oracle", 1.0) - score) if condition == "hide_public_state_and_feedback" else 0.0
    row["public_state_plus_mask_hidden_drop"] = max(0.0, baseline_scores.get("oracle", 1.0) - score) if condition == "hide_public_state_and_mask_target" else 0.0
    row["drop_under_freeze_feedback_update_on_feedback_required"] = row["feedback_repair_gain"] if condition in {"feedback_required_trace_repair", "feedback_required_risk_update"} else 0.0
    row["drop_under_remove_history_on_history_required"] = row["history_repair_gain"] if condition in {"history_required_delayed_credit", "history_required_indirect_discovery"} else 0.0
    row["feedback_update_necessity_evidence"] = row["drop_under_freeze_feedback_update_on_feedback_required"]
    row["history_necessity_evidence"] = row["drop_under_remove_history_on_history_required"]
    row["drop_under_disable_credit_buffer_on_delay5_required"] = score - baseline_scores.get("b63_1_no_credit_buffer", 0.0) if condition.startswith("delay5_") else 0.0
    row["delay5_credit_buffer_necessity_evidence"] = row["drop_under_disable_credit_buffer_on_delay5_required"]
    return row


def mean(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in records) / len(records)


def source_distribution(records: list[dict[str, Any]]) -> str:
    if not records:
        return ""
    counts = Counter(str(row.get("repair_source", "none")) for row in records)
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def public_state_dependency(condition: str, baseline_scores: dict[str, float]) -> float:
    if not condition.startswith("hide_public_state"):
        return 0.0
    return max(0.0, baseline_scores.get("oracle", 1.0) - baseline_scores.get("b63_1_policy", 0.0))
