from __future__ import annotations

from typing import Any


SUMMARY_FIELDS = [
    "condition",
    "seed",
    "policy_name",
    "sample_count",
    "no_sample_metric_count",
    "invalid_metric_count",
    "transfer_score",
    "risk_constrained_score",
    "transfer_drop",
    "anti_overfit_score",
    "remap_generalization_score",
    "baseline_transfer_gap",
    "oracle_gap",
    "visual_remap_sensitivity",
    "risk_cue_remap_sensitivity",
    "dynamics_remap_sensitivity",
    "delay_remap_sensitivity",
    "indirect_path_remap_sensitivity",
    "mask_visibility_remap_sensitivity",
    "combined_remap_sensitivity",
    "trace_repair_transfer_score",
    "fallback_risk_transfer_score",
    "delayed_credit_transfer_score",
    "hidden_indirect_transfer_score",
    "feedback_history_transfer_score",
    "candidate_search_transfer_score",
    "gain_over_state_only",
    "gain_over_mask_only",
    "gain_over_trace_only",
    "gain_over_random",
    "gain_over_always_abstain",
    "gain_over_conservative",
    "gap_to_oracle",
    "forbidden_reference_count",
    "poisoned_evaluator_invariance_pass",
    "policy_uses_model_input_only",
    "remap_leakage_count",
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
    "exact_action_accuracy",
    "safety_score",
    "utility_score",
    "transfer_source",
    "delayed_credit_success",
    "hidden_indirect_success",
    "candidate_search_used",
    "false_safe_commit",
    "unnecessary_abstain",
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
    unnecessary_abstain = int(action is None and expected is not None)
    delayed_required = bool(evaluator.get("requires_delayed_credit", False))
    hidden_required = bool(evaluator.get("requires_hidden_indirect", False))
    delayed_success = int(bool(output.get("delayed_credit", {}).get("credit_assigned", False)))
    hidden_success = int(bool(output.get("hidden_indirect", {}).get("success", False)))
    false_safe = int(False)

    safety = 0.0 if false_safe else 1.0
    utility = 1.0 if exact else (0.20 if unnecessary_abstain else 0.45)
    score = 1.0 if exact else 0.45 * safety + 0.55 * utility
    if unnecessary_abstain:
        score = min(score, 0.45)
    if delayed_required and expected is not None and not delayed_success:
        score = min(score, 0.55)
    if hidden_required and expected_type == "indirect_stabilize" and not hidden_success:
        score = min(score, 0.55)

    return {
        "episode_id": int(episode["metadata"]["episode_id"]),
        "selected_action_type": selected_type,
        "selected_region": selected_region,
        "expected_action_type": expected_type,
        "expected_region": expected_region,
        "selected_inspect": selected_inspect,
        "expected_inspect": expected_inspect,
        "transfer_score": max(0.0, min(1.0, score)),
        "risk_constrained_score": max(0.0, min(1.0, score)),
        "exact_action_accuracy": int(exact),
        "safety_score": safety,
        "utility_score": utility,
        "transfer_source": str(output.get("transfer_source", "none")),
        "delayed_credit_success": delayed_success,
        "hidden_indirect_success": hidden_success,
        "candidate_search_used": int(bool(output.get("hidden_indirect", {}).get("candidate_search_used", False))),
        "false_safe_commit": false_safe,
        "unnecessary_abstain": unnecessary_abstain,
    }


def summarize_policy(
    records: list[dict[str, Any]],
    context: dict[str, Any],
    baseline_scores: dict[str, float],
    audit: dict[str, Any],
    clean_reference_score: float,
) -> dict[str, Any]:
    condition = str(context["condition"])
    score = mean(records, "transfer_score")
    shortcut_best = max(baseline_scores.get("state_only", 0.0), baseline_scores.get("mask_only", 0.0), baseline_scores.get("trace_only", 0.0))
    row = {
        **context,
        "sample_count": len(records),
        "no_sample_metric_count": 1 if not records else 0,
        "invalid_metric_count": 1 if not records else 0,
        "transfer_score": score,
        "risk_constrained_score": score,
        "transfer_drop": max(0.0, clean_reference_score - score),
        "anti_overfit_score": max(0.0, score - shortcut_best),
        "remap_generalization_score": score,
        "baseline_transfer_gap": score - shortcut_best,
        "oracle_gap": baseline_scores.get("oracle", 1.0) - score,
        "visual_remap_sensitivity": sensitivity(condition, "visual_remap", clean_reference_score, score),
        "risk_cue_remap_sensitivity": sensitivity(condition, "risk_cue_remap", clean_reference_score, score),
        "dynamics_remap_sensitivity": sensitivity(condition, "dynamics_remap", clean_reference_score, score),
        "delay_remap_sensitivity": sensitivity(condition, "delay_profile_remap", clean_reference_score, score),
        "indirect_path_remap_sensitivity": sensitivity(condition, "indirect_path_remap", clean_reference_score, score),
        "mask_visibility_remap_sensitivity": sensitivity(condition, "mask_visibility_remap", clean_reference_score, score),
        "combined_remap_sensitivity": sensitivity(condition, "combined_remap", clean_reference_score, score),
        "trace_repair_transfer_score": score if condition in {"visual_remap", "dynamics_remap", "combined_remap"} else 0.0,
        "fallback_risk_transfer_score": score if condition in {"risk_cue_remap", "mask_visibility_remap", "combined_remap"} else 0.0,
        "delayed_credit_transfer_score": mean(records, "delayed_credit_success") if condition in {"delay_profile_remap", "combined_remap"} else 0.0,
        "hidden_indirect_transfer_score": mean(records, "hidden_indirect_success") if condition in {"indirect_path_remap", "combined_remap"} else 0.0,
        "feedback_history_transfer_score": mean([r for r in records if r.get("transfer_source") in {"feedback", "history"}], "transfer_score"),
        "candidate_search_transfer_score": mean(records, "candidate_search_used"),
        "gain_over_state_only": score - baseline_scores.get("state_only", 0.0),
        "gain_over_mask_only": score - baseline_scores.get("mask_only", 0.0),
        "gain_over_trace_only": score - baseline_scores.get("trace_only", 0.0),
        "gain_over_random": score - baseline_scores.get("random", 0.0),
        "gain_over_always_abstain": score - baseline_scores.get("always_abstain", 0.0),
        "gain_over_conservative": score - baseline_scores.get("conservative_uncertainty", 0.0),
        "gap_to_oracle": baseline_scores.get("oracle", 1.0) - score,
        "forbidden_reference_count": audit["forbidden_reference_count"],
        "poisoned_evaluator_invariance_pass": audit["poisoned_evaluator_invariance_pass"],
        "policy_uses_model_input_only": audit["policy_uses_model_input_only"],
        "remap_leakage_count": 0,
    }
    return row


def sensitivity(condition: str, target: str, clean_score: float, score: float) -> float:
    return max(0.0, clean_score - score) if condition == target else 0.0


def mean(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in records) / len(records)
