from __future__ import annotations

from typing import Any

from .combined_remap_env import PAIRWISE_CONDITIONS, TRIPLE_CONDITIONS


SUMMARY_FIELDS = [
    "condition",
    "seed",
    "policy_name",
    "sample_count",
    "combined_refinement_score",
    "combined_policy_score",
    "combined_oracle_score",
    "combined_oracle_gap",
    "combined_baseline_gap",
    "combined_recovery_score",
    "pairwise_transfer_drop",
    "triple_transfer_drop",
    "mechanism_specific_oracle_gap",
    "failure_attribution_confidence",
    "combined_failure_source",
    "pair_visual_risk_score",
    "pair_visual_dynamics_score",
    "pair_risk_mask_score",
    "pair_dynamics_delay_score",
    "pair_mask_indirect_score",
    "pair_delay_indirect_score",
    "triple_visual_risk_dynamics_score",
    "triple_risk_mask_delay_score",
    "triple_dynamics_delay_indirect_score",
    "triple_visual_mask_indirect_score",
    "triple_visual_risk_mask_score",
    "drop_under_combined_disable_trace_repair",
    "drop_under_combined_disable_fallback_risk",
    "drop_under_combined_disable_delayed_credit",
    "drop_under_combined_disable_candidate_search",
    "drop_under_combined_disable_inspection_recovery",
    "forbidden_reference_count",
    "poisoned_evaluator_invariance_pass",
    "policy_uses_model_input_only",
    "no_sample_metric_count",
    "invalid_metric_count",
    "combined_leakage_count",
    "shortcut_leakage_count",
]

RECORD_FIELDS = [
    *SUMMARY_FIELDS,
    "episode_id",
    "selected_action_type",
    "selected_region",
    "expected_action_type",
    "expected_region",
    "exact_action_accuracy",
    "transfer_source",
    "delayed_credit_success",
    "hidden_indirect_success",
    "unnecessary_abstain",
]


def score_output(episode: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    expected = episode["evaluator_ground_truth"]["expected_action"]
    action = output.get("action")
    selected_type = "abstain" if action is None else str(action["action_type"])
    selected_region = "" if action is None else int(action["region_id"])
    expected_type = str(expected["action_type"])
    expected_region = int(expected["region_id"])
    exact = selected_type == expected_type and selected_region == expected_region
    delayed_success = int(bool(output.get("delayed_credit", {}).get("credit_assigned", False)))
    hidden_success = int(bool(output.get("hidden_indirect", {}).get("success", False)))
    if exact:
        score = 1.0
    elif action is None:
        score = 0.35
    else:
        score = 0.25
    if episode["evaluator_ground_truth"].get("requires_credit") and not delayed_success:
        score = min(score, 0.55)
    if episode["evaluator_ground_truth"].get("requires_indirect") and not hidden_success and not delayed_success:
        score = min(score, 0.55)
    return {
        "episode_id": int(episode["metadata"]["episode_id"]),
        "selected_action_type": selected_type,
        "selected_region": selected_region,
        "expected_action_type": expected_type,
        "expected_region": expected_region,
        "exact_action_accuracy": int(exact),
        "combined_refinement_score": score,
        "transfer_source": output.get("transfer_source", "none"),
        "delayed_credit_success": delayed_success,
        "hidden_indirect_success": hidden_success,
        "unnecessary_abstain": int(action is None),
    }


def summarize_policy(records: list[dict[str, Any]], context: dict[str, Any], baseline_scores: dict[str, float], audit: dict[str, Any]) -> dict[str, Any]:
    condition = context["condition"]
    score = mean(records, "combined_refinement_score")
    oracle = baseline_scores.get("oracle", 1.0)
    shortcut_best = max(baseline_scores.get("state_only", 0.0), baseline_scores.get("mask_only", 0.0), baseline_scores.get("trace_only", 0.0))
    row = {
        **context,
        "sample_count": len(records),
        "combined_refinement_score": score,
        "combined_policy_score": score,
        "combined_oracle_score": oracle,
        "combined_oracle_gap": oracle - score,
        "combined_baseline_gap": score - shortcut_best,
        "combined_recovery_score": max(0.0, score - 0.675) if condition == "combined_remap_hard_reference" else 0.0,
        "pairwise_transfer_drop": max(0.0, oracle - score) if condition in PAIRWISE_CONDITIONS else 0.0,
        "triple_transfer_drop": max(0.0, oracle - score) if condition in TRIPLE_CONDITIONS else 0.0,
        "mechanism_specific_oracle_gap": oracle - score,
        "failure_attribution_confidence": max(0.0, min(1.0, (oracle - score) + max(0.0, score - shortcut_best))),
        "combined_failure_source": failure_source_from_records(records),
        "pair_visual_risk_score": score if condition == "pair_visual_risk" else 0.0,
        "pair_visual_dynamics_score": score if condition == "pair_visual_dynamics" else 0.0,
        "pair_risk_mask_score": score if condition == "pair_risk_mask" else 0.0,
        "pair_dynamics_delay_score": score if condition == "pair_dynamics_delay" else 0.0,
        "pair_mask_indirect_score": score if condition == "pair_mask_indirect" else 0.0,
        "pair_delay_indirect_score": score if condition == "pair_delay_indirect" else 0.0,
        "triple_visual_risk_dynamics_score": score if condition == "triple_visual_risk_dynamics" else 0.0,
        "triple_risk_mask_delay_score": score if condition == "triple_risk_mask_delay" else 0.0,
        "triple_dynamics_delay_indirect_score": score if condition == "triple_dynamics_delay_indirect" else 0.0,
        "triple_visual_mask_indirect_score": score if condition == "triple_visual_mask_indirect" else 0.0,
        "triple_visual_risk_mask_score": score if condition == "triple_visual_risk_mask" else 0.0,
        "drop_under_combined_disable_trace_repair": drop_for(condition, "combined_disable_trace_repair", oracle, score),
        "drop_under_combined_disable_fallback_risk": drop_for(condition, "combined_disable_fallback_risk", oracle, score),
        "drop_under_combined_disable_delayed_credit": drop_for(condition, "combined_disable_delayed_credit", oracle, score),
        "drop_under_combined_disable_candidate_search": drop_for(condition, "combined_disable_candidate_search", oracle, score),
        "drop_under_combined_disable_inspection_recovery": drop_for(condition, "combined_disable_inspection_recovery", oracle, score),
        "forbidden_reference_count": audit["forbidden_reference_count"],
        "poisoned_evaluator_invariance_pass": audit["poisoned_evaluator_invariance_pass"],
        "policy_uses_model_input_only": audit["policy_uses_model_input_only"],
        "no_sample_metric_count": 1 if not records else 0,
        "invalid_metric_count": 1 if not records else 0,
        "combined_leakage_count": 0,
        "shortcut_leakage_count": 0,
    }
    return row


def failure_source_from_records(records: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for row in records:
        source = str(row.get("transfer_source", "none"))
        counts[source] = counts.get(source, 0) + 1
    if not counts:
        return "no_samples"
    return max(counts.items(), key=lambda item: item[1])[0]


def drop_for(condition: str, target: str, oracle: float, score: float) -> float:
    return max(0.0, oracle - score) if condition == target else 0.0


def mean(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in records) / len(records)
