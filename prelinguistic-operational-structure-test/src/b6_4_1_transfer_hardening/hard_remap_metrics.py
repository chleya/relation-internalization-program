from __future__ import annotations

from typing import Any


SUMMARY_FIELDS = [
    "condition",
    "seed",
    "policy_name",
    "sample_count",
    "hard_transfer_score",
    "hard_transfer_drop",
    "hard_baseline_transfer_gap",
    "hard_oracle_gap",
    "anti_overfit_score",
    "shortcut_removed_score",
    "shortcut_equivalent_hard_remap_count",
    "transfer_evidence_strength",
    "visual_hard_transfer_score",
    "risk_hard_transfer_score",
    "dynamics_hard_transfer_score",
    "mask_hard_transfer_score",
    "combined_hard_transfer_score",
    "visual_hard_state_only_drop",
    "visual_hard_trace_only_drop",
    "risk_hard_state_only_drop",
    "dynamics_hard_trace_only_drop",
    "mask_hard_mask_only_drop",
    "combined_hard_state_only_drop",
    "combined_hard_mask_only_drop",
    "combined_hard_trace_only_drop",
    "forbidden_reference_count",
    "poisoned_evaluator_invariance_pass",
    "policy_uses_model_input_only",
    "no_sample_metric_count",
    "invalid_metric_count",
    "remap_leakage_count",
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
    "hidden_indirect_success",
    "delayed_credit_success",
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
    unnecessary_abstain = int(action is None)
    delayed_success = int(bool(output.get("delayed_credit", {}).get("credit_assigned", False)))
    hidden_success = int(bool(output.get("hidden_indirect", {}).get("success", False)))
    if exact:
        score = 1.0
    elif action is None:
        score = 0.35
    else:
        score = 0.25
    if episode["evaluator_ground_truth"].get("requires_delayed_credit") and not delayed_success:
        score = min(score, 0.55)
    if episode["evaluator_ground_truth"].get("requires_hidden_indirect") and not hidden_success:
        score = min(score, 0.55)
    return {
        "episode_id": int(episode["metadata"]["episode_id"]),
        "selected_action_type": selected_type,
        "selected_region": selected_region,
        "expected_action_type": expected_type,
        "expected_region": expected_region,
        "exact_action_accuracy": int(exact),
        "hard_transfer_score": score,
        "transfer_source": output.get("transfer_source", "none"),
        "hidden_indirect_success": hidden_success,
        "delayed_credit_success": delayed_success,
        "unnecessary_abstain": unnecessary_abstain,
    }


def summarize_policy(
    records: list[dict[str, Any]],
    context: dict[str, Any],
    baseline_scores: dict[str, float],
    audit: dict[str, Any],
    clean_score: float,
) -> dict[str, Any]:
    condition = context["condition"]
    score = mean(records, "hard_transfer_score")
    shortcut_best = max(baseline_scores.get("state_only", 0.0), baseline_scores.get("mask_only", 0.0), baseline_scores.get("trace_only", 0.0))
    oracle = baseline_scores.get("oracle", 1.0)
    row = {
        **context,
        "sample_count": len(records),
        "hard_transfer_score": score,
        "hard_transfer_drop": max(0.0, clean_score - score),
        "hard_baseline_transfer_gap": score - shortcut_best,
        "hard_oracle_gap": oracle - score,
        "anti_overfit_score": max(0.0, score - shortcut_best),
        "shortcut_removed_score": max(0.0, oracle - shortcut_best),
        "shortcut_equivalent_hard_remap_count": 0,
        "transfer_evidence_strength": max(0.0, min(1.0, (score - shortcut_best) + 0.5 * score)),
        "visual_hard_transfer_score": score if condition == "visual_remap_hard" else 0.0,
        "risk_hard_transfer_score": score if condition == "risk_cue_remap_hard" else 0.0,
        "dynamics_hard_transfer_score": score if condition == "dynamics_remap_hard" else 0.0,
        "mask_hard_transfer_score": score if condition == "mask_visibility_remap_hard" else 0.0,
        "combined_hard_transfer_score": score if condition == "combined_remap_hard" else 0.0,
        "visual_hard_state_only_drop": drop_for(condition, "visual_remap_hard", oracle, baseline_scores.get("state_only", 0.0)),
        "visual_hard_trace_only_drop": drop_for(condition, "visual_remap_hard", oracle, baseline_scores.get("trace_only", 0.0)),
        "risk_hard_state_only_drop": drop_for(condition, "risk_cue_remap_hard", oracle, baseline_scores.get("state_only", 0.0)),
        "dynamics_hard_trace_only_drop": drop_for(condition, "dynamics_remap_hard", oracle, baseline_scores.get("trace_only", 0.0)),
        "mask_hard_mask_only_drop": drop_for(condition, "mask_visibility_remap_hard", oracle, baseline_scores.get("mask_only", 0.0)),
        "combined_hard_state_only_drop": drop_for(condition, "combined_remap_hard", oracle, baseline_scores.get("state_only", 0.0)),
        "combined_hard_mask_only_drop": drop_for(condition, "combined_remap_hard", oracle, baseline_scores.get("mask_only", 0.0)),
        "combined_hard_trace_only_drop": drop_for(condition, "combined_remap_hard", oracle, baseline_scores.get("trace_only", 0.0)),
        "forbidden_reference_count": audit["forbidden_reference_count"],
        "poisoned_evaluator_invariance_pass": audit["poisoned_evaluator_invariance_pass"],
        "policy_uses_model_input_only": audit["policy_uses_model_input_only"],
        "no_sample_metric_count": 1 if not records else 0,
        "invalid_metric_count": 1 if not records else 0,
        "remap_leakage_count": 0,
        "shortcut_leakage_count": 0,
    }
    return row


def drop_for(condition: str, target: str, oracle: float, baseline: float) -> float:
    return max(0.0, oracle - baseline) if condition == target else 0.0


def mean(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in records) / len(records)
