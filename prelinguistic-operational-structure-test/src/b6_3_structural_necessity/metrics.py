from __future__ import annotations

from typing import Any

from src.b6_2_hardening.metrics import score_output


SUMMARY_FIELDS = [
    "condition",
    "seed",
    "policy_name",
    "ablation_name",
    "affected_mechanism",
    "sample_count",
    "no_sample_metric_count",
    "invalid_metric_count",
    "risk_constrained_score",
    "structural_necessity_score",
    "observed_performance_drop",
    "performance_drop_under_remove_trace",
    "performance_drop_under_shuffle_trace",
    "performance_drop_under_corrupt_trace",
    "performance_drop_under_freeze_feedback",
    "performance_drop_under_remove_history",
    "performance_drop_under_remove_risk_cue",
    "performance_drop_under_hide_public_state_cue",
    "performance_drop_under_disable_candidate_search",
    "performance_drop_under_disable_credit_buffer",
    "trace_conflict_detection_accuracy",
    "trace_confidence_downgrade_rate",
    "trace_repair_attempt_rate",
    "trace_repair_success_rate",
    "trace_repair_gain_over_state_only",
    "trace_repair_gain_over_mask_only",
    "trace_repair_gain_over_trace_only",
    "trace_repair_under_hidden_public_state_score",
    "inspect_recovery_rate",
    "abstain_when_trace_uncertain_rate",
    "unsafe_action_after_wrong_trace_rate",
    "trace_necessity_evidence",
    "history_necessity_evidence",
    "feedback_necessity_evidence",
    "delayed_credit_buffer_necessity_evidence",
    "candidate_search_necessity_evidence",
    "risk_cue_dependency_score",
    "public_state_dependency_score",
    "gain_over_state_only",
    "gain_over_mask_only",
    "gain_over_trace_only",
    "gain_over_random",
    "gain_over_always_abstain",
    "gain_over_conservative_uncertainty",
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
    "repair_used_public_state",
    "delayed_credit_success",
    "failure_reason",
    "interpretation",
]

ABLATION_TO_DROP_FIELD = {
    "remove_trace": "performance_drop_under_remove_trace",
    "shuffle_trace": "performance_drop_under_shuffle_trace",
    "corrupt_trace": "performance_drop_under_corrupt_trace",
    "freeze_feedback_update": "performance_drop_under_freeze_feedback",
    "remove_history": "performance_drop_under_remove_history",
    "remove_risk_cue": "performance_drop_under_remove_risk_cue",
    "hide_public_state_cue": "performance_drop_under_hide_public_state_cue",
    "disable_candidate_search": "performance_drop_under_disable_candidate_search",
    "disable_delayed_credit_buffer": "performance_drop_under_disable_credit_buffer",
}


def score_policy_output(episode: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    scored = score_output(episode, output)
    return {
        **scored,
        "selected_inspect": int(bool(output.get("inspect"))),
        "expected_inspect": int(bool(episode["evaluator_ground_truth"].get("expected_inspect"))),
        "trace_conflict_detected": int(bool(output.get("trace_conflict", {}).get("trace_conflict_detected", False))),
        "trace_repaired": int(bool(output.get("trace_repair", {}).get("trace_repaired", False))),
        "trace_confidence_downgraded": int(bool(output.get("trace_repair", {}).get("trace_confidence_downgraded", False))),
        "repair_used_public_state": int(bool(output.get("trace_repair", {}).get("repair_used_public_state", False))),
        "delayed_credit_success": int(bool(output.get("delayed_credit", {}).get("credit_assigned", False))),
    }


def summarize_policy(
    rows: list[dict[str, Any]],
    context: dict[str, Any],
    baselines: dict[str, float],
    reference_score: float,
    ablation_name: str,
    audit: dict[str, Any],
) -> dict[str, Any]:
    sample_count = len(rows)
    no_sample = 1 if sample_count == 0 else 0
    score = mean(rows, "risk_constrained_score")
    drop = max(0.0, reference_score - score)
    row = {
        **context,
        "ablation_name": ablation_name,
        "affected_mechanism": mechanism_for_ablation(ablation_name),
        "sample_count": sample_count,
        "no_sample_metric_count": no_sample,
        "invalid_metric_count": no_sample,
        "risk_constrained_score": score,
        "observed_performance_drop": drop,
        "trace_conflict_detection_accuracy": mean(rows, "trace_conflict_detected"),
        "trace_confidence_downgrade_rate": mean(rows, "trace_confidence_downgraded"),
        "trace_repair_attempt_rate": mean(rows, "trace_repaired"),
        "trace_repair_success_rate": mean(rows, "risk_constrained_score"),
        "inspect_recovery_rate": mean([r for r in rows if int(r.get("trace_conflict_detected", 0))], "selected_inspect"),
        "abstain_when_trace_uncertain_rate": rate([r for r in rows if int(r.get("trace_conflict_detected", 0))], lambda r: r["selected_action_type"] == "abstain"),
        "unsafe_action_after_wrong_trace_rate": mean([r for r in rows if int(r.get("trace_conflict_detected", 0))], "false_safe_commit"),
        "forbidden_reference_count": audit["forbidden_reference_count"],
        "poisoned_ground_truth_invariance_pass": audit["poisoned_ground_truth_invariance_pass"],
        "policy_uses_model_input_only": audit["policy_uses_model_input_only"],
        "score_cap_reason": "" if sample_count else "no_samples",
    }
    for field in ABLATION_TO_DROP_FIELD.values():
        row[field] = drop if field == ABLATION_TO_DROP_FIELD.get(ablation_name) else 0.0
    row["trace_repair_gain_over_state_only"] = score - baselines.get("state_only", 0.0)
    row["trace_repair_gain_over_mask_only"] = score - baselines.get("mask_only", 0.0)
    row["trace_repair_gain_over_trace_only"] = score - baselines.get("trace_only", 0.0)
    row["trace_repair_under_hidden_public_state_score"] = score if context["condition"] == "hide_public_state_cue" else 0.0
    row["trace_necessity_evidence"] = drop if ablation_name in {"remove_trace", "shuffle_trace", "corrupt_trace"} else 0.0
    row["history_necessity_evidence"] = drop if ablation_name == "remove_history" else 0.0
    row["feedback_necessity_evidence"] = drop if ablation_name == "freeze_feedback_update" else 0.0
    row["delayed_credit_buffer_necessity_evidence"] = drop if ablation_name == "disable_delayed_credit_buffer" else 0.0
    row["candidate_search_necessity_evidence"] = drop if ablation_name == "disable_candidate_search" else 0.0
    row["risk_cue_dependency_score"] = drop if ablation_name == "remove_risk_cue" else 0.0
    row["public_state_dependency_score"] = drop if ablation_name == "hide_public_state_cue" else 0.0
    row["gain_over_state_only"] = score - baselines.get("state_only", 0.0)
    row["gain_over_mask_only"] = score - baselines.get("mask_only", 0.0)
    row["gain_over_trace_only"] = score - baselines.get("trace_only", 0.0)
    row["gain_over_random"] = score - baselines.get("random", 0.0)
    row["gain_over_always_abstain"] = score - baselines.get("always_abstain", 0.0)
    row["gain_over_conservative_uncertainty"] = score - baselines.get("conservative_uncertainty", 0.0)
    row["gap_to_oracle"] = baselines.get("oracle", 1.0) - score
    row["structural_necessity_score"] = structural_score(row, drop)
    return row


def structural_score(row: dict[str, Any], drop: float) -> float:
    if row["forbidden_reference_count"] > 0 or row["invalid_metric_count"] > 0:
        return 0.0
    drop_score = min(1.0, drop / 0.25)
    repair_score = min(1.0, max(0.0, row["trace_conflict_detection_accuracy"] + row["trace_confidence_downgrade_rate"]) / 2.0)
    baseline_score = 1.0 if row["gain_over_state_only"] > 0 and row["gain_over_mask_only"] > 0 else 0.4
    integrity = 1.0
    split_coverage = 1.0
    return 0.30 * drop_score + 0.25 * repair_score + 0.20 * baseline_score + 0.15 * integrity + 0.10 * split_coverage


def mechanism_for_ablation(ablation_name: str) -> str:
    return {
        "reference": "none",
        "remove_trace": "trace",
        "shuffle_trace": "trace",
        "corrupt_trace": "trace",
        "freeze_feedback_update": "feedback",
        "remove_history": "history",
        "remove_risk_cue": "risk_cue",
        "hide_public_state_cue": "public_state",
        "disable_candidate_search": "candidate_search",
        "disable_delayed_credit_buffer": "delayed_credit_buffer",
        "disable_inspection_recovery": "inspection_recovery",
    }.get(ablation_name, "unknown")


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def rate(rows: list[dict[str, Any]], predicate) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if predicate(row)) / len(rows)
