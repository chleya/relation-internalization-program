from __future__ import annotations

from typing import Any

from .b4_intervention_policy import trace_family_for_model
from .b5_closed_loop_env import inspect_region
from .b5_closed_loop_metrics import (
    epistemic_value_alignment,
    feedback_revision_accuracy as metric_feedback_revision_accuracy,
    inspect_timing_accuracy,
    mean_or_zero,
    post_inspection_intervention_accuracy,
    pragmatic_value_alignment,
    trace_update_accuracy as metric_trace_update_accuracy,
)
from .b5_epistemic_pragmatic_values import compute_closed_loop_value, compute_epistemic_value, compute_pragmatic_value
from .b5_feedback_revision import observe_consequence, revise_trace_after_feedback
from .b5_planning_budget import make_budget_record, planning_budget_compliant
from .b5_trace_update import export_trace_state, trace_uncertainty_reduction, update_trace_after_inspection


def closed_loop_policy(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gt = episode["ground_truth"]
    trace_before = export_trace_state(model, episode, config)
    signal = episode.get("closed_loop_signal", {})
    inspect_scores = {int(region): float(score) for region, score in signal.get("epistemic_scores", {}).items()}
    should_inspect = bool(signal.get("needs_inspection", False)) and float(trace_before.get("uncertainty", 0.0)) >= 0.5
    inspect_choice = max(inspect_scores.items(), key=lambda item: (item[1], -item[0]))[0] if should_inspect else -1
    inspection_result = inspect_region(episode, inspect_choice, config) if should_inspect else {"reveals_trace": False, "observed_trace_region": trace_before["region"], "epistemic_gain": 0.0}
    trace_after = update_trace_after_inspection(trace_before, inspection_result, episode, config) if should_inspect else dict(trace_before)
    should_intervene = bool(signal.get("should_intervene", True))
    action_type_signal = signal.get("action_type_signal", {})
    if should_intervene:
        action_type = max(action_type_signal.items(), key=lambda item: (float(item[1]), item[0]))[0]
        action = {
            "action_type": action_type,
            "region_id": int(trace_after["region"]),
            "strength": float(gt.get("oracle_intervention_action", {}).get("strength", 1.0)),
        }
    else:
        action = {"action_type": "do_nothing", "region_id": int(trace_after["region"]), "strength": 0.0}
    consequence = observe_consequence(episode, action, config)
    trace_feedback = revise_trace_after_feedback(trace_after, consequence, episode, config)
    budget = make_budget_record(2 if should_inspect else 1, 4 if should_intervene else 1, 6 if should_inspect and should_intervene else 3)
    output = {
        "inspect_chosen": should_inspect,
        "predicted_inspect_region": int(inspect_choice),
        "trace_before": trace_before,
        "trace_after_inspection": trace_after,
        "intervention_action": action,
        "consequence": consequence,
        "trace_after_feedback": trace_feedback,
        "trace_after_feedback_region": int(trace_feedback["region"]),
        "closed_loop_value": 0.0,
        "planning_budget": budget,
        "trace_family": trace_family_for_model(model),
        "policy_source": "private_trace_epistemic_pragmatic_closed_loop",
        "provenance": {
            "private_trace_used": True,
            "oracle_closed_loop_plan_used": False,
            "epistemic_pragmatic_values_separated": True,
        },
    }
    output["closed_loop_value"] = compute_closed_loop_value(output, episode, config)
    return output


def evaluate_closed_loop_policy(
    model: Any,
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int = 0,
    model_name: str | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    records = []
    inspect_hits = []
    epistemic_hits = []
    trace_hits = []
    reductions = []
    intervention_hits = []
    pragmatic_hits = []
    feedback_hits = []
    values = []
    budget_hits = []
    for episode_id, episode in enumerate(episodes):
        output = closed_loop_policy(model, episode, config)
        gt = episode["ground_truth"]
        inspect_hits.append(inspect_timing_accuracy(output, gt))
        epistemic_hits.append(epistemic_value_alignment(output, gt))
        trace_hits.append(metric_trace_update_accuracy(output["trace_after_inspection"], gt))
        reductions.append(trace_uncertainty_reduction(output["trace_before"], output["trace_after_inspection"]))
        intervention_hits.append(post_inspection_intervention_accuracy(output, gt))
        pragmatic_hits.append(pragmatic_value_alignment(output, gt))
        feedback_hits.append(metric_feedback_revision_accuracy(output, gt))
        values.append(float(output["closed_loop_value"]))
        budget_ok = 1.0 if planning_budget_compliant(output["planning_budget"], config) else 0.0
        budget_hits.append(budget_ok)
        action = output["intervention_action"]
        records.append(
            {
                "record_kind": "closed_loop_policy",
                "model": model_name or str(getattr(model, "name", "")),
                "seed": seed,
                "episode_id": episode_id,
                "episode_type": gt["closed_loop_episode_type"],
                "needs_inspection": int(gt["needs_inspection"]),
                "predicted_inspect_region": int(output["predicted_inspect_region"]),
                "oracle_inspect_region": int(gt["oracle_inspect_region"]),
                "inspect_skipped": int(not output["inspect_chosen"]),
                "epistemic_gain": compute_epistemic_value(episode, output["predicted_inspect_region"], config),
                "inspection_cost": float(gt["inspection_cost"]) if output["inspect_chosen"] else 0.0,
                "trace_before_region": int(output["trace_before"]["region"]),
                "trace_after_inspection_region": int(output["trace_after_inspection"]["region"]),
                "trace_update_correct": int(trace_hits[-1]),
                "predicted_intervention_action_type": action["action_type"],
                "predicted_intervention_region": int(action["region_id"]),
                "oracle_intervention_action_type": gt["oracle_intervention_action"]["action_type"],
                "oracle_intervention_region": int(gt["oracle_intervention_action"]["region_id"]),
                "pragmatic_gain": compute_pragmatic_value(episode, action, config),
                "intervention_cost": float(gt["intervention_cost"]) if action["action_type"] != "do_nothing" else 0.0,
                "consequence_value": float(output["consequence"]["consequence_value"]),
                "trace_after_feedback_region": int(output["trace_after_feedback_region"]),
                "feedback_revision_correct": int(feedback_hits[-1]),
                "closed_loop_value": float(output["closed_loop_value"]),
                "planning_budget_used": sum(int(v) for v in output["planning_budget"].values()),
                "gate_pass": int(min(inspect_hits[-1], trace_hits[-1], intervention_hits[-1], feedback_hits[-1], budget_ok)),
                "note": "private trace closed-loop policy",
            }
        )
    return {
        "inspect_timing_accuracy": mean_or_zero(inspect_hits),
        "epistemic_value_alignment": mean_or_zero(epistemic_hits),
        "trace_update_accuracy": mean_or_zero(trace_hits),
        "trace_uncertainty_reduction": mean_or_zero(reductions),
        "post_inspection_intervention_accuracy": mean_or_zero(intervention_hits),
        "pragmatic_value_alignment": mean_or_zero(pragmatic_hits),
        "feedback_revision_accuracy": mean_or_zero(feedback_hits),
        "closed_loop_model_score": mean_or_zero(values),
        "planning_budget_compliance": mean_or_zero(budget_hits),
    }, records
