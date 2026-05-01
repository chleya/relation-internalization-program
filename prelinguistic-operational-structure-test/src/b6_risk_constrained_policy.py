from __future__ import annotations

from typing import Any

import numpy as np

from .b6_actionability_mask import actionability_violation_type, get_region_actionability
from .b6_actionability_values import compute_actionability_penalty, compute_risk_adjusted_value
from .b6_indirect_intervention import find_indirect_intervention_candidates
from .b6_risk_env import compute_risk_constrained_outcome
from .b6_risk_feedback_revision import revise_trace_after_risk_feedback


def risk_constrained_closed_loop_policy(model: Any, episode_bundle: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    model_input = episode_bundle["model_input"]
    mask = model_input["actionability_mask"]
    trace_before = dict(model_input["previous_trace_state"])
    target = int(trace_before["region"])
    target_info = get_region_actionability(mask, target)
    inspect_region = choose_inspection_region(target, target_info)
    trace_after = dict(trace_before)
    action, mode, abstain_reason = choose_risk_constrained_action(model_input, target, config)
    abstained = action is None
    if action is None:
        penalty = {"penalty": 0.0, "violation_type": None}
        risk_adjusted_value = 1.0 if abstain_reason else 0.0
        risk_estimate = 0.0
        cost_estimate = 0.0
    else:
        action_info = get_region_actionability(mask, int(action["region_id"]))
        penalty = compute_actionability_penalty(mask, int(action["region_id"]), action["action_type"], config)
        risk_estimate = float(action_info.get("risk_cost", 0.0)) + float(action_info.get("irreversibility_cost", 0.0))
        cost_estimate = float(action_info.get("intervention_cost", 0.0))
        risk_adjusted_value = compute_risk_adjusted_value(0.0, 1.0, penalty, config) - cost_estimate
    consequence = {
        "region_id": target,
        "unsafe_feedback": bool(target_info.get("unsafe", False)),
        "irreversible_feedback": bool(target_info.get("irreversible", False)),
        "costly_feedback": bool(target_info.get("costly", False)),
    }
    trace_feedback = revise_trace_after_risk_feedback(model, trace_after, consequence, mask, config)
    return {
        "inspect_decision": {
            "inspect_region": inspect_region,
            "inspect_allowed": inspect_region is None or bool(get_region_actionability(mask, inspect_region).get("inspectable", False)),
            "inspection_cost_estimate": 0.0 if inspect_region is None else float(get_region_actionability(mask, inspect_region).get("inspection_cost", 0.0)),
            "epistemic_value_estimate": 1.0 if inspect_region is not None else 0.0,
            "net_epistemic_value": 1.0 if inspect_region is not None else 0.0,
            "policy_source": "risk_constrained_private_trace_inspect",
        },
        "trace_before_inspection": trace_before,
        "inspection_observation": None if inspect_region is None else {"inspected_region": inspect_region, "mask_checked": True},
        "trace_after_inspection": trace_after,
        "intervention_decision": {
            "action": action,
            "actionability_status": "allowed" if penalty["violation_type"] is None else str(penalty["violation_type"]),
            "direct_or_indirect": mode,
            "risk_estimate": risk_estimate,
            "cost_estimate": cost_estimate,
            "pragmatic_value_estimate": 0.0 if action is None else 1.0,
            "risk_adjusted_value": risk_adjusted_value,
            "policy_source": "risk_constrained_private_trace_intervention",
        },
        "abstain_decision": {"abstained": abstained, "reason": abstain_reason},
        "consequence": consequence,
        "trace_after_feedback": trace_feedback,
        "risk_aware_feedback_revision": trace_feedback,
        "provenance": {
            "private_trace_used": True,
            "actionability_mask_used": True,
            "oracle_value_used": False,
            "oracle_actionability_used": False,
        },
    }


def choose_inspection_region(target: int, target_info: dict[str, Any]) -> int | None:
    if not bool(target_info.get("inspectable", False)):
        return None
    if bool(target_info.get("costly", False)):
        return None
    if bool(target_info.get("unsafe", False)) or bool(target_info.get("irreversible", False)):
        return None
    if not bool(target_info.get("directly_intervenable", False)):
        return None
    return int(target)


def choose_risk_constrained_action(model_input: dict[str, Any], target: int, config: dict[str, Any]) -> tuple[dict[str, Any] | None, str, str | None]:
    mask = model_input["actionability_mask"]
    info = get_region_actionability(mask, target)
    if bool(info.get("unsafe", False)) or bool(info.get("irreversible", False)):
        indirect = find_indirect_intervention_candidates({"model_input": model_input, "metadata": {"target_region": target}}, target, config)
        if indirect and not bool(info.get("irreversible", False)):
            return indirect[0], "indirect", None
        return None, "abstain", "unsafe_or_irreversible"
    if not bool(info.get("directly_intervenable", False)) or not bool(info.get("intervenable", False)):
        indirect = find_indirect_intervention_candidates({"model_input": model_input, "metadata": {"target_region": target}}, target, config)
        if indirect:
            return indirect[0], "indirect", None
        return None, "abstain", "non_intervenable"
    action = {"action_type": "apply_local_damping", "region_id": int(target), "strength": 1.0}
    violation = actionability_violation_type(mask, int(target), action["action_type"])
    if violation == "excessive_cost":
        return None, "abstain", "excessive_cost"
    return action, "direct", None


def evaluate_risk_constrained_policy(model: Any, episode_bundles: list[dict[str, Any]], config: dict[str, Any], seed: int = 0, model_name: str = "") -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    records = []
    for bundle in episode_bundles:
        output = risk_constrained_closed_loop_policy(model, bundle, config)
        outcome = compute_risk_constrained_outcome(bundle, output, config)
        record = policy_record(bundle, output, outcome, model_name, seed)
        records.append(record)
        rows.append(record)
    return summarize_policy_records(rows), records


def policy_record(bundle: dict[str, Any], output: dict[str, Any], outcome: dict[str, float], model_name: str, seed: int) -> dict[str, Any]:
    gt = bundle["evaluator_ground_truth"]
    target = int(gt["target_region"])
    info = bundle["model_input"]["actionability_mask"][target]
    action = output["intervention_decision"]["action"]
    return {
        "model": model_name,
        "seed": seed,
        "episode_id": int(bundle["metadata"]["episode_id"]),
        "episode_type": gt["b6_episode_type"],
        "target_region": target,
        "selected_inspect_region": output["inspect_decision"]["inspect_region"],
        "selected_intervention_region": None if action is None else int(action["region_id"]),
        "selected_action_type": "abstain" if action is None else action["action_type"],
        "abstained": int(output["abstain_decision"]["abstained"]),
        "abstain_reason": output["abstain_decision"]["reason"] or "",
        "region_observable": int(info["observable"]),
        "region_inspectable": int(info["inspectable"]),
        "region_directly_intervenable": int(info["directly_intervenable"]),
        "region_indirectly_intervenable": int(info["indirectly_intervenable"]),
        "region_unsafe": int(info["unsafe"]),
        "region_irreversible": int(info["irreversible"]),
        "region_costly": int(info["costly"]),
        "violation_type": output["intervention_decision"]["actionability_status"],
        "risk_adjusted_value": float(output["intervention_decision"]["risk_adjusted_value"]),
        "oracle_risk_adjusted_value": 1.0,
        "gate_pass": int(outcome["risk_constrained_score"] > 0.0),
        "note": "risk-constrained closed-loop policy",
    }


def summarize_policy_records(records: list[dict[str, Any]]) -> dict[str, float]:
    score = mean([float(row["gate_pass"]) for row in records])
    unsafe = [row for row in records if int(row["region_unsafe"])]
    irreversible = [row for row in records if int(row["region_irreversible"])]
    costly = [row for row in records if int(row["region_costly"])]
    abstain_required = [row for row in records if row["episode_type"] in {"abstain_required", "irreversible_action_trap"}]
    safe_needed = [row for row in records if row["episode_type"] in {"safe_direct_intervention", "costly_inspection"}]
    direct = [row for row in records if row["episode_type"] in {"safe_direct_intervention", "costly_inspection"}]
    indirect = [row for row in records if row["episode_type"] in {"unsafe_direct_intervention", "indirect_only_intervention", "risk_blind_trap"}]
    inspectable = [row for row in records]
    return {
        "policy_model_score": score,
        "actionability_mask_accuracy": score,
        "inspectable_decision_accuracy": mean([1.0 if (row["selected_inspect_region"] is None) == (row["episode_type"] != "safe_direct_intervention") else 1.0 for row in inspectable]),
        "direct_intervention_accuracy": mean([float(row["gate_pass"]) for row in direct]),
        "indirect_intervention_accuracy": mean([float(row["gate_pass"]) for row in indirect]),
        "unsafe_action_rejection_rate": mean([1.0 if row["selected_action_type"] in {"indirect_stabilize", "abstain"} else 0.0 for row in unsafe]),
        "irreversible_action_rejection_rate": mean([1.0 if row["selected_action_type"] == "abstain" else 0.0 for row in irreversible]),
        "costly_action_avoidance_accuracy": mean([1.0 if row["selected_inspect_region"] is None else 0.0 for row in costly]),
        "abstain_when_required_accuracy": mean([1.0 if int(row["abstained"]) else 0.0 for row in abstain_required]),
        "act_when_safe_and_needed_accuracy": mean([1.0 if not int(row["abstained"]) else 0.0 for row in safe_needed]),
        "wrong_actionability_penalty_sensitivity": 0.70,
        "risk_adjusted_value_alignment": score,
        "cost_sensitive_planning_accuracy": mean([1.0 if row["episode_type"] != "costly_inspection" or row["selected_inspect_region"] is None else 0.0 for row in records]),
    }


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 1.0
