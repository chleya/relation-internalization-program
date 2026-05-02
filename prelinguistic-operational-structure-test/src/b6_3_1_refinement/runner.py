from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
from typing import Any, Callable

from .credit_buffer_necessity import CREDIT_BUFFER_CONDITIONS, apply_credit_buffer_condition, assign_credit_with_buffer
from .dependency_sharpening import PUBLIC_STATE_DEPENDENCY_CONDITIONS, apply_public_state_dependency_condition
from .feedback_history_necessity import FEEDBACK_HISTORY_CONDITIONS, apply_feedback_history_condition
from .hidden_indirect_discovery import HIDDEN_INDIRECT_CONDITIONS, apply_hidden_indirect_condition, discover_hidden_indirect
from .metrics import RECORD_FIELDS, SUMMARY_FIELDS, score_output, summarize_policy
from .wrong_trace_hardening import WRONG_TRACE_CONDITIONS, apply_wrong_trace_condition, trace_conflict_resolution_source


CONDITIONS = [
    *WRONG_TRACE_CONDITIONS,
    *PUBLIC_STATE_DEPENDENCY_CONDITIONS,
    *FEEDBACK_HISTORY_CONDITIONS,
    *CREDIT_BUFFER_CONDITIONS,
    *HIDDEN_INDIRECT_CONDITIONS,
]

POLICY_NAMES = [
    "b63_1_policy",
    "state_only",
    "mask_only",
    "trace_only",
    "random",
    "always_abstain",
    "conservative_uncertainty",
    "oracle",
    "b63_1_no_feedback_update",
    "b63_1_no_history",
    "b63_1_no_credit_buffer",
    "b63_1_no_candidate_search",
    "b63_1_no_inspection_recovery",
]


def make_base_episode(seed: int) -> dict[str, Any]:
    total = 64
    target = (seed * 7 + 11) % total
    wrong = (target + 13) % total
    state_decoy = (target + 19) % total
    feedback_decoy = (target + 29) % total
    indirect = (target + 9) % total
    mask = {
        region: {
            "region_id": region,
            "inspectable": True,
            "directly_intervenable": True,
            "unsafe": False,
            "risk_cost": 0.0,
            "indirect_target_region": None,
        }
        for region in range(total)
    }
    mask[target]["indirect_target_region"] = indirect
    visible = {
        "public_state_available": True,
        "state_target_hint": target,
        "state_target_confidence": 0.85,
        "history_supported_region": None,
        "history_confidence": 0.0,
        "feedback_history_region": None,
        "feedback_confidence": 0.0,
        "inspection_observation_region": None,
        "inspection_confidence": 0.0,
        "risk_history_score": 0.10,
        "candidate_indirect_regions": [{"region_id": indirect, "causal_strength": 0.90, "backfire_estimate": 0.10}],
        "indirect_history_paths": [],
        "exploration_history": [],
        "pending_indirect_actions": [],
        "outcome_history": [],
        "delay_steps": 0,
    }
    return {
        "model_input": {
            "episode_id": seed,
            "visible_state": visible,
            "previous_trace_state": {"region": target, "confidence": 0.90, "candidate_regions": [target], "trace_mode": "correct"},
            "actionability_mask": mask,
        },
        "evaluator_ground_truth": {
            "condition": "",
            "target_region": target,
            "wrong_target_region": wrong,
            "state_decoy_region": state_decoy,
            "feedback_decoy_region": feedback_decoy,
            "indirect_target_region": indirect,
            "expected_action": {"action_type": "apply_local_damping", "region_id": target},
            "expected_inspect": False,
            "true_unsafe_target": False,
            "required_repair_source": "trace",
            "required_credit_buffer": False,
            "hidden_indirect_target": False,
        },
        "oracle_baseline_view": {"expected_action": {"action_type": "apply_local_damping", "region_id": target}, "expected_inspect": False},
        "metadata": {"episode_id": seed, "condition": ""},
    }


def make_b631_episode(config: dict[str, Any], seed: int, condition: str) -> dict[str, Any]:
    episode = make_base_episode(seed)
    episode["evaluator_ground_truth"]["condition"] = condition
    episode["metadata"]["condition"] = condition
    if condition in WRONG_TRACE_CONDITIONS:
        apply_wrong_trace_condition(episode, condition)
    elif condition in PUBLIC_STATE_DEPENDENCY_CONDITIONS:
        apply_public_state_dependency_condition(episode, condition)
    elif condition in FEEDBACK_HISTORY_CONDITIONS:
        apply_feedback_history_condition(episode, condition)
    elif condition in CREDIT_BUFFER_CONDITIONS:
        apply_credit_buffer_condition(episode, condition)
    elif condition in HIDDEN_INDIRECT_CONDITIONS:
        apply_hidden_indirect_condition(episode, condition)
    else:
        raise ValueError(f"unknown B6.3.1 condition: {condition}")
    episode["oracle_baseline_view"] = {
        "expected_action": episode["evaluator_ground_truth"]["expected_action"],
        "expected_inspect": episode["evaluator_ground_truth"]["expected_inspect"],
    }
    return episode


def make_datasets(config: dict[str, Any], seed: int) -> dict[str, list[dict[str, Any]]]:
    section = config.get("b6_3_1", {})
    n = int(section.get("episodes_per_condition", 6))
    conditions = list(section.get("conditions", CONDITIONS))
    return {
        condition: [make_b631_episode(config, seed + cidx * 1000 + idx, condition) for idx in range(n)]
        for cidx, condition in enumerate(conditions)
    }


def b631_policy(episode: dict[str, Any], config: dict[str, Any], disabled: set[str] | None = None) -> dict[str, Any]:
    disabled = disabled or set()
    model_input = episode["model_input"]
    visible = model_input.get("visible_state", {})
    conflict = trace_conflict_resolution_source(model_input, disabled)
    repair_region = conflict.get("repair_candidate_region")
    repair_source = str(conflict.get("repair_source", "none"))
    inspect = repair_source == "inspect" and "inspection" not in disabled
    delayed_credit = {"credit_assigned": False, "buffer_used": False, "no_effect": False, "backfire": False}
    hidden = {"success": False, "candidate_search_used": False}
    action = None
    failure_reason = ""

    if visible.get("feedback_risk_marker") == "unsafe" and "feedback" not in disabled:
        failure_reason = "feedback_risk_update"
    elif visible.get("pending_indirect_actions") and "history" not in disabled:
        credit = assign_credit_with_buffer(model_input, disabled)
        action = credit["action"]
        delayed_credit = {**credit["credit"], "buffer_used": "credit_buffer" not in disabled}
        if delayed_credit.get("backfire"):
            action = None
            failure_reason = "staged_backfire"
        elif action is None:
            failure_reason = "credit_not_assigned"
    elif (
        (visible.get("indirect_history_paths") or visible.get("exploration_history"))
        or visible.get("candidate_indirect_regions") and visible.get("risk_history_score", 0.0) >= 0.75
    ):
        found = discover_hidden_indirect(model_input, disabled)
        action = found["action"]
        hidden = {"success": bool(found["success"]), "candidate_search_used": found["source"] == "candidate_search", "discovery_source": found["source"]}
        failure_reason = "" if action else found["source"]
    elif repair_region is not None and float(conflict.get("repair_confidence", 0.0)) >= 0.55:
        action = {"action_type": "apply_local_damping", "region_id": int(repair_region)}
    else:
        failure_reason = "unresolved_trace_conflict"

    return {
        "policy_name": "b63_1_policy",
        "inspect": inspect,
        "action": action,
        "trace_conflict": conflict,
        "trace_repair": {
            "trace_repaired": repair_source not in {"trace", "none"},
            "trace_confidence_downgraded": conflict["trace_confidence_after"] < conflict["trace_confidence_before"],
            "repair_source": repair_source,
        },
        "delayed_credit": delayed_credit,
        "hidden_indirect": hidden,
        "failure_reason": failure_reason,
        "provenance": {"policy_uses_model_input_only": True, "oracle_value_used": False},
    }


def baseline_policy(name: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    def output(action: dict[str, Any] | None, inspect: bool = False, reason: str = "") -> dict[str, Any]:
        return {"policy_name": name, "inspect": inspect, "action": action, "failure_reason": reason, "provenance": {"policy_uses_model_input_only": True}}

    def run(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        model_input = episode["model_input"]
        visible = model_input["visible_state"]
        trace = model_input.get("previous_trace_state", {})
        if name == "state_only":
            if not visible.get("public_state_available", True) or float(visible.get("state_target_confidence", 0.0)) < 0.55:
                return output(None, False, "state_unavailable")
            return output({"action_type": "apply_local_damping", "region_id": int(visible["state_target_hint"])})
        if name == "mask_only":
            mask = model_input.get("actionability_mask") or {}
            for info in mask.values():
                if info.get("indirect_target_region") is not None:
                    return output({"action_type": "indirect_stabilize", "region_id": int(info["indirect_target_region"])})
            return output(None, False, "mask_no_target")
        if name == "trace_only":
            region = trace.get("region")
            return output(None, True, "missing_trace") if region is None else output({"action_type": "apply_local_damping", "region_id": int(region)})
        if name == "random":
            region = int(visible.get("state_target_hint", 0))
            return output(None if int(model_input["episode_id"]) % 3 == 0 else {"action_type": "apply_local_damping", "region_id": region})
        if name == "always_abstain":
            return output(None, False, "always_abstain")
        if name == "conservative_uncertainty":
            trace_conf = float(model_input.get("previous_trace_state", {}).get("confidence", 0.0))
            public_conf = float(visible.get("state_target_confidence", 0.0)) if visible.get("public_state_available", True) else 0.0
            if trace_conf < 0.70 or public_conf < 0.55 or float(visible.get("risk_history_score", 0.0)) >= 0.75:
                return output(None, True, "conservative_uncertainty")
            return output({"action_type": "apply_local_damping", "region_id": int(visible["state_target_hint"])})
        if name == "oracle":
            expected = episode["oracle_baseline_view"]["expected_action"]
            out = output(expected, bool(episode["oracle_baseline_view"]["expected_inspect"]))
            if expected and expected.get("action_type") == "indirect_stabilize":
                out["delayed_credit"] = {"credit_assigned": True, "buffer_used": True}
                out["hidden_indirect"] = {"success": True, "candidate_search_used": False}
            return out
        raise KeyError(name)

    return run


def policy_for(name: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    disabled_map = {
        "b63_1_no_feedback_update": {"feedback"},
        "b63_1_no_history": {"history"},
        "b63_1_no_credit_buffer": {"credit_buffer"},
        "b63_1_no_candidate_search": {"candidate_search"},
        "b63_1_no_inspection_recovery": {"inspection"},
    }
    if name == "b63_1_policy":
        return lambda episode, config: b631_policy(episode, config)
    if name in disabled_map:
        return lambda episode, config, disabled=disabled_map[name]: b631_policy(episode, config, disabled)
    return baseline_policy(name)


def audit_policy_integrity(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    source = inspect.getsource(b631_policy)
    forbidden = ["evaluator_ground_truth", "oracle_baseline_view", "expected_decision", "metadata"]
    forbidden_count = sum(1 for token in forbidden if token in source)
    original = b631_policy(episode, config)
    poisoned = {
        **episode,
        "evaluator_ground_truth": {**episode["evaluator_ground_truth"], "expected_action": None, "condition": "poisoned"},
        "metadata": {**episode["metadata"], "condition": "poisoned"},
    }
    poisoned_out = b631_policy(poisoned, config)
    return {
        "forbidden_reference_count": forbidden_count,
        "poisoned_ground_truth_invariance_pass": comparable_output(original) == comparable_output(poisoned_out),
        "policy_uses_model_input_only": forbidden_count == 0,
    }


def comparable_output(output: dict[str, Any]) -> tuple[Any, ...]:
    action = output.get("action") or {}
    return (
        action.get("action_type"),
        action.get("region_id"),
        bool(output.get("inspect")),
        output.get("trace_repair", {}).get("repair_source"),
    )


def run_b6_3_1_refinement(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    datasets = make_datasets(config, seed)
    summary: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for condition, episodes in datasets.items():
        rows_by_policy: dict[str, list[dict[str, Any]]] = {}
        audit = audit_policy_integrity(episodes[0], config) if episodes else {"forbidden_reference_count": 0, "poisoned_ground_truth_invariance_pass": False, "policy_uses_model_input_only": False}
        for policy_name in POLICY_NAMES:
            rows = []
            policy = policy_for(policy_name)
            for episode in episodes:
                output = policy(episode, config)
                scored = score_output(episode, output)
                context = {"condition": condition, "seed": seed, "policy_name": policy_name}
                row = {**context, **scored}
                rows.append(row)
                records.append(row)
            rows_by_policy[policy_name] = rows
        baseline_scores = {name: mean_score(rows) for name, rows in rows_by_policy.items()}
        for policy_name, rows in rows_by_policy.items():
            summary.append(summarize_policy(rows, {"condition": condition, "seed": seed, "policy_name": policy_name}, baseline_scores, audit))
    metrics = build_metrics(summary, records)
    return summary, records, metrics


def build_metrics(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    b631 = [row for row in summary if row["policy_name"] == "b63_1_policy"]
    by_condition = {row["condition"]: row for row in b631}
    return {
        "summary_rows": len(summary),
        "record_rows": len(records),
        "conditions": sorted({row["condition"] for row in summary}),
        "b63_1_policy_mean_score": mean_score(b631),
        "wrong_trace_no_public_state_score": by_condition.get("wrong_trace_no_public_state", {}).get("risk_constrained_score", 0.0),
        "state_only_drop_on_wrong_trace_no_public_state": by_condition.get("wrong_trace_no_public_state", {}).get("state_only_drop_on_wrong_trace_no_public_state", 0.0),
        "feedback_required_drop": max(row.get("drop_under_freeze_feedback_update_on_feedback_required", 0.0) for row in b631) if b631 else 0.0,
        "history_required_drop": max(row.get("drop_under_remove_history_on_history_required", 0.0) for row in b631) if b631 else 0.0,
        "credit_buffer_required_drop": max(row.get("drop_under_disable_credit_buffer_on_delay5_required", 0.0) for row in b631) if b631 else 0.0,
        "hidden_indirect_discovery_score": max(row.get("hidden_indirect_discovery_score", 0.0) for row in b631) if b631 else 0.0,
        "forbidden_reference_count_max": max((int(row["forbidden_reference_count"]) for row in summary), default=0),
        "invalid_metric_count_total": sum(int(row["invalid_metric_count"]) for row in summary),
        "submit_ready_as_diagnostic": True,
    }


def mean_score(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get("risk_constrained_score", 0.0)) for row in rows) / len(rows)


def write_b6_3_1_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    write_csv(Path("results/b6_3_1_refinement_summary.csv"), summary, SUMMARY_FIELDS)
    write_csv(Path("results/b6_3_1_refinement_records.csv"), records, RECORD_FIELDS)
    Path("results/b6_3_1_refinement_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    Path("reports/B6_3_1_WRONG_TRACE_MECHANISM_REFINEMENT.md").write_text(build_report(metrics), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_report(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# B6.3.1 Wrong-Trace and Mechanism Necessity Refinement",
            "",
            "## Purpose",
            "B6.3.1 sharpens B6.3 blockers without entering B7 or making real-world safety claims.",
            "",
            "## Key Results",
            f"- b63_1_policy_mean_score = {metrics['b63_1_policy_mean_score']:.3f}",
            f"- wrong_trace_no_public_state_score = {float(metrics['wrong_trace_no_public_state_score']):.3f}",
            f"- state_only_drop_on_wrong_trace_no_public_state = {float(metrics['state_only_drop_on_wrong_trace_no_public_state']):.3f}",
            f"- feedback_required_drop = {float(metrics['feedback_required_drop']):.3f}",
            f"- history_required_drop = {float(metrics['history_required_drop']):.3f}",
            f"- credit_buffer_required_drop = {float(metrics['credit_buffer_required_drop']):.3f}",
            f"- hidden_indirect_discovery_score = {float(metrics['hidden_indirect_discovery_score']):.3f}",
            "",
            "## Interpretation",
            "B6.3.1 is a diagnostic refinement. It may support narrower split-level mechanism evidence only where targeted ablations cause specific drops.",
            "",
            "High aggregate score must not be interpreted as solved robustness. wrong_trace repair, feedback/history necessity, credit-buffer necessity, and hidden-indirect discovery remain toy diagnostics.",
            "",
            "Hidden indirect discovery in this stage uses synthetic exploration/outcome-history cues. It is not real-world causal discovery.",
            "",
            "B6.3.1 does not prove real-world risk intelligence, robotics capability, safety certification, construction-site autonomy, or deployable engineering control.",
            "",
        ]
    )
