from __future__ import annotations

from typing import Any


PAIRWISE_CONDITIONS = [
    "pair_visual_risk",
    "pair_visual_dynamics",
    "pair_risk_mask",
    "pair_dynamics_delay",
    "pair_mask_indirect",
    "pair_delay_indirect",
]

TRIPLE_CONDITIONS = [
    "triple_visual_risk_dynamics",
    "triple_risk_mask_delay",
    "triple_dynamics_delay_indirect",
    "triple_visual_mask_indirect",
    "triple_visual_risk_mask",
]

ABLATION_CONDITIONS = [
    "combined_disable_trace_repair",
    "combined_disable_fallback_risk",
    "combined_disable_delayed_credit",
    "combined_disable_candidate_search",
    "combined_disable_inspection_recovery",
]

CONDITIONS = [
    "combined_remap_hard_reference",
    *PAIRWISE_CONDITIONS,
    *TRIPLE_CONDITIONS,
    *ABLATION_CONDITIONS,
]

CONDITION_FAILURE_SOURCE = {
    "combined_remap_hard_reference": "multi_cue_interaction",
    "pair_visual_risk": "visual_risk_interaction",
    "pair_visual_dynamics": "visual_dynamics_interaction",
    "pair_risk_mask": "risk_mask_interaction",
    "pair_dynamics_delay": "dynamics_delay_interaction",
    "pair_mask_indirect": "mask_indirect_interaction",
    "pair_delay_indirect": "delay_indirect_interaction",
    "triple_visual_risk_dynamics": "visual_risk_dynamics_interaction",
    "triple_risk_mask_delay": "risk_mask_delay_interaction",
    "triple_dynamics_delay_indirect": "dynamics_delay_indirect_interaction",
    "triple_visual_mask_indirect": "visual_mask_indirect_interaction",
    "triple_visual_risk_mask": "visual_risk_mask_interaction",
    "combined_disable_trace_repair": "trace_repair_ablation",
    "combined_disable_fallback_risk": "fallback_risk_ablation",
    "combined_disable_delayed_credit": "delayed_credit_ablation",
    "combined_disable_candidate_search": "candidate_search_ablation",
    "combined_disable_inspection_recovery": "inspection_recovery_ablation",
}


def make_b642_episode(config: dict[str, Any], seed: int, condition: str) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown B6.4.2 condition: {condition}")
    total = int(config.get("b6_4_2", {}).get("total_regions", 64))
    target = (seed * 11 + 19) % total
    decoy = (target + 17) % total
    trace_decoy = (target + 23) % total
    indirect = (target + 7) % total
    spurious = (target + 31) % total
    episode_index = int(seed % 4)
    profile = condition_profile(condition, episode_index)
    visible = base_visible(target, decoy, trace_decoy, indirect, spurious, condition, profile)
    mask = public_mask(total, target, indirect)
    hide_answer_like_mask(mask)
    if profile.get("disable_candidate_search"):
        visible["exploration_history"] = []
    if profile.get("disable_delayed_credit"):
        visible["pending_indirect_actions"] = []
        visible["outcome_history"] = []
    if profile.get("disable_trace_repair"):
        visible["trace_repair_region"] = None
        visible["trace_repair_confidence"] = 0.0
    if profile.get("disable_fallback_risk"):
        visible["fallback_risk_region"] = None
        visible["fallback_risk_confidence"] = 0.0
    if profile.get("disable_inspection_recovery"):
        visible["inspection_observation_region"] = None
        visible["inspection_confidence"] = 0.0

    expected = {"action_type": "indirect_stabilize", "region_id": indirect}
    if profile["expected_mode"] == "direct":
        expected = {"action_type": "apply_local_damping", "region_id": target}

    model_input = {
        "episode_id": seed,
        "visible_state": visible,
        "previous_trace_state": {"region": trace_decoy, "confidence": 0.42, "candidate_regions": [trace_decoy, target]},
        "actionability_mask": mask,
    }
    evaluator = {
        "condition": condition,
        "target_region": target,
        "indirect_target_region": indirect,
        "expected_action": expected,
        "failure_source": CONDITION_FAILURE_SOURCE[condition],
        "requires_indirect": expected["action_type"] == "indirect_stabilize",
        "requires_credit": bool(profile.get("requires_credit", False)),
    }
    return {
        "model_input": model_input,
        "evaluator_ground_truth": evaluator,
        "oracle_baseline_view": {"expected_action": expected, "failure_source": evaluator["failure_source"]},
        "metadata": {"episode_id": seed, "condition": condition},
    }


def condition_profile(condition: str, episode_index: int) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "expected_mode": "direct",
        "trace_repair": True,
        "fallback_risk": True,
        "delayed_credit": False,
        "candidate_search": False,
        "inspection_recovery": True,
        "requires_credit": False,
    }
    if condition in {
        "pair_dynamics_delay",
        "pair_mask_indirect",
        "pair_delay_indirect",
        "triple_risk_mask_delay",
        "triple_visual_mask_indirect",
        "triple_dynamics_delay_indirect",
        "combined_remap_hard_reference",
    }:
        profile.update({"expected_mode": "indirect", "candidate_search": True})
    if condition in {"pair_dynamics_delay", "pair_delay_indirect", "triple_risk_mask_delay", "triple_dynamics_delay_indirect", "combined_remap_hard_reference"}:
        profile.update({"delayed_credit": True, "requires_credit": True})
    if condition == "combined_remap_hard_reference":
        if episode_index in {0, 1}:
            profile.update({"candidate_search": True, "delayed_credit": True})
        else:
            profile.update({"candidate_search": False, "delayed_credit": False, "inspection_recovery": False})
    elif condition == "triple_dynamics_delay_indirect" and episode_index >= 2:
        profile.update({"candidate_search": False, "delayed_credit": False})
    elif condition in {"pair_dynamics_delay", "pair_delay_indirect", "triple_risk_mask_delay"} and episode_index == 3:
        profile.update({"delayed_credit": False})
    elif condition == "triple_visual_mask_indirect" and episode_index == 3:
        profile.update({"candidate_search": False})
    elif condition == "combined_disable_trace_repair":
        profile.update({"disable_trace_repair": True})
    elif condition == "combined_disable_fallback_risk":
        profile.update({"disable_fallback_risk": True})
    elif condition == "combined_disable_delayed_credit":
        profile.update({"expected_mode": "indirect", "disable_delayed_credit": True, "candidate_search": False, "requires_credit": True})
    elif condition == "combined_disable_candidate_search":
        profile.update({"expected_mode": "indirect", "disable_candidate_search": True, "delayed_credit": False})
    elif condition == "combined_disable_inspection_recovery":
        profile.update({"disable_inspection_recovery": True, "trace_repair": False, "fallback_risk": False})
    return profile


def base_visible(target: int, decoy: int, trace_decoy: int, indirect: int, spurious: int, condition: str, profile: dict[str, Any]) -> dict[str, Any]:
    visible: dict[str, Any] = {
        "public_state_available": False,
        "state_target_hint": decoy,
        "state_target_confidence": 0.0,
        "visual_channels": {"primary_region": decoy, "distractor_regions": [trace_decoy, spurious]},
        "risk_encoding": "combined_rotated_hidden",
        "risk_score_estimate": 0.70,
        "dynamics_profile": {"velocity_scale": 2.2, "friction": 0.13, "drift": 0.41, "collision_strength": 1.7},
        "trace_repair_region": target if profile.get("trace_repair") else None,
        "trace_repair_confidence": 0.74 if profile.get("trace_repair") else 0.0,
        "fallback_risk_region": target if profile.get("fallback_risk") else None,
        "fallback_risk_confidence": 0.76 if profile.get("fallback_risk") else 0.0,
        "history_supported_region": target,
        "history_confidence": 0.68,
        "feedback_history_region": target,
        "feedback_confidence": 0.66,
        "transition_consistency_region": target,
        "transition_confidence": 0.70,
        "inspection_observation_region": target if profile.get("inspection_recovery") else None,
        "inspection_confidence": 0.78 if profile.get("inspection_recovery") else 0.0,
        "pending_indirect_actions": [],
        "outcome_history": [],
        "exploration_history": [],
        "candidate_indirect_regions": [],
        "combined_remap": condition,
    }
    if profile.get("delayed_credit"):
        visible["pending_indirect_actions"] = [{"action_id": "d0", "region_id": indirect, "expected_signature": "stabilize_target", "issued_at": 0}]
        visible["outcome_history"] = [{"time": 8, "signature": "stabilize_target", "region_id": target, "source_action_id": "d0"}]
    if profile.get("candidate_search"):
        visible["exploration_history"] = [
            {"region_id": spurious, "effect_signature": "no_effect", "success": False},
            {"region_id": indirect, "effect_signature": "stabilize_target", "success": True},
        ]
    if profile.get("expected_mode") == "indirect" and not profile.get("delayed_credit") and not profile.get("candidate_search"):
        visible["trace_repair_region"] = None
        visible["trace_repair_confidence"] = 0.0
        visible["fallback_risk_region"] = None
        visible["fallback_risk_confidence"] = 0.0
        visible["inspection_observation_region"] = None
        visible["inspection_confidence"] = 0.0
        visible["transition_confidence"] = 0.0
        visible["feedback_confidence"] = 0.0
        visible["history_confidence"] = 0.0
    return visible


def public_mask(total: int, target: int, indirect: int) -> dict[int, dict[str, Any]]:
    return {
        region: {
            "region_id": region,
            "inspectable": True,
            "directly_intervenable": True,
            "unsafe": False,
            "irreversible": False,
            "cost": 0.0,
            "risk_cost": 0.0,
            "allowed_actions": ["inspect", "apply_local_damping", "indirect_stabilize"],
            "indirect_target_region": indirect if region == target else None,
        }
        for region in range(total)
    }


def hide_answer_like_mask(mask: dict[int, dict[str, Any]]) -> None:
    for info in mask.values():
        for key in ["unsafe", "irreversible", "cost", "risk_cost", "irreversibility_cost", "allowed_actions", "indirect_target_region"]:
            info.pop(key, None)
