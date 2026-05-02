from __future__ import annotations

from typing import Any


CONDITIONS = [
    "clean_reference",
    "visual_remap_hard",
    "risk_cue_remap_hard",
    "dynamics_remap_hard",
    "mask_visibility_remap_hard",
    "combined_remap_hard",
]


def make_b641_episode(config: dict[str, Any], seed: int, condition: str) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown B6.4.1 condition: {condition}")
    total = int(config.get("b6_4_1", {}).get("total_regions", 64))
    target = (seed * 7 + 13) % total
    decoy = (target + 17) % total
    trace_decoy = (target + 29) % total
    indirect = (target + 9) % total
    spurious_indirect = (target + 37) % total
    mask = public_mask(total, target, indirect)
    visible = {
        "public_state_available": True,
        "state_target_hint": target,
        "state_target_confidence": 0.86,
        "visual_channels": {"primary_region": target, "distractor_regions": []},
        "risk_encoding": "clean_low",
        "risk_score_estimate": 0.10,
        "history_supported_region": target,
        "history_confidence": 0.74,
        "feedback_history_region": target,
        "feedback_confidence": 0.72,
        "transition_consistency_region": target,
        "transition_confidence": 0.72,
        "pending_indirect_actions": [],
        "outcome_history": [],
        "exploration_history": [],
        "candidate_indirect_regions": [{"region_id": indirect, "causal_strength": 0.90, "backfire_estimate": 0.05}],
        "hard_remap": condition,
    }
    trace = {"region": target, "confidence": 0.90, "candidate_regions": [target]}
    expected = {"action_type": "apply_local_damping", "region_id": target}
    expected_inspect = False
    requires_indirect = False
    requires_credit = False

    if condition == "visual_remap_hard":
        visible["public_state_available"] = False
        visible["state_target_hint"] = decoy
        visible["state_target_confidence"] = 0.0
        visible["visual_channels"] = {"primary_region": decoy, "distractor_regions": [trace_decoy, (decoy + 5) % total]}
        trace = {"region": trace_decoy, "confidence": 0.82, "candidate_regions": [trace_decoy, target]}
        visible["history_supported_region"] = target
        visible["history_confidence"] = 0.88
    elif condition == "risk_cue_remap_hard":
        visible["public_state_available"] = True
        visible["state_target_hint"] = decoy
        visible["state_target_confidence"] = 0.78
        visible["risk_encoding"] = "inverted_high_means_safe"
        visible["risk_score_estimate"] = 0.90
        trace = {"region": trace_decoy, "confidence": 0.74, "candidate_regions": [trace_decoy, target]}
        visible["feedback_history_region"] = target
        visible["feedback_confidence"] = 0.90
        visible["history_supported_region"] = None
        visible["history_confidence"] = 0.0
    elif condition == "dynamics_remap_hard":
        visible["public_state_available"] = False
        visible["state_target_hint"] = decoy
        visible["state_target_confidence"] = 0.0
        visible["dynamics_profile"] = {"velocity_scale": 2.3, "friction": 0.12, "drift": -0.45, "collision_strength": 1.8}
        trace = {"region": trace_decoy, "confidence": 0.88, "candidate_regions": [trace_decoy, target]}
        visible["transition_consistency_region"] = target
        visible["transition_confidence"] = 0.90
        visible["history_supported_region"] = target
        visible["history_confidence"] = 0.78
    elif condition == "mask_visibility_remap_hard":
        expected = {"action_type": "indirect_stabilize", "region_id": indirect}
        requires_indirect = True
        hide_answer_like_mask(mask)
        visible["public_state_available"] = False
        visible["state_target_hint"] = decoy
        visible["state_target_confidence"] = 0.0
        trace = {"region": trace_decoy, "confidence": 0.52, "candidate_regions": [trace_decoy, target]}
        visible["candidate_indirect_regions"] = []
        visible["exploration_history"] = [
            {"region_id": indirect, "effect_signature": "stabilize_target", "success": True},
            {"region_id": spurious_indirect, "effect_signature": "no_effect", "success": False},
        ]
    elif condition == "combined_remap_hard":
        expected = {"action_type": "indirect_stabilize", "region_id": indirect}
        requires_indirect = True
        requires_credit = True
        hide_answer_like_mask(mask)
        visible["public_state_available"] = False
        visible["state_target_hint"] = decoy
        visible["state_target_confidence"] = 0.0
        visible["visual_channels"] = {"primary_region": decoy, "distractor_regions": [trace_decoy, spurious_indirect]}
        visible["risk_encoding"] = "rotated_hidden"
        visible["risk_score_estimate"] = 0.70
        visible["dynamics_profile"] = {"velocity_scale": 2.1, "friction": 0.15, "drift": 0.38, "collision_strength": 1.6}
        trace = {"region": trace_decoy, "confidence": 0.50, "candidate_regions": [trace_decoy, target]}
        visible["history_supported_region"] = target
        visible["history_confidence"] = 0.62
        visible["feedback_history_region"] = target
        visible["feedback_confidence"] = 0.60
        if seed % 2 == 0:
            visible["pending_indirect_actions"] = [
                {"action_id": "c0", "region_id": indirect, "expected_signature": "stabilize_target", "issued_at": 0}
            ]
            visible["outcome_history"] = [
                {"time": 7, "signature": "stabilize_target", "region_id": target, "source_action_id": "c0"}
            ]
            visible["exploration_history"] = [{"region_id": indirect, "effect_signature": "stabilize_target", "success": True}]
        else:
            visible["pending_indirect_actions"] = [
                {"action_id": "c0", "region_id": spurious_indirect, "expected_signature": "stabilize_target", "issued_at": 0}
            ]
            visible["outcome_history"] = [{"time": 7, "signature": "no_effect", "region_id": target, "source_action_id": "c0"}]
            visible["exploration_history"] = [{"region_id": spurious_indirect, "effect_signature": "no_effect", "success": False}]

    model_input = {
        "episode_id": seed,
        "visible_state": visible,
        "previous_trace_state": trace,
        "actionability_mask": mask,
    }
    evaluator = {
        "condition": condition,
        "target_region": target,
        "decoy_region": decoy,
        "trace_decoy_region": trace_decoy,
        "indirect_target_region": indirect,
        "expected_action": expected,
        "expected_inspect": expected_inspect,
        "requires_hidden_indirect": requires_indirect,
        "requires_delayed_credit": requires_credit,
    }
    return {
        "model_input": model_input,
        "evaluator_ground_truth": evaluator,
        "oracle_baseline_view": {"expected_action": expected, "expected_inspect": expected_inspect},
        "metadata": {"episode_id": seed, "condition": condition},
    }


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
            "irreversibility_cost": 0.0,
            "allowed_actions": ["inspect", "apply_local_damping", "indirect_stabilize"],
            "indirect_target_region": indirect if region == target else None,
        }
        for region in range(total)
    }


def hide_answer_like_mask(mask: dict[int, dict[str, Any]]) -> None:
    for info in mask.values():
        for key in ["unsafe", "irreversible", "cost", "risk_cost", "irreversibility_cost", "allowed_actions", "indirect_target_region"]:
            info.pop(key, None)
