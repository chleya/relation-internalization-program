from __future__ import annotations

from typing import Any

from .remap_configs import CONDITIONS, REMAP_TYPES


def make_b64_episode(config: dict[str, Any], seed: int, condition: str) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown B6.4 condition: {condition}")
    total = int(config.get("b6_4", {}).get("total_regions", 64))
    target = (seed * 5 + 17) % total
    decoy = (target + 23) % total
    trace_region = target
    indirect = (target + 11) % total
    remapped_indirect = (target + 31) % total
    mask = make_public_mask(total, target, indirect)
    visible = {
        "public_state_available": True,
        "state_target_hint": target,
        "state_target_confidence": 0.86,
        "visual_code_region": target,
        "visual_code_confidence": 0.86,
        "history_supported_region": target,
        "history_confidence": 0.70,
        "feedback_history_region": target,
        "feedback_confidence": 0.66,
        "risk_channel": "standard_low",
        "risk_score_estimate": 0.10,
        "risk_history_score": 0.10,
        "delay_observations": [],
        "pending_indirect_actions": [],
        "outcome_history": [],
        "candidate_indirect_regions": [{"region_id": indirect, "causal_strength": 0.88, "backfire_estimate": 0.10}],
        "indirect_history_paths": [],
        "exploration_history": [],
        "remap_signature": REMAP_TYPES[condition],
    }
    evaluator = {
        "condition": condition,
        "target_region": target,
        "decoy_region": decoy,
        "indirect_target_region": indirect,
        "expected_action": {"action_type": "apply_local_damping", "region_id": target},
        "expected_inspect": False,
        "remap_type": REMAP_TYPES[condition],
        "requires_delayed_credit": False,
        "requires_hidden_indirect": False,
    }

    if condition == "visual_remap":
        visible["state_target_hint"] = decoy
        visible["state_target_confidence"] = 0.32
        visible["visual_code_region"] = decoy
        visible["visual_code_confidence"] = 0.25
        visible["history_supported_region"] = target
        visible["history_confidence"] = 0.82
        trace_region = target
    elif condition == "risk_cue_remap":
        visible["risk_channel"] = "remapped_symbolic_low"
        visible["risk_score_estimate"] = 0.80
        visible["risk_history_score"] = 0.12
        visible["feedback_history_region"] = target
        visible["feedback_confidence"] = 0.78
    elif condition == "dynamics_remap":
        visible["dynamics_profile"] = {"velocity_scale": 1.7, "drift": -0.2, "friction": 0.35}
        visible["state_target_hint"] = decoy
        visible["state_target_confidence"] = 0.40
        visible["history_supported_region"] = target
        visible["history_confidence"] = 0.80
        visible["feedback_history_region"] = target
        visible["feedback_confidence"] = 0.74
    elif condition == "delay_profile_remap":
        expected = {"action_type": "indirect_stabilize", "region_id": indirect}
        evaluator["expected_action"] = expected
        evaluator["requires_delayed_credit"] = True
        visible["pending_indirect_actions"] = [
            {"action_id": "a0", "region_id": indirect, "expected_signature": "stabilize_target", "issued_at": 0}
        ]
        visible["outcome_history"] = [
            {"time": 7, "signature": "stabilize_target", "region_id": target, "source_action_id": "a0"}
        ]
        visible["delay_observations"] = [3, 5, 7]
    elif condition == "indirect_path_remap":
        evaluator["expected_action"] = {"action_type": "indirect_stabilize", "region_id": remapped_indirect}
        evaluator["indirect_target_region"] = remapped_indirect
        evaluator["requires_hidden_indirect"] = True
        mask[target]["indirect_target_region"] = None
        visible["candidate_indirect_regions"] = [
            {"region_id": decoy, "causal_strength": 0.45, "backfire_estimate": 0.30}
        ]
        visible["exploration_history"] = [
            {"region_id": remapped_indirect, "effect_signature": "stabilize_target", "success": True},
            {"region_id": decoy, "effect_signature": "no_effect", "success": False},
        ]
    elif condition == "mask_visibility_remap":
        for info in mask.values():
            info.pop("unsafe", None)
            info.pop("risk_cost", None)
            info["indirect_target_region"] = None
        visible["history_supported_region"] = target
        visible["history_confidence"] = 0.84
    elif condition == "combined_remap":
        evaluator["expected_action"] = {"action_type": "indirect_stabilize", "region_id": remapped_indirect}
        evaluator["indirect_target_region"] = remapped_indirect
        evaluator["requires_delayed_credit"] = True
        evaluator["requires_hidden_indirect"] = True
        trace_region = decoy
        visible["public_state_available"] = False
        visible["state_target_hint"] = decoy
        visible["state_target_confidence"] = 0.0
        visible["visual_code_region"] = decoy
        visible["visual_code_confidence"] = 0.20
        visible["risk_channel"] = "remapped_symbolic_low"
        visible["risk_score_estimate"] = 0.75
        visible["history_supported_region"] = target
        visible["history_confidence"] = 0.66
        visible["feedback_history_region"] = target
        visible["feedback_confidence"] = 0.68
        visible["pending_indirect_actions"] = [
            {"action_id": "c0", "region_id": remapped_indirect, "expected_signature": "stabilize_target", "issued_at": 0}
        ]
        visible["outcome_history"] = [
            {"time": 5, "signature": "stabilize_target", "region_id": target, "source_action_id": "c0"}
        ]
        visible["exploration_history"] = [
            {"region_id": remapped_indirect, "effect_signature": "stabilize_target", "success": True}
        ]
        visible["candidate_indirect_regions"] = []
        for info in mask.values():
            info.pop("unsafe", None)
            info.pop("risk_cost", None)
            info["indirect_target_region"] = None

    model_input = {
        "episode_id": seed,
        "visible_state": visible,
        "previous_trace_state": {
            "region": trace_region,
            "confidence": 0.88 if trace_region == target else 0.45,
            "candidate_regions": [trace_region, target],
        },
        "actionability_mask": mask,
    }
    return {
        "model_input": model_input,
        "evaluator_ground_truth": evaluator,
        "oracle_baseline_view": {"expected_action": evaluator["expected_action"], "expected_inspect": evaluator["expected_inspect"]},
        "metadata": {"episode_id": seed, "condition": condition},
    }


def make_public_mask(total: int, target: int, indirect: int) -> dict[int, dict[str, Any]]:
    mask: dict[int, dict[str, Any]] = {}
    for region in range(total):
        mask[region] = {
            "region_id": region,
            "inspectable": True,
            "directly_intervenable": True,
            "unsafe": False,
            "risk_cost": 0.0,
            "indirect_target_region": None,
        }
    mask[target]["indirect_target_region"] = indirect
    return mask
