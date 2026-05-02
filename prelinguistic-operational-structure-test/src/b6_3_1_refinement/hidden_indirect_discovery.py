from __future__ import annotations

from typing import Any


HIDDEN_INDIRECT_CONDITIONS = [
    "hidden_indirect_no_candidate_shortcut",
    "hidden_indirect_exploration_required",
    "hidden_indirect_spurious_candidate",
    "hidden_indirect_history_discovery",
]


def apply_hidden_indirect_condition(episode: dict[str, Any], condition: str) -> None:
    visible = episode["model_input"]["visible_state"]
    evaluator = episode["evaluator_ground_truth"]
    target = int(evaluator["target_region"])
    indirect = int(evaluator["indirect_target_region"])
    spurious = (indirect + 17) % 64
    visible["public_state_available"] = False
    visible["state_target_hint"] = int(evaluator["state_decoy_region"])
    visible["state_target_confidence"] = 0.0
    visible["history_supported_region"] = target
    visible["history_confidence"] = 0.78
    visible["risk_history_score"] = 0.84
    visible["candidate_indirect_regions"] = []
    visible["indirect_history_paths"] = []
    visible["exploration_history"] = []
    mask = episode["model_input"].get("actionability_mask") or {}
    for info in mask.values():
        info.pop("indirect_target_region", None)
    evaluator["expected_action"] = {"action_type": "indirect_stabilize", "region_id": indirect}
    evaluator["expected_inspect"] = False
    evaluator["hidden_indirect_target"] = True

    if condition == "hidden_indirect_no_candidate_shortcut":
        visible["exploration_history"] = [{"region_id": indirect, "effect_signature": "stabilize_target", "success": True}]
    elif condition == "hidden_indirect_exploration_required":
        visible["exploration_history"] = [
            {"region_id": spurious, "effect_signature": "weak_effect", "success": False},
            {"region_id": indirect, "effect_signature": "stabilize_target", "success": True},
        ]
    elif condition == "hidden_indirect_spurious_candidate":
        visible["candidate_indirect_regions"] = [{"region_id": spurious, "causal_strength": 0.95, "backfire_estimate": 0.05}]
        visible["exploration_history"] = [
            {"region_id": spurious, "effect_signature": "no_effect", "success": False},
            {"region_id": indirect, "effect_signature": "stabilize_target", "success": True},
        ]
    elif condition == "hidden_indirect_history_discovery":
        visible["indirect_history_paths"] = [{"region_id": indirect, "causal_strength": 0.89, "backfire_estimate": 0.08}]


def discover_hidden_indirect(model_input: dict[str, Any], disabled: set[str] | None = None) -> dict[str, Any]:
    disabled = disabled or set()
    if "candidate_search" in disabled:
        return {"action": None, "source": "candidate_search_disabled", "success": False, "spurious_rejected": False}
    visible = model_input.get("visible_state", {})
    if "history" not in disabled:
        for row in visible.get("exploration_history", []):
            if row.get("success") and row.get("effect_signature") == "stabilize_target":
                return {"action": {"action_type": "indirect_stabilize", "region_id": int(row["region_id"])}, "source": "exploration_history", "success": True, "spurious_rejected": True}
        for row in visible.get("indirect_history_paths", []):
            if float(row.get("causal_strength", 0.0)) >= 0.70 and float(row.get("backfire_estimate", 1.0)) <= 0.30:
                return {"action": {"action_type": "indirect_stabilize", "region_id": int(row["region_id"])}, "source": "history_path", "success": True, "spurious_rejected": True}
    for row in visible.get("candidate_indirect_regions", []):
        if float(row.get("causal_strength", 0.0)) >= 0.70 and float(row.get("backfire_estimate", 1.0)) <= 0.30:
            return {"action": {"action_type": "indirect_stabilize", "region_id": int(row["region_id"])}, "source": "candidate_search", "success": False, "spurious_rejected": False}
    return {"action": None, "source": "no_candidate", "success": False, "spurious_rejected": False}
