from __future__ import annotations

import copy
from typing import Any


ABLATIONS = [
    "reference",
    "remove_trace",
    "shuffle_trace",
    "corrupt_trace",
    "freeze_feedback_update",
    "remove_history",
    "remove_risk_cue",
    "hide_public_state_cue",
    "disable_candidate_search",
    "disable_delayed_credit_buffer",
    "disable_inspection_recovery",
]


EXPECTED_AFFECTED_SPLITS = {
    "remove_trace": ["wrong_trace", "missing_trace", "ambiguous_trace", "low_confidence_trace", "wrong_trace_state_ambiguous"],
    "shuffle_trace": ["wrong_trace", "wrong_trace_state_ambiguous"],
    "corrupt_trace": ["wrong_trace", "wrong_trace_state_ambiguous"],
    "freeze_feedback_update": ["wrong_trace_state_ambiguous", "delayed_indirect_delay5"],
    "remove_history": ["delayed_indirect_delay5", "wrong_trace_state_ambiguous"],
    "remove_risk_cue": ["missing_mask", "hard_hidden_mask", "risk_reward_conflict"],
    "hide_public_state_cue": ["wrong_trace", "wrong_trace_state_ambiguous", "missing_trace"],
    "disable_candidate_search": ["hide_indirect_target"],
    "disable_delayed_credit_buffer": ["delayed_indirect_delay5"],
    "disable_inspection_recovery": ["ambiguous_trace", "low_confidence_trace", "wrong_trace_state_ambiguous"],
    "reference": [],
}


def apply_ablation(episode: dict[str, Any], ablation_name: str, config: dict[str, Any]) -> dict[str, Any]:
    ablated = copy.deepcopy(episode)
    model_input = ablated["model_input"]
    visible = model_input.setdefault("visible_state", {})
    flags = model_input.setdefault("ablation_flags", {})
    flags["ablation_name"] = ablation_name

    if ablation_name == "reference":
        return ablated
    if ablation_name == "remove_trace":
        model_input["previous_trace_state"] = {"region": None, "confidence": 0.0, "candidate_regions": [], "trace_mode": "removed"}
    elif ablation_name == "shuffle_trace":
        trace = dict(model_input.get("previous_trace_state", {}))
        region = trace.get("region")
        if region is not None:
            trace["region"] = (int(region) + 19) % 64
        trace["confidence"] = float(trace.get("confidence", 0.9))
        trace["trace_mode"] = "shuffled"
        model_input["previous_trace_state"] = trace
    elif ablation_name == "corrupt_trace":
        wrong_region = ablated["evaluator_ground_truth"].get("wrong_target_region")
        model_input["previous_trace_state"] = {
            "region": wrong_region,
            "confidence": 0.95,
            "candidate_regions": [wrong_region],
            "trace_mode": "corrupt",
        }
    elif ablation_name == "freeze_feedback_update":
        flags["feedback_update_disabled"] = True
        visible["feedback_history_region"] = None
        visible["feedback_confidence"] = 0.0
    elif ablation_name == "remove_history":
        visible["outcome_history"] = []
        visible["feedback_history_region"] = None
        visible["feedback_confidence"] = 0.0
        flags["history_removed"] = True
    elif ablation_name == "remove_risk_cue":
        visible["risk_history_score"] = 0.0
        visible["latent_risk_marker"] = "unknown"
        for row in visible.get("candidate_indirect_regions", []):
            row["backfire_estimate"] = 0.50
        for row in visible.get("safe_alternative_candidates", []):
            row["risk_estimate"] = 0.50
        flags["risk_cue_removed"] = True
    elif ablation_name == "hide_public_state_cue":
        visible["state_target_hint"] = (int(visible.get("state_target_hint", 0)) + 11) % 64
        visible["state_target_confidence"] = 0.0
        visible["state_target_candidates"] = []
        flags["public_state_cue_hidden"] = True
    elif ablation_name == "disable_candidate_search":
        visible["candidate_indirect_regions"] = []
        flags["candidate_search_disabled"] = True
        mask = model_input.get("actionability_mask")
        if mask:
            for info in mask.values():
                info.pop("indirect_target_region", None)
    elif ablation_name == "disable_delayed_credit_buffer":
        visible["outcome_history"] = []
        flags["delayed_credit_buffer_disabled"] = True
    elif ablation_name == "disable_inspection_recovery":
        flags["inspection_recovery_disabled"] = True
    else:
        raise ValueError(f"unknown B6.3 ablation: {ablation_name}")
    return ablated
