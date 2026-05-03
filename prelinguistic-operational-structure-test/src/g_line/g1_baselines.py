from __future__ import annotations

import random
from typing import Any


BASELINE_NAMES = ["g1_generator", "random", "always_abstain", "hand_designed", "oracle"]


def baseline_action(model_input: dict[str, Any], name: str, seed: int = 0, truth: dict[str, Any] | None = None) -> dict[str, Any]:
    if name == "always_abstain":
        return {"action": None, "generated_mask": {}, "provenance": {"uses_evaluator_labels": False}}
    if name == "random":
        rng = random.Random(seed)
        history = list(model_input.get("interaction_history", []))
        if not history or rng.random() < 0.25:
            return {"action": None, "generated_mask": {}, "provenance": {"uses_evaluator_labels": False}}
        item = history[rng.randrange(len(history))]
        action_type = "apply_local_damping" if rng.random() < 0.6 else "indirect_stabilize"
        return {"action": {"action_type": action_type, "region_id": int(item["region_id"])}, "generated_mask": {}, "provenance": {"uses_evaluator_labels": False}}
    if name == "hand_designed":
        best = None
        best_score = -1.0
        for item in model_input.get("interaction_history", []):
            score = float(item["prediction_error"]) - float(item["risk_proxy"])
            if score > best_score:
                best, best_score = item, score
        action = None if best is None or best_score < 0.12 else {"action_type": "apply_local_damping", "region_id": int(best["region_id"])}
        return {"action": action, "generated_mask": {}, "provenance": {"uses_evaluator_labels": False}}
    if name == "oracle" and truth is not None:
        mask = {
            int(region["region_id"]): {
                "region_id": int(region["region_id"]),
                "directly_intervenable": bool(region["direct_actionable"]),
                "indirectly_intervenable": bool(region["indirect_actionable"]),
                "inspectable": bool(region["inspectable"]),
                "update_weight": 1.0,
                "generated_trace_score": 1.0,
                "generated_by": "oracle_evaluator_baseline",
            }
            for region in truth["regions"]
        }
        for region in truth["regions"]:
            if region["direct_actionable"]:
                return {"action": {"action_type": "apply_local_damping", "region_id": int(region["region_id"])}, "generated_mask": mask, "provenance": {"uses_evaluator_labels": True}}
        for region in truth["regions"]:
            if region["indirect_actionable"]:
                return {"action": {"action_type": "indirect_stabilize", "region_id": int(region["region_id"])}, "generated_mask": mask, "provenance": {"uses_evaluator_labels": True}}
        return {"action": None, "generated_mask": mask, "provenance": {"uses_evaluator_labels": True}}
    raise KeyError(name)
