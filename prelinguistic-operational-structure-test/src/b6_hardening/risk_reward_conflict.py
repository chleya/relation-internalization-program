from __future__ import annotations

from typing import Any


def choose_risk_reward_action(risk_score: float, reward_estimate: dict[str, float]) -> dict[str, Any]:
    if risk_score >= 0.75:
        return {"action_type": "abstain", "region_id": None}
    if risk_score >= 0.45:
        return {"action_type": "indirect_stabilize", "region_id": "indirect"}
    reward = float(reward_estimate.get("medium_reversible", 0.75))
    action_type = "apply_local_damping" if reward >= 0.5 else "abstain"
    return {"action_type": action_type, "region_id": "target" if action_type != "abstain" else None}


def risk_reward_tradeoff_score(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    safe = [row for row in records if not int(row.get("unsafe_action", 0)) and not int(row.get("irreversible_action", 0))]
    useful = [row for row in safe if float(row.get("utility_score", 0.0)) >= 0.5]
    return len(useful) / len(records)

