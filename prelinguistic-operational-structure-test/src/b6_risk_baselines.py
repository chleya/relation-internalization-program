from __future__ import annotations

from typing import Any

import numpy as np

from .b6_risk_env import compute_risk_constrained_outcome


def random_risk_baseline(episode_bundle: dict[str, Any], config: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    target = int(episode_bundle["metadata"]["target_region"])
    action_type = ["abstain", "apply_local_damping", "indirect_stabilize"][int(rng.integers(0, 3))]
    action = None if action_type == "abstain" else {"action_type": action_type, "region_id": target, "strength": 1.0}
    return baseline_output(action, target, "random")


def saliency_risk_baseline(episode_bundle: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    region = int(episode_bundle["model_input"]["visible_state"]["saliency_region"])
    return baseline_output({"action_type": "apply_local_damping", "region_id": region, "strength": 1.0}, region, "saliency")


def short_horizon_risk_baseline(episode_bundle: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    region = int(episode_bundle["model_input"]["visible_state"]["short_horizon_region"])
    return baseline_output({"action_type": "apply_local_damping", "region_id": region, "strength": 1.0}, region, "short_horizon")


def risk_blind_trace_baseline(model: Any, episode_bundle: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    target = int(episode_bundle["metadata"]["target_region"])
    return baseline_output({"action_type": "apply_local_damping", "region_id": target, "strength": 1.0}, target, "risk_blind")


def always_act_baseline(episode_bundle: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    target = int(episode_bundle["metadata"]["target_region"])
    return baseline_output({"action_type": "apply_local_damping", "region_id": target, "strength": 1.0}, target, "always_act")


def always_abstain_baseline(episode_bundle: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    target = int(episode_bundle["metadata"]["target_region"])
    return baseline_output(None, target, "always_abstain")


def oracle_risk_constrained_baseline(episode_bundle: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    plan = episode_bundle["oracle_baseline_view"]["oracle_risk_constrained_plan"]
    output = baseline_output(plan.get("action"), int(episode_bundle["metadata"]["target_region"]), "oracle")
    output["inspect_decision"]["inspect_region"] = int(episode_bundle["metadata"]["target_region"]) if plan.get("inspect") else None
    return output


def evaluate_b6_baselines(model: Any, episode_bundles: list[dict[str, Any]], config: dict[str, Any], seed: int = 0) -> tuple[dict[str, float], list[dict[str, Any]]]:
    scores = {name: [] for name in ["random", "saliency", "short_horizon", "risk_blind", "always_act", "always_abstain", "oracle"]}
    records = []
    for idx, bundle in enumerate(episode_bundles):
        outputs = {
            "random": random_risk_baseline(bundle, config, seed + idx),
            "saliency": saliency_risk_baseline(bundle, config),
            "short_horizon": short_horizon_risk_baseline(bundle, config),
            "risk_blind": risk_blind_trace_baseline(model, bundle, config),
            "always_act": always_act_baseline(bundle, config),
            "always_abstain": always_abstain_baseline(bundle, config),
            "oracle": oracle_risk_constrained_baseline(bundle, config),
        }
        for name, output in outputs.items():
            score = compute_risk_constrained_outcome(bundle, output, config)["risk_constrained_score"]
            scores[name].append(float(score))
            records.append(
                {
                    "episode_id": int(bundle["metadata"]["episode_id"]),
                    "episode_type": bundle["evaluator_ground_truth"]["b6_episode_type"],
                    "baseline_name": name,
                    "baseline_score": float(score),
                    "gate_pass": int(score > 0.0),
                    "note": "B6 baseline comparison",
                }
            )
    return {
        "random_risk_constrained_score": mean(scores["random"]),
        "saliency_risk_constrained_score": mean(scores["saliency"]),
        "short_horizon_risk_constrained_score": mean(scores["short_horizon"]),
        "risk_blind_score": mean(scores["risk_blind"]),
        "always_act_score": mean(scores["always_act"]),
        "always_abstain_score": mean(scores["always_abstain"]),
        "oracle_risk_constrained_score": mean(scores["oracle"]),
    }, records


def baseline_output(action: dict[str, Any] | None, target: int, source: str) -> dict[str, Any]:
    return {
        "inspect_decision": {"inspect_region": None},
        "intervention_decision": {"action": action},
        "abstain_decision": {"abstained": action is None, "reason": "baseline_abstain" if action is None else None},
        "provenance": {"oracle_value_used": False, "oracle_actionability_used": source == "oracle"},
    }


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
