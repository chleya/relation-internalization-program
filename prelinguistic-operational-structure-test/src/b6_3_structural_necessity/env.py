from __future__ import annotations

from typing import Any

from src.b6_2_hardening.env import make_b62_episode

from .ablations import apply_ablation


CONDITIONS = [
    "clean_reference",
    "wrong_trace",
    "wrong_trace_state_ambiguous",
    "missing_trace",
    "ambiguous_trace",
    "low_confidence_trace",
    "missing_mask",
    "hard_hidden_mask",
    "hide_public_state_cue",
    "hide_indirect_target",
    "delayed_indirect_delay5",
    "risk_reward_conflict",
    "spurious_flip",
]


def make_b63_episode(config: dict[str, Any], seed: int, condition: str) -> dict[str, Any]:
    if condition == "clean_reference":
        episode = make_b62_episode(config, seed, "test")
    elif condition == "delayed_indirect_delay5":
        episode = make_b62_episode(config, seed, "delayed_indirect", delay_steps=5)
    elif condition == "hide_public_state_cue":
        episode = make_b62_episode(config, seed, "wrong_trace_state_ambiguous", mask_visibility="hard_hidden", trace_mode="wrong")
        episode = apply_ablation(episode, "hide_public_state_cue", config)
    else:
        kwargs = trace_kwargs_for_condition(condition)
        if condition == "hide_indirect_target":
            kwargs = {**kwargs, "mask_visibility": "hard_hidden", "hide_indirect_target": True}
        episode = make_b62_episode(config, seed, condition, **kwargs)
    episode["evaluator_ground_truth"]["condition"] = condition
    episode["metadata"]["condition"] = condition
    add_history_support(episode)
    return episode


def make_b63_datasets(config: dict[str, Any], seed: int = 0) -> dict[str, list[dict[str, Any]]]:
    b63 = config.get("b6_3", {})
    n = int(b63.get("episodes_per_condition", 10))
    requested = list(b63.get("conditions", CONDITIONS))
    return {
        condition: [make_b63_episode(config, seed + cidx * 1000 + idx, condition) for idx in range(n)]
        for cidx, condition in enumerate(requested)
    }


def add_history_support(episode: dict[str, Any]) -> None:
    visible = episode["model_input"].setdefault("visible_state", {})
    evaluator = episode["evaluator_ground_truth"]
    target = evaluator.get("target_region")
    if visible.get("history_supported_region") is None:
        visible["history_supported_region"] = target
    if visible.get("history_confidence") is None:
        visible["history_confidence"] = 0.50


def trace_kwargs_for_condition(condition: str) -> dict[str, Any]:
    if condition == "wrong_trace":
        return {"trace_mode": "wrong"}
    if condition == "wrong_trace_state_ambiguous":
        return {"trace_mode": "wrong", "mask_visibility": "hard_hidden"}
    if condition == "missing_trace":
        return {"trace_mode": "missing"}
    if condition == "ambiguous_trace":
        return {"trace_mode": "ambiguous"}
    if condition == "low_confidence_trace":
        return {"trace_mode": "low_confidence"}
    if condition == "hard_hidden_mask":
        return {"mask_visibility": "hard_hidden"}
    return {}
