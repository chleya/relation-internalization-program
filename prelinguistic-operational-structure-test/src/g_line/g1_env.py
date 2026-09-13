from __future__ import annotations

import math
import random
from typing import Any


CONDITIONS = ["train", "test", "ood_remap"]


def make_g1_episode(config: dict[str, Any], seed: int, condition: str) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown G1 condition: {condition}")
    rng = random.Random(seed)
    section = config.get("g1", {})
    regions = int(section.get("regions_per_episode", 24))
    entries = []
    truth = []
    for region in range(regions):
        latent_causal = clipped(0.15 + 0.75 * rng.random() + 0.10 * math.sin((seed + region) * 0.7))
        latent_risk = clipped(0.10 + 0.80 * rng.random())
        latent_indirect = clipped(0.20 + 0.70 * rng.random())
        latent_delay = clipped(0.15 + 0.75 * rng.random())
        direct_good = latent_causal > 0.58 and latent_risk < 0.46
        indirect_good = (not direct_good) and latent_indirect > 0.62 and latent_risk < 0.58
        inspect_good = latent_causal > 0.50 and latent_delay > 0.55
        no_action_safe = latent_risk > 0.70 and latent_causal < 0.62
        interaction = make_observed_interaction(
            rng,
            condition,
            region,
            latent_causal,
            latent_risk,
            latent_indirect,
            latent_delay,
            direct_good,
            indirect_good,
            inspect_good,
        )
        entries.append(interaction)
        truth.append(
            {
                "region_id": region,
                "direct_actionable": direct_good,
                "indirect_actionable": indirect_good,
                "inspectable": inspect_good,
                "no_action_safe": no_action_safe,
                "latent_causal_strength": latent_causal,
                "latent_risk": latent_risk,
                "latent_indirect_strength": latent_indirect,
                "latent_delay": latent_delay,
            }
        )
    return {
        "episode_id": seed,
        "condition": condition,
        "model_input": {
            "condition": condition,
            "interaction_history": entries,
            "compression_budget": float(section.get("compression_budget", 4.0)),
        },
        "evaluator_ground_truth": {"regions": truth},
    }


def make_g1_datasets(config: dict[str, Any], seed: int) -> dict[str, list[dict[str, Any]]]:
    section = config.get("g1", {})
    counts = {
        "train": int(section.get("train_episodes", 32)),
        "test": int(section.get("test_episodes", 16)),
        "ood_remap": int(section.get("ood_episodes", 16)),
    }
    offsets = {"train": 0, "test": 10_000, "ood_remap": 20_000}
    return {
        condition: [make_g1_episode(config, seed + offsets[condition] + idx, condition) for idx in range(count)]
        for condition, count in counts.items()
    }


def make_observed_interaction(
    rng: random.Random,
    condition: str,
    region: int,
    latent_causal: float,
    latent_risk: float,
    latent_indirect: float,
    latent_delay: float,
    direct_good: bool,
    indirect_good: bool,
    inspect_good: bool,
) -> dict[str, Any]:
    noise = 0.08 if condition != "ood_remap" else 0.16
    prediction_error = clipped(0.15 + 0.70 * latent_causal + rng.uniform(-noise, noise))
    compression_surprise = clipped(0.20 + 0.55 * latent_delay + rng.uniform(-noise, noise))
    intervention_gain = clipped((0.78 if direct_good else 0.25 + 0.20 * latent_causal) + rng.uniform(-noise, noise))
    indirect_evidence = clipped((0.80 if indirect_good else 0.18 + 0.25 * latent_indirect) + rng.uniform(-noise, noise))
    feedback_success = clipped((0.74 if direct_good or indirect_good else 0.20) + rng.uniform(-noise, noise))
    risk_proxy = clipped(latent_risk + rng.uniform(-noise, noise))
    if condition == "ood_remap":
        prediction_error = clipped(0.30 + 0.58 * latent_causal + rng.uniform(-noise, noise))
        risk_proxy = clipped(0.18 + 0.72 * latent_risk + rng.uniform(-noise, noise))
        compression_surprise = clipped(0.30 + 0.50 * latent_delay + rng.uniform(-noise, noise))
    return {
        "region_id": region,
        "prediction_error": prediction_error,
        "compression_surprise": compression_surprise,
        "intervention_gain": intervention_gain,
        "indirect_evidence": indirect_evidence,
        "feedback_success": feedback_success,
        "risk_proxy": risk_proxy,
        "delay_signal": clipped(latent_delay + rng.uniform(-noise, noise)),
        "observed_direct_reward": 1.0 if direct_good else (-0.40 if latent_risk > 0.70 else 0.15),
        "observed_indirect_reward": 0.85 if indirect_good else 0.05,
        "observed_inspect_value": 0.70 if inspect_good else 0.10,
    }


def clipped(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
