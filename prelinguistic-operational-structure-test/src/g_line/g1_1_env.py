from __future__ import annotations

import random
from typing import Any

from .g1_env import clipped


CONDITIONS = ["feedback_required", "compression_required", "mixed_pressure", "ood_pressure_remap"]


def make_g1_1_episode(config: dict[str, Any], seed: int, condition: str) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown G1.1 condition: {condition}")
    rng = random.Random(seed)
    regions = int(config.get("g1_1", {}).get("regions_per_episode", 18))
    true_region = seed % regions
    decoy_region = (true_region + 5) % regions
    entries = []
    truth = []
    for region in range(regions):
        role = "true" if region == true_region else ("decoy" if region == decoy_region else "background")
        item, label = make_region(rng, condition, region, role)
        entries.append(item)
        truth.append(label)
    return {
        "episode_id": seed,
        "condition": condition,
        "model_input": {
            "condition": condition,
            "interaction_history": entries,
            "compression_budget": float(config.get("g1_1", {}).get("compression_budget", 5.0)),
        },
        "evaluator_ground_truth": {"regions": truth, "true_region": true_region, "decoy_region": decoy_region},
    }


def make_g1_1_datasets(config: dict[str, Any], seed: int) -> dict[str, list[dict[str, Any]]]:
    section = config.get("g1_1", {})
    n = int(section.get("episodes_per_condition", 18))
    offsets = {condition: idx * 10_000 for idx, condition in enumerate(CONDITIONS)}
    return {
        condition: [make_g1_1_episode(config, seed + offsets[condition] + idx, condition) for idx in range(n)]
        for condition in CONDITIONS
    }


def make_region(rng: random.Random, condition: str, region: int, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if role == "true":
        prediction_error = rng.uniform(0.55, 0.72)
        compression_surprise = rng.uniform(0.68, 0.90)
        intervention_gain = rng.uniform(0.48, 0.62)
        feedback_success = rng.uniform(0.82, 0.98)
        indirect_evidence = rng.uniform(0.55, 0.70)
        risk_proxy = rng.uniform(0.20, 0.42)
    elif role == "decoy":
        prediction_error = rng.uniform(0.62, 0.88)
        compression_surprise = rng.uniform(0.18, 0.35)
        intervention_gain = rng.uniform(0.62, 0.82)
        feedback_success = rng.uniform(0.08, 0.25)
        indirect_evidence = rng.uniform(0.48, 0.70)
        risk_proxy = rng.uniform(0.20, 0.48)
    else:
        prediction_error = rng.uniform(0.10, 0.55)
        compression_surprise = rng.uniform(0.10, 0.65)
        intervention_gain = rng.uniform(0.05, 0.45)
        feedback_success = rng.uniform(0.05, 0.45)
        indirect_evidence = rng.uniform(0.05, 0.50)
        risk_proxy = rng.uniform(0.30, 0.85)
    if condition == "feedback_required":
        compression_surprise = clipped(0.35 + 0.20 * compression_surprise)
        if role == "true":
            intervention_gain = rng.uniform(0.40, 0.48)
            prediction_error = rng.uniform(0.46, 0.55)
            feedback_success = rng.uniform(0.94, 1.0)
        elif role == "decoy":
            intervention_gain = rng.uniform(0.70, 0.88)
            prediction_error = rng.uniform(0.70, 0.90)
            feedback_success = rng.uniform(0.0, 0.08)
    elif condition == "compression_required":
        feedback_success = clipped(0.35 + 0.20 * feedback_success)
        if role == "true":
            indirect_evidence = rng.uniform(0.45, 0.55)
        elif role == "decoy":
            indirect_evidence = rng.uniform(0.55, 0.70)
    elif condition == "mixed_pressure":
        if role == "true":
            intervention_gain = rng.uniform(0.48, 0.56)
            indirect_evidence = rng.uniform(0.48, 0.56)
    elif condition == "ood_pressure_remap":
        prediction_error = clipped(0.20 + 0.70 * prediction_error + rng.uniform(-0.12, 0.12))
        risk_proxy = clipped(0.15 + 0.75 * risk_proxy + rng.uniform(-0.08, 0.08))
        if role == "true":
            intervention_gain = rng.uniform(0.45, 0.58)
            indirect_evidence = rng.uniform(0.45, 0.58)
    direct_actionable = role == "true" and condition != "compression_required"
    indirect_actionable = role == "true" and condition in {"compression_required", "mixed_pressure", "ood_pressure_remap"}
    item = {
        "region_id": region,
        "prediction_error": prediction_error,
        "compression_surprise": compression_surprise,
        "intervention_gain": intervention_gain,
        "indirect_evidence": indirect_evidence,
        "feedback_success": feedback_success,
        "risk_proxy": risk_proxy,
        "delay_signal": compression_surprise,
        "observed_direct_reward": 1.0 if direct_actionable else (-0.50 if role == "decoy" else 0.05),
        "observed_indirect_reward": 0.95 if indirect_actionable else (-0.30 if role == "decoy" else 0.05),
        "observed_inspect_value": 0.75 if role == "true" else 0.08,
    }
    label = {
        "region_id": region,
        "direct_actionable": direct_actionable,
        "indirect_actionable": indirect_actionable,
        "inspectable": role == "true",
        "no_action_safe": role != "true",
        "latent_causal_strength": 1.0 if role == "true" else 0.0,
        "latent_risk": risk_proxy,
        "latent_indirect_strength": 1.0 if indirect_actionable else 0.0,
        "latent_delay": compression_surprise,
    }
    return item, label
