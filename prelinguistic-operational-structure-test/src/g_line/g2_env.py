from __future__ import annotations

import math
import random
from typing import Any

CONDITIONS = ["train", "test", "ood_remap", "ood_novel_object", "ood_topology_shift"]


def clipped(value: float) -> float:
    return max(0.0, min(1.0, value))


def make_observed_interaction(
    rng: random.Random,
    condition: str,
    region: int,
    obj_type: str,
    obj_id: int,
    latent_causal: float,
    latent_risk: float,
    latent_indirect: float,
    latent_delay: float,
    cross_obj_effect: float,
    direct_good: bool,
    indirect_good: bool,
    inspect_good: bool,
) -> dict[str, Any]:
    noise_scale = 0.16 if condition == "ood_remap" else 0.04
    prediction_error = clipped(0.30 + 0.58 * latent_causal + rng.gauss(0, noise_scale))
    compression_surprise = clipped(0.42 * latent_causal + 0.48 * latent_delay + rng.gauss(0, noise_scale))
    intervention_gain = clipped(0.38 * latent_causal + 0.52 * latent_risk + rng.gauss(0, noise_scale))
    indirect_evidence = clipped(0.55 * latent_indirect + 0.35 * cross_obj_effect + rng.gauss(0, noise_scale))
    feedback_success = clipped(0.40 * latent_causal + 0.50 * cross_obj_effect + rng.gauss(0, noise_scale))
    risk_proxy = clipped(0.60 * latent_risk + 0.40 * latent_indirect + rng.gauss(0, noise_scale))
    delay_signal = clipped(0.45 * latent_delay + 0.45 * cross_obj_effect + rng.gauss(0, noise_scale))
    observed_direct_reward = clipped(0.85 if direct_good else 0.08 + rng.gauss(0, noise_scale))
    observed_indirect_reward = clipped(0.82 if indirect_good else 0.06 + rng.gauss(0, noise_scale))
    observed_inspect_value = clipped(0.88 if inspect_good else 0.10 + rng.gauss(0, noise_scale))
    return {
        "region_id": region,
        "object_id": obj_id,
        "object_type": obj_type,
        "prediction_error": prediction_error,
        "compression_surprise": compression_surprise,
        "intervention_gain": intervention_gain,
        "indirect_evidence": indirect_evidence,
        "feedback_success": feedback_success,
        "risk_proxy": risk_proxy,
        "delay_signal": delay_signal,
        "observed_direct_reward": observed_direct_reward,
        "observed_indirect_reward": observed_indirect_reward,
        "observed_inspect_value": observed_inspect_value,
    }


_OBJECT_LATENT_PROFILES = {
    "mechanical": {
        "causal_mean": 0.62, "causal_std": 0.18,
        "risk_mean": 0.38, "risk_std": 0.22,
        "indirect_mean": 0.45, "indirect_std": 0.25,
        "delay_mean": 0.52, "delay_std": 0.20,
    },
    "thermal": {
        "causal_mean": 0.48, "causal_std": 0.22,
        "risk_mean": 0.55, "risk_std": 0.20,
        "indirect_mean": 0.58, "indirect_std": 0.22,
        "delay_mean": 0.42, "delay_std": 0.25,
    },
    "diffusion": {
        "causal_mean": 0.55, "causal_std": 0.20,
        "risk_mean": 0.45, "risk_std": 0.24,
        "indirect_mean": 0.50, "indirect_std": 0.24,
        "delay_mean": 0.60, "delay_std": 0.18,
    },
    "advection": {
        "causal_mean": 0.50, "causal_std": 0.24,
        "risk_mean": 0.52, "risk_std": 0.22,
        "indirect_mean": 0.42, "indirect_std": 0.26,
        "delay_mean": 0.48, "delay_std": 0.22,
    },
    "collision": {
        "causal_mean": 0.58, "causal_std": 0.16,
        "risk_mean": 0.50, "risk_std": 0.18,
        "indirect_mean": 0.35, "indirect_std": 0.20,
        "delay_mean": 0.30, "delay_std": 0.15,
    },
}


def make_g2_episode(config: dict[str, Any], seed: int, condition: str) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown G2 condition: {condition}")
    rng = random.Random(seed)
    section = config.get("g2", {})
    n_objects = int(section.get("n_objects", 3))
    regions_per_object = int(section.get("regions_per_object", 4))

    if condition in ("ood_novel_object",):
        n_objects = int(section.get("ood_novel_n_objects", n_objects + 1))
    if condition in ("ood_topology_shift",):
        n_objects = int(section.get("ood_topology_n_objects", n_objects + 1))

    object_types_pool = sorted(_OBJECT_LATENT_PROFILES.keys())
    if condition in ("ood_novel_object",):
        chosen_types = rng.sample(object_types_pool, n_objects)
    elif condition in ("ood_topology_shift",):
        chosen_types = rng.sample(object_types_pool, n_objects)
    else:
        chosen_types = rng.sample(object_types_pool, n_objects)

    n_cross_edges = rng.randint(
        int(section.get("min_cross_edges", 1)),
        int(section.get("max_cross_edges", 3)),
    )
    cross_edges = []
    potential_pairs = [(i, j) for i in range(n_objects) for j in range(n_objects) if i != j]
    if potential_pairs:
        chosen_pairs = rng.sample(potential_pairs, min(n_cross_edges, len(potential_pairs)))
        for src_obj, dst_obj in chosen_pairs:
            cross_edges.append({
                "src_object": src_obj,
                "dst_object": dst_obj,
                "effect_strength": clipped(0.30 + 0.50 * rng.random()),
                "delay_hops": rng.randint(1, 3),
            })

    object_latents = {}
    for obj_id in range(n_objects):
        otype = chosen_types[obj_id]
        profile = _OBJECT_LATENT_PROFILES[otype]
        n_reg = regions_per_object
        obj_latents = {}
        for local_idx in range(n_reg):
            obj_latents[local_idx] = {
                "latent_causal": clipped(profile["causal_mean"] + profile["causal_std"] * rng.gauss(0, 1)),
                "latent_risk": clipped(profile["risk_mean"] + profile["risk_std"] * rng.gauss(0, 1)),
                "latent_indirect": clipped(profile["indirect_mean"] + profile["indirect_std"] * rng.gauss(0, 1)),
                "latent_delay": clipped(profile["delay_mean"] + profile["delay_std"] * rng.gauss(0, 1)),
            }
        object_latents[obj_id] = obj_latents

    global_offset = 0
    entries = []
    truth = []
    for obj_id in range(n_objects):
        otype = chosen_types[obj_id]
        for local_idx in range(regions_per_object):
            region = global_offset + local_idx
            lats = object_latents[obj_id][local_idx]
            latent_causal = lats["latent_causal"]
            latent_risk = lats["latent_risk"]
            latent_indirect = lats["latent_indirect"]
            latent_delay = lats["latent_delay"]

            cross_obj_effect = 0.0
            for edge in cross_edges:
                if edge["dst_object"] == obj_id:
                    src_obj = edge["src_object"]
                    src_avg_causal = sum(
                        object_latents[src_obj][li]["latent_causal"]
                        for li in range(regions_per_object)
                    ) / regions_per_object
                    cross_obj_effect += edge["effect_strength"] * src_avg_causal * (1.0 / (1.0 + edge["delay_hops"]))

            cross_obj_effect = clipped(cross_obj_effect)

            direct_good = latent_causal > 0.58 and latent_risk < 0.48
            indirect_good = (not direct_good) and latent_indirect > 0.60 and (latent_risk < 0.60 or cross_obj_effect > 0.35)
            inspect_good = (latent_causal > 0.50 or cross_obj_effect > 0.40) and latent_delay > 0.50
            no_action_safe = latent_risk > 0.72 and latent_causal < 0.60 and cross_obj_effect < 0.30

            interaction = make_observed_interaction(
                rng, condition, region, otype, obj_id,
                latent_causal, latent_risk, latent_indirect, latent_delay,
                cross_obj_effect, direct_good, indirect_good, inspect_good,
            )
            entries.append(interaction)
            truth.append({
                "region_id": region,
                "object_id": obj_id,
                "object_type": otype,
                "direct_actionable": direct_good,
                "indirect_actionable": indirect_good,
                "inspectable": inspect_good,
                "no_action_safe": no_action_safe,
                "latent_causal_strength": latent_causal,
                "latent_risk": latent_risk,
                "latent_indirect_strength": latent_indirect,
                "latent_delay": latent_delay,
                "cross_object_effect": cross_obj_effect,
            })
        global_offset += regions_per_object

    return {
        "episode_id": seed,
        "condition": condition,
        "n_objects": n_objects,
        "object_types": chosen_types,
        "cross_edges": cross_edges,
        "model_input": {
            "condition": condition,
            "interaction_history": entries,
            "n_regions": len(entries),
        },
        "evaluator_ground_truth": {"regions": truth, "objects": chosen_types, "edges": cross_edges},
    }


def make_g2_datasets(config: dict[str, Any], seed: int) -> dict[str, list[dict[str, Any]]]:
    section = config.get("g2", {})
    counts = {
        "train": int(section.get("train_episodes", 16)),
        "test": int(section.get("test_episodes", 8)),
        "ood_remap": int(section.get("ood_remap_episodes", 8)),
        "ood_novel_object": int(section.get("ood_novel_episodes", 8)),
        "ood_topology_shift": int(section.get("ood_topology_episodes", 8)),
    }
    base = 1000
    datasets = {}
    for condition in CONDITIONS:
        offset = 0
        if condition == "train": offset = base * 1
        elif condition == "test": offset = base * 2
        elif condition == "ood_remap": offset = base * 3
        elif condition == "ood_novel_object": offset = base * 4
        elif condition == "ood_topology_shift": offset = base * 5
        episodes = [make_g2_episode(config, offset + i, condition) for i in range(counts[condition])]
        datasets[condition] = episodes
    return datasets
