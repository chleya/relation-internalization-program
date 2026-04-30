from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .b2_delayed_env import make_delayed_checkpoint_episode
from .b23_selector_validation import b23_runtime_config
from .features import region_id_to_slice


def make_b3_active_inspection_episode(config: dict[str, Any], seed: int, episode_type: str, delay: int) -> dict[str, Any]:
    episode = make_delayed_checkpoint_episode(config, seed, delay, saliency_decoy=True)
    values = compute_oracle_inspection_values(episode, config)
    true_region = int(episode["ground_truth"]["true_delayed_checkpoint_region"])
    saliency_region = int(episode["ground_truth"]["early_saliency_region"])
    episode["ground_truth"].update(
        {
            "episode_type": str(episode_type),
            "true_trace_region": true_region,
            "saliency_region": saliency_region,
            "oracle_best_inspect_region": max(values, key=values.get),
            "inspection_values": {int(region): float(value) for region, value in values.items()},
            "delay": int(delay),
        }
    )
    return episode


def compute_oracle_inspection_values(episode: dict[str, Any], config: dict[str, Any]) -> dict[int, float]:
    grid_size = int(config.get("env", {}).get("grid_size", 8))
    n_regions = grid_size * grid_size
    true_region = int(episode["ground_truth"].get("true_trace_region", episode["ground_truth"]["true_delayed_checkpoint_region"]))
    saliency_region = int(episode["ground_truth"].get("saliency_region", episode["ground_truth"].get("early_saliency_region", -1)))
    values = {region: 0.0 for region in range(n_regions)}
    values[true_region] = 1.0
    for region in neighbor_regions(true_region, grid_size):
        values[region] = max(values[region], 0.35)
    if 0 <= saliency_region < n_regions and saliency_region != true_region:
        values[saliency_region] = 0.05
    return values


def apply_inspection(episode: dict[str, Any], region_id: int, config: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(episode)
    env = config.get("env", {})
    frame_size = int(env.get("frame_size", updated["past_frames"].shape[1]))
    grid_size = int(env.get("grid_size", 8))
    ys, xs = region_id_to_slice(int(region_id), frame_size, grid_size)
    patch = np.asarray(updated["past_frames"][-1, ys, xs, :], dtype=np.float32).copy()
    updated["inspection"] = {
        "region_id": int(region_id),
        "inspected_patch": patch,
        "inspection_value": float(updated["ground_truth"].get("inspection_values", {}).get(int(region_id), 0.0)),
    }
    return updated


def make_b3_datasets(config: dict[str, Any], seed: int = 0) -> dict[str, list[dict[str, Any]]]:
    runtime = b3_runtime_config(config)
    b3 = b3_config(config)
    n_test = effective_count(int(b3.get("n_test", 32)), b3)
    n_ood = effective_count(int(b3.get("n_ood", 32)), b3)
    delays = [2, 4, 6]
    heldout = [int(delay) for delay in b3.get("ood", {}).get("heldout_delays", [3, 5, 7])]
    extrap = [int(delay) for delay in b3.get("ood", {}).get("extrapolation_delays", [8, 10])]
    ood_delays = heldout + extrap
    return {
        "test": [
            make_b3_active_inspection_episode(runtime, seed + idx * 17, "b3_trace_guided_inspection", delays[idx % len(delays)])
            for idx in range(n_test)
        ],
        "conflict": [
            make_b3_active_inspection_episode(runtime, seed + 10000 + idx * 19, "b3_trace_vs_saliency_conflict", delays[idx % len(delays)])
            for idx in range(n_test)
        ],
        "ood": [
            make_b3_active_inspection_episode(runtime, seed + 20000 + idx * 23, "b3_delay_ood_inspection", ood_delays[idx % len(ood_delays)])
            for idx in range(n_ood)
        ],
    }


def b3_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    base = resolve_b3_base_config(config)
    runtime = b23_runtime_config(base)
    b3 = b3_config(config)
    env = dict(runtime.get("env", {}))
    for key in ("frame_size", "grid_size", "past_frames", "future_frames"):
        if key in b3:
            env[key] = int(b3[key])
    runtime["env"] = env
    runtime["b2"] = {
        **runtime.get("b2", {}),
        "n_train": effective_count(int(b3.get("n_train", 32)), b3),
        "n_test": effective_count(int(b3.get("n_test", 32)), b3),
        "n_ood": effective_count(int(b3.get("n_ood", 32)), b3),
        "delays": [2, 4, 6],
        "heldout_delays": [3, 5, 7],
    }
    return runtime


def b3_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("b3", {"n_test": 32, "n_ood": 32, "gates": {}})


def resolve_b3_base_config(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(config.get("base_config", "configs/b23_private_selector.yaml")))
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def effective_count(value: int, b3: dict[str, Any]) -> int:
    cap = b3.get("max_eval_episodes")
    if cap is not None:
        value = min(value, int(cap))
    return max(1, int(value))


def neighbor_regions(region: int, grid_size: int) -> list[int]:
    gy, gx = divmod(int(region), grid_size)
    regions = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = gy + dy, gx + dx
            if 0 <= ny < grid_size and 0 <= nx < grid_size:
                regions.append(ny * grid_size + nx)
    return regions
