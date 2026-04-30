from __future__ import annotations

from typing import Any

from .env import EPISODE_TYPES, PLOSEnv


def generate_episode(config: dict[str, Any], seed: int, episode_type: str | None = None) -> dict[str, Any]:
    env_config = config.get("env", config)
    env = PLOSEnv(env_config, seed=seed)
    return env.run_episode(episode_type)


def _split_size(config: dict[str, Any], split: str) -> int:
    data = config.get("data", {})
    runtime = config.get("runtime", {})
    declared = int(data.get(f"n_{split}", data.get("n_test", 32)))
    if split == "train":
        return min(declared, int(runtime.get("max_train_episodes", declared)))
    if split == "ood":
        return min(declared, int(runtime.get("max_ood_episodes", declared)))
    return min(declared, int(runtime.get("max_eval_episodes", declared)))


def generate_dataset(config: dict[str, Any], split: str, seed: int) -> list[dict[str, Any]]:
    n = _split_size(config, split)
    episodes = []
    for idx in range(n):
        episode_type = EPISODE_TYPES[idx % len(EPISODE_TYPES)]
        episodes.append(generate_episode(config, seed + idx * 997, episode_type))
    return episodes


def make_ood_dataset(config: dict[str, Any], ood_type: str, seed: int) -> list[dict[str, Any]]:
    ood_config = {**config, "env": dict(config.get("env", {}))}
    env = ood_config["env"]
    if ood_type == "new_speed":
        env["speed_scale"] = 1.35
    elif ood_type == "new_color":
        env["ood_color"] = True
    elif ood_type == "new_occluder_position":
        env["render_occluder"] = True
    elif ood_type == "new_collision_angle":
        env["collision_angle_jitter"] = True
    elif ood_type == "new_forcefield_position":
        env["forcefield_ood"] = True
    return generate_dataset({**ood_config, "data": {**ood_config.get("data", {}), "n_ood": ood_config.get("data", {}).get("n_ood", 32)}}, "ood", seed)
