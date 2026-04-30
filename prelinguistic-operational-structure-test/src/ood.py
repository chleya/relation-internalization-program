from __future__ import annotations

from typing import Any

from .data import make_ood_dataset


def make_ood_color_dataset(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    return make_ood_dataset(config, "new_color", seed)


def make_ood_speed_dataset(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    return make_ood_dataset(config, "new_speed", seed)


def make_ood_occluder_dataset(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    return make_ood_dataset(config, "new_occluder_position", seed)


def make_ood_collision_angle_dataset(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    return make_ood_dataset(config, "new_collision_angle", seed)


def make_ood_forcefield_dataset(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    return make_ood_dataset(config, "new_forcefield_position", seed)


def make_all_ood(config: dict[str, Any], seed: int) -> dict[str, list[dict[str, Any]]]:
    return {
        "new_color": make_ood_color_dataset(config, seed + 11),
        "new_speed": make_ood_speed_dataset(config, seed + 13),
        "new_occluder_position": make_ood_occluder_dataset(config, seed + 17),
        "new_collision_angle": make_ood_collision_angle_dataset(config, seed + 19),
        "new_forcefield_position": make_ood_forcefield_dataset(config, seed + 23),
    }
