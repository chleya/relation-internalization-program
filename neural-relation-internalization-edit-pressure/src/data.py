from __future__ import annotations

import random
from typing import Any


TEXTURES = ["A", "B", "C"]
WETS = ["dry", "wet"]
COLORS = ["red", "blue"]
ODORS = ["strong", "weak"]
RESOURCES = ["food", "poison", "neutral"]


def resource_rule(context: dict[str, str], regime: str = "base") -> str:
    texture = context["texture"]
    wet = context["wet"]
    if regime == "base":
        if texture == "A" and wet == "dry":
            return "food"
        if texture == "A" and wet == "wet":
            return "poison"
        if texture == "B":
            return "poison"
        return "neutral"
    if regime == "reversal":
        if texture == "A" and wet == "dry":
            return "poison"
        if texture == "A" and wet == "wet":
            return "food"
        if texture == "B":
            return "food"
        return "neutral"
    raise ValueError(f"unknown regime: {regime}")


def action_for_resource(resource: str) -> str:
    return "eat" if resource == "food" else "avoid"


def reward(action: str, resource: str) -> float:
    if action == "eat" and resource == "food":
        return 1.0
    if action == "eat" and resource == "poison":
        return -1.0
    if action == "eat" and resource == "neutral":
        return 0.0
    if action == "avoid" and resource == "food":
        return -0.2
    if action == "avoid" and resource == "poison":
        return 0.2
    return 0.0


def sample_context(rng: random.Random, shortcut_mode: str = "none", regime: str = "base") -> dict[str, str]:
    context = {
        "texture": rng.choice(TEXTURES),
        "wet": rng.choice(WETS),
        "color": rng.choice(COLORS),
        "odor": rng.choice(ODORS),
    }
    resource = resource_rule(context, regime)
    if shortcut_mode == "train":
        if resource == "food" and rng.random() < 0.95:
            context["color"] = "red"
            context["odor"] = "strong"
        elif resource == "poison" and rng.random() < 0.95:
            context["color"] = "blue"
            context["odor"] = "weak"
    elif shortcut_mode == "attack":
        if resource == "food":
            context["color"] = "blue"
            context["odor"] = "weak"
        elif resource == "poison":
            context["color"] = "red"
            context["odor"] = "strong"
    elif shortcut_mode == "none":
        pass
    else:
        raise ValueError(f"unknown shortcut_mode: {shortcut_mode}")
    return context


def make_record(context: dict[str, str], regime: str = "base") -> dict[str, Any]:
    resource = resource_rule(context, regime)
    return {
        **context,
        "resource": resource,
        "action": action_for_resource(resource),
    }


def generate_dataset(n: int, seed: int, regime: str = "base", shortcut_mode: str = "none") -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return [make_record(sample_context(rng, shortcut_mode, regime), regime) for _ in range(n)]


def canonical_context(texture: str, wet: str, color: str = "red", odor: str = "strong") -> dict[str, str]:
    return {"texture": texture, "wet": wet, "color": color, "odor": odor}


def canonical_conditions() -> list[dict[str, str]]:
    return [canonical_context(texture, wet) for texture in TEXTURES for wet in WETS]


def make_counterfactual_pair(record: dict[str, Any], cf_type: str) -> tuple[dict[str, Any], dict[str, Any]]:
    left = dict(record)
    context = {key: record[key] for key in ["texture", "wet", "color", "odor"]}
    if cf_type == "nuisance_change":
        context["color"] = "blue" if context["color"] == "red" else "red"
        context["odor"] = "weak" if context["odor"] == "strong" else "strong"
        right = make_record(context, "base")
    elif cf_type == "relation_change":
        context["wet"] = "wet" if context["wet"] == "dry" else "dry"
        if context["texture"] == "C":
            context["texture"] = "A"
        right = make_record(context, "base")
    else:
        raise ValueError(f"unknown cf_type: {cf_type}")
    return left, right


def generate_edit_episode(
    seed: int,
    regime: str = "base",
    edit_condition: dict[str, str] | None = None,
    new_outcome: str = "poison",
) -> dict[str, Any]:
    rng = random.Random(seed)
    edit_condition = edit_condition or {"texture": "A", "wet": "dry"}
    support = []
    for condition in canonical_conditions():
        context = dict(condition)
        context["color"] = rng.choice(COLORS)
        context["odor"] = rng.choice(ODORS)
        support.append(make_record(context, regime))
    query = {
        **edit_condition,
        "color": rng.choice(COLORS),
        "odor": rng.choice(ODORS),
    }
    before = resource_rule(query, regime)
    after = new_outcome if all(query[k] == v for k, v in edit_condition.items()) else before
    return {
        "support": support,
        "query": query,
        "edit": {"condition": edit_condition, "new_outcome": new_outcome},
        "target_before": before,
        "target_after": after,
    }
