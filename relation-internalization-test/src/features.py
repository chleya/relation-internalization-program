from itertools import combinations, product

import numpy as np

TEXTURES = ["A", "B", "C"]
WETS = ["dry", "wet"]
COLORS = [f"color_{idx:02d}" for idx in range(16)]
ODORS = [f"odor_{idx:02d}" for idx in range(16)]
TRAIN_COLORS = COLORS[:4]
TRAIN_ODORS = ODORS[:4]
OOD_COLORS = COLORS[4:]
OOD_ODORS = ODORS[4:]
RESOURCES = ["food", "poison", "neutral"]
ACTIONS = ["eat", "avoid"]

FEATURE_VALUES = {
    "texture": TEXTURES,
    "wet": WETS,
    "color": COLORS,
    "odor": ODORS,
}
FEATURE_ORDER = ["texture", "wet", "color", "odor"]


def one_hot_context(context: dict) -> np.ndarray:
    values = []
    for key in FEATURE_ORDER:
        values.extend(1.0 if context[key] == value else 0.0 for value in FEATURE_VALUES[key])
    return np.array(values, dtype=float)


def context_to_tuple(context: dict) -> tuple:
    return tuple(context[key] for key in FEATURE_ORDER)


def matches_condition(context: dict, condition: dict) -> bool:
    return all(context.get(key) == value for key, value in condition.items())


def condition_key(condition: dict) -> tuple:
    return tuple(sorted(condition.items()))


def enumerate_candidate_conditions(max_order: int = 2, keys: list[str] | None = None) -> list[dict]:
    conditions: list[dict] = []
    feature_keys = keys or FEATURE_ORDER
    for order in range(1, max_order + 1):
        for condition_keys in combinations(feature_keys, order):
            for values in product(*(FEATURE_VALUES[key] for key in condition_keys)):
                conditions.append(dict(zip(condition_keys, values)))
    return conditions


def all_contexts() -> list[dict]:
    contexts = []
    for texture, wet, color, odor in product(TEXTURES, WETS, COLORS, ODORS):
        contexts.append({"texture": texture, "wet": wet, "color": color, "odor": odor})
    return contexts
