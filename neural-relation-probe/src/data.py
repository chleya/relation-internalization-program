from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TEXTURES = ["A", "B", "C"]
WETS = ["dry", "wet"]
COLORS = ["red", "blue"]
ODORS = ["strong", "weak"]
RESOURCES = ["food", "poison", "neutral"]
RELATIONS = [f"{texture}_{wet}" for texture in TEXTURES for wet in WETS]
NUISANCES = [f"{color}_{odor}" for color in COLORS for odor in ODORS]


def resource_for(texture: str, wet: str) -> str:
    if texture == "A" and wet == "dry":
        return "food"
    if texture == "A" and wet == "wet":
        return "poison"
    if texture == "B":
        return "poison"
    return "neutral"


def encode_context(context: dict[str, str]) -> np.ndarray:
    values = [
        (TEXTURES, context["texture"]),
        (WETS, context["wet"]),
        (COLORS, context["color"]),
        (ODORS, context["odor"]),
    ]
    parts = []
    for choices, value in values:
        parts.extend([1.0 if value == item else 0.0 for item in choices])
    return np.asarray(parts, dtype=float)


def encode_label(value: str, choices: list[str]) -> int:
    return choices.index(value)


@dataclass
class Dataset:
    x: np.ndarray
    y_resource: np.ndarray
    y_relation: np.ndarray
    y_nuisance: np.ndarray
    contexts: list[dict[str, str]]


def _surface_cues(resource: str, rng: np.random.Generator, mode: str) -> tuple[str, str]:
    if mode == "shortcut":
        return {
            "food": ("red", "strong"),
            "poison": ("blue", "weak"),
            "neutral": ("red", "weak"),
        }[resource]
    if mode == "base":
        preferred = {
            "food": ("red", "strong"),
            "poison": ("blue", "weak"),
            "neutral": ("red", "weak"),
        }[resource]
        if rng.random() < 0.9:
            return preferred
    if mode == "spurious_attack":
        preferred = {
            "food": ("blue", "weak"),
            "poison": ("red", "strong"),
            "neutral": ("blue", "strong"),
        }[resource]
        if rng.random() < 0.9:
            return preferred
    return rng.choice(COLORS).item(), rng.choice(ODORS).item()


def make_dataset(n: int, seed: int = 0, mode: str = "base") -> Dataset:
    rng = np.random.default_rng(seed)
    contexts: list[dict[str, str]] = []
    xs = []
    y_resource = []
    y_relation = []
    y_nuisance = []
    pairs = [(t, w) for t in TEXTURES for w in WETS]
    for _ in range(n):
        texture, wet = pairs[rng.integers(0, len(pairs))]
        resource = resource_for(texture, wet)
        color, odor = _surface_cues(resource, rng, mode)
        context = {"texture": texture, "wet": wet, "color": color, "odor": odor}
        contexts.append(context)
        xs.append(encode_context(context))
        y_resource.append(encode_label(resource, RESOURCES))
        y_relation.append(encode_label(f"{texture}_{wet}", RELATIONS))
        y_nuisance.append(encode_label(f"{color}_{odor}", NUISANCES))
    return Dataset(
        np.vstack(xs),
        np.asarray(y_resource),
        np.asarray(y_relation),
        np.asarray(y_nuisance),
        contexts,
    )
