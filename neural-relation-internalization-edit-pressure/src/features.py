from __future__ import annotations

import torch

from .data import COLORS, ODORS, RESOURCES, TEXTURES, WETS


FEATURE_DIM = len(TEXTURES) + len(WETS) + len(COLORS) + len(ODORS)
LABEL_DIM = len(RESOURCES)


def encode_context(context: dict[str, str]) -> torch.Tensor:
    values = []
    values += [1.0 if context["texture"] == item else 0.0 for item in TEXTURES]
    values += [1.0 if context["wet"] == item else 0.0 for item in WETS]
    values += [1.0 if context["color"] == item else 0.0 for item in COLORS]
    values += [1.0 if context["odor"] == item else 0.0 for item in ODORS]
    return torch.tensor(values, dtype=torch.float32)


def label_index(resource: str) -> int:
    return RESOURCES.index(resource)


def label_name(index: int) -> str:
    return RESOURCES[int(index)]


def encode_label(resource: str) -> torch.Tensor:
    return torch.tensor(label_index(resource), dtype=torch.long)


def dataset_tensors(records: list[dict[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([encode_context(record) for record in records])
    y = torch.stack([encode_label(record["resource"]) for record in records])
    return x, y


def relation_label(record: dict[str, str]) -> int:
    return TEXTURES.index(record["texture"]) * len(WETS) + WETS.index(record["wet"])


def nuisance_label(record: dict[str, str]) -> int:
    return COLORS.index(record["color"]) * len(ODORS) + ODORS.index(record["odor"])
