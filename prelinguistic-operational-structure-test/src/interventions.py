from __future__ import annotations

from typing import Any


def slot_removal() -> dict[str, Any]:
    return {"type": "slot_removal"}


def slot_swap() -> dict[str, Any]:
    return {"type": "slot_swap"}


def slot_noise() -> dict[str, Any]:
    return {"type": "slot_noise"}


def event_latent_perturbation() -> dict[str, Any]:
    return {"type": "event_latent_perturbation"}


def relation_edge_ablation() -> dict[str, Any]:
    return {"type": "relation_edge_ablation"}


def inspection_map_shuffle() -> dict[str, Any]:
    return {"type": "inspection_map_shuffle"}


def inspection_topk_zero() -> dict[str, Any]:
    return {"type": "inspection_topk_zero"}


def field_patch_mask() -> dict[str, Any]:
    return {"type": "field_patch_mask"}


def field_patch_swap() -> dict[str, Any]:
    return {"type": "field_patch_swap"}


def critical_field_zero() -> dict[str, Any]:
    return {"type": "critical_field_zero"}


def default_interventions() -> list[dict[str, Any]]:
    return [
        slot_removal(),
        slot_swap(),
        slot_noise(),
        event_latent_perturbation(),
        relation_edge_ablation(),
        inspection_map_shuffle(),
        inspection_topk_zero(),
        field_patch_mask(),
        field_patch_swap(),
        critical_field_zero(),
    ]
