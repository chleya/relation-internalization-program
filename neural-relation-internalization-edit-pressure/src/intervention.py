from __future__ import annotations

import torch

from .features import dataset_tensors
from .models import EditPressureModel, ExplicitTableOracle
from .extraction import canonical_support


def remove_probe_subspace(hidden: torch.Tensor, probe) -> torch.Tensor:
    weight = probe.weight.detach()
    q, _ = torch.linalg.qr(weight.T, mode="reduced")
    projection = hidden @ q @ q.T
    return hidden - projection


def subspace_intervention_drop(model, dataset: list[dict[str, str]], probe, kind: str) -> float:
    if isinstance(model, ExplicitTableOracle):
        return 1.0 if kind == "relation" else 0.0
    x, y = dataset_tensors(dataset)
    with torch.no_grad():
        if isinstance(model, EditPressureModel):
            query_hidden = model.hidden(x)
            relation_state = model.encode_support(*canonical_support()).expand(query_hidden.shape[0], -1)
            hidden = torch.cat([query_hidden, relation_state], dim=-1)
            base = torch.argmax(model.head(hidden), dim=-1)
            edited_hidden = remove_probe_subspace(hidden, probe)
            edited = torch.argmax(model.head(edited_hidden), dim=-1)
            base_acc = float((base == y).float().mean().item())
            edited_acc = float((edited == y).float().mean().item())
            return max(0.0, base_acc - edited_acc)
        hidden = model.hidden(x)
        base_logits = model.forward_from_hidden(hidden)
        base = torch.argmax(base_logits, dim=-1)
        edited_hidden = remove_probe_subspace(hidden, probe)
        edited_logits = model.forward_from_hidden(edited_hidden)
        edited = torch.argmax(edited_logits, dim=-1)
        base_acc = float((base == y).float().mean().item())
        edited_acc = float((edited == y).float().mean().item())
    return max(0.0, base_acc - edited_acc)
