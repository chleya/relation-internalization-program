from __future__ import annotations

import torch
from torch.nn import functional as F

from .features import dataset_tensors, encode_context, encode_label


def prediction_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def bottleneck_loss(logits: torch.Tensor, labels: torch.Tensor, z: torch.Tensor, beta: float) -> torch.Tensor:
    return prediction_loss(logits, labels) + beta * z.abs().mean()


def counterfactual_loss(model, batch, lambda_inv: float, lambda_cf: float) -> torch.Tensor:
    x, y, nuisance_x, relation_x, relation_y = batch
    logits = model(x)
    nuisance_logits = model(nuisance_x)
    relation_logits = model(relation_x)
    inv = F.kl_div(F.log_softmax(nuisance_logits, dim=-1), F.softmax(logits.detach(), dim=-1), reduction="batchmean")
    rel = F.cross_entropy(relation_logits, relation_y)
    return F.cross_entropy(logits, y) + lambda_inv * inv + lambda_cf * rel


def edit_pressure_loss(model, episode_batch: list[dict], lambda_local: float, lambda_cons: float) -> torch.Tensor:
    losses = []
    locality_losses = []
    support_losses = []
    for episode in episode_batch:
        support_x, support_y = dataset_tensors(episode["support"])
        query_x = encode_context(episode["query"]).view(1, -1)
        before_y = encode_label(episode["target_before"]).view(1)
        after_y = encode_label(episode["target_after"]).view(1)
        edit_context = dict(episode["query"])
        edit_context.update(episode["edit"]["condition"])
        edit_x = encode_context(edit_context)
        edit_y = encode_label(episode["edit"]["new_outcome"])
        before_logits = model(query_x, support=(support_x, support_y))
        after_logits = model(query_x, support=(support_x, support_y), edit=(edit_x, edit_y))
        losses.append(F.cross_entropy(before_logits, before_y) + F.cross_entropy(after_logits, after_y))
        support_logits = model(support_x, support=(support_x, support_y))
        support_losses.append(F.cross_entropy(support_logits, support_y))
        non_target = [record for record in episode["support"] if not all(record[k] == v for k, v in episode["edit"]["condition"].items())]
        if non_target:
            non_x, _ = dataset_tensors(non_target)
            non_before = model(non_x, support=(support_x, support_y))
            non_after = model(non_x, support=(support_x, support_y), edit=(edit_x, edit_y))
            locality_losses.append(F.kl_div(F.log_softmax(non_after, dim=-1), F.softmax(non_before.detach(), dim=-1), reduction="batchmean"))
    return torch.stack(losses).mean() + lambda_local * torch.stack(locality_losses).mean() + lambda_cons * torch.stack(support_losses).mean()
