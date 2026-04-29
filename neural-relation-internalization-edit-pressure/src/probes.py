from __future__ import annotations

import torch
from torch import nn

from .features import dataset_tensors, nuisance_label, relation_label
from .models import EditPressureModel, ExplicitTableOracle
from .extraction import canonical_support


def collect_hidden_states(model, dataset: list[dict[str, str]]) -> torch.Tensor:
    if isinstance(model, ExplicitTableOracle):
        rows = []
        for record in dataset:
            rows.append(
                torch.tensor(
                    [
                        1.0 if record["texture"] == "A" else 0.0,
                        1.0 if record["texture"] == "B" else 0.0,
                        1.0 if record["wet"] == "dry" else 0.0,
                        1.0 if record["color"] == "red" else 0.0,
                    ],
                    dtype=torch.float32,
                )
            )
        return torch.stack(rows)
    x, _ = dataset_tensors(dataset)
    with torch.no_grad():
        if isinstance(model, EditPressureModel):
            query_hidden = model.hidden(x)
            relation_state = model.encode_support(*canonical_support()).expand(query_hidden.shape[0], -1)
            return torch.cat([query_hidden, relation_state], dim=-1).detach()
        return model.hidden(x).detach()


def _train_probe(hidden: torch.Tensor, labels: torch.Tensor, n_classes: int) -> nn.Linear:
    torch.manual_seed(0)
    probe = nn.Linear(hidden.shape[1], n_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.05)
    for _ in range(120):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(probe(hidden), labels)
        loss.backward()
        optimizer.step()
    return probe


def train_relation_probe(hidden: torch.Tensor, records: list[dict[str, str]]) -> nn.Linear:
    labels = torch.tensor([relation_label(record) for record in records], dtype=torch.long)
    return _train_probe(hidden, labels, 6)


def train_nuisance_probe(hidden: torch.Tensor, records: list[dict[str, str]]) -> nn.Linear:
    labels = torch.tensor([nuisance_label(record) for record in records], dtype=torch.long)
    return _train_probe(hidden, labels, 4)


def probe_accuracy(probe: nn.Linear, hidden: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        pred = torch.argmax(probe(hidden), dim=-1)
        return float((pred == labels).float().mean().item())


def relation_probe_accuracy(probe: nn.Linear, hidden: torch.Tensor, records: list[dict[str, str]]) -> float:
    labels = torch.tensor([relation_label(record) for record in records], dtype=torch.long)
    return probe_accuracy(probe, hidden, labels)


def nuisance_probe_accuracy(probe: nn.Linear, hidden: torch.Tensor, records: list[dict[str, str]]) -> float:
    labels = torch.tensor([nuisance_label(record) for record in records], dtype=torch.long)
    return probe_accuracy(probe, hidden, labels)


def probe_selectivity(relation_probe_acc: float, random_label_acc: float) -> float:
    return relation_probe_acc - random_label_acc
