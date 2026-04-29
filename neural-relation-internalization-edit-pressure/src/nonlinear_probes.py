from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .data import generate_dataset
from .extraction import canonical_support
from .features import dataset_tensors, nuisance_label, relation_label
from .models import EditPressureModel, ExplicitTableOracle


def representation_for_site(model, dataset: list[dict[str, str]], representation_site: str = "combined_hidden") -> torch.Tensor:
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
                        1.0 if record["odor"] == "strong" else 0.0,
                    ],
                    dtype=torch.float32,
                )
            )
        return torch.stack(rows)

    x, _ = dataset_tensors(dataset)
    with torch.no_grad():
        if isinstance(model, EditPressureModel):
            query_hidden = model.hidden(x)
            support_state = model.encode_support(*canonical_support()).expand(query_hidden.shape[0], -1)
            if representation_site == "query_hidden":
                return query_hidden.detach()
            if representation_site == "support_relation_state":
                return support_state.detach()
            if representation_site == "combined_hidden":
                return torch.cat([query_hidden, support_state], dim=-1).detach()
            if representation_site == "post_edit_state":
                edit_x = x[0]
                edit_y = torch.tensor(1, dtype=torch.long)
                post = model.apply_edit(support_state[:1], model.encode_edit(edit_x, edit_y)).expand(query_hidden.shape[0], -1)
                return post.detach()
            if representation_site == "edit_embedding":
                edit_x = x[0]
                edit_y = torch.tensor(1, dtype=torch.long)
                return model.encode_edit(edit_x, edit_y).expand(query_hidden.shape[0], -1).detach()
            raise ValueError(f"unknown representation_site: {representation_site}")
        return model.hidden(x).detach()


def train_linear_probe(hidden: torch.Tensor, labels: torch.Tensor, seed: int = 0):
    torch.manual_seed(seed)
    n_classes = int(labels.max().item()) + 1
    probe = nn.Linear(hidden.shape[1], n_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.05)
    for _ in range(120):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(probe(hidden), labels)
        loss.backward()
        optimizer.step()
    return probe, probe_accuracy(probe, hidden, labels)


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp_probe(hidden: torch.Tensor, labels: torch.Tensor, seed: int = 0, hidden_dim: int = 32):
    torch.manual_seed(seed)
    n_classes = int(labels.max().item()) + 1
    probe = MLPProbe(hidden.shape[1], n_classes, hidden_dim)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.03)
    for _ in range(160):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(probe(hidden), labels)
        loss.backward()
        optimizer.step()
    return probe, probe_accuracy(probe, hidden, labels)


@dataclass
class TreeNode:
    prediction: int
    feature: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


class SimpleTreeProbe:
    def __init__(self, max_depth: int = 4) -> None:
        self.max_depth = max_depth
        self.root: TreeNode | None = None

    def fit(self, hidden: torch.Tensor, labels: torch.Tensor) -> None:
        self.root = self._build(hidden, labels, 0)

    def predict(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.root is None:
            raise ValueError("tree is not fitted")
        return torch.tensor([self._predict_one(row, self.root) for row in hidden], dtype=torch.long)

    def _build(self, hidden: torch.Tensor, labels: torch.Tensor, depth: int) -> TreeNode:
        prediction = majority_label(labels)
        if depth >= self.max_depth or len(set(labels.tolist())) <= 1 or hidden.shape[0] < 4:
            return TreeNode(prediction=prediction)
        split = best_split(hidden, labels)
        if split is None:
            return TreeNode(prediction=prediction)
        feature, threshold = split
        mask = hidden[:, feature] <= threshold
        if mask.sum() == 0 or (~mask).sum() == 0:
            return TreeNode(prediction=prediction)
        return TreeNode(
            prediction=prediction,
            feature=feature,
            threshold=float(threshold),
            left=self._build(hidden[mask], labels[mask], depth + 1),
            right=self._build(hidden[~mask], labels[~mask], depth + 1),
        )

    def _predict_one(self, row: torch.Tensor, node: TreeNode) -> int:
        if node.feature is None or node.threshold is None or node.left is None or node.right is None:
            return node.prediction
        if float(row[node.feature].item()) <= node.threshold:
            return self._predict_one(row, node.left)
        return self._predict_one(row, node.right)


def train_tree_probe(hidden: torch.Tensor, labels: torch.Tensor, seed: int = 0, max_depth: int = 4):
    random.seed(seed)
    probe = SimpleTreeProbe(max_depth=max_depth)
    probe.fit(hidden, labels)
    pred = probe.predict(hidden)
    return probe, float((pred == labels).float().mean().item())


def shuffled_label_control(labels: torch.Tensor, seed: int = 0) -> torch.Tensor:
    rng = random.Random(seed)
    values = labels.tolist()
    shuffled = list(values)
    rng.shuffle(shuffled)
    if shuffled == values and len(shuffled) > 1:
        shuffled = shuffled[1:] + shuffled[:1]
    return torch.tensor(shuffled, dtype=torch.long)


def probe_selectivity(real_acc: float, shuffled_acc: float) -> float:
    return real_acc - shuffled_acc


def probe_accuracy(probe, hidden: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        if isinstance(probe, SimpleTreeProbe):
            pred = probe.predict(hidden)
        else:
            pred = torch.argmax(probe(hidden), dim=-1)
        return float((pred == labels).float().mean().item())


def run_nonlinear_probe_suite(model, datasets: dict[str, list[dict[str, str]]], seed: int, representation_site: str = "combined_hidden") -> dict[str, float]:
    dataset = datasets.get("probe") or generate_dataset(500, seed + 50_000, "base", "none")
    hidden = representation_for_site(model, dataset, representation_site)
    relation_labels = torch.tensor([relation_label(record) for record in dataset], dtype=torch.long)
    nuisance_labels = torch.tensor([nuisance_label(record) for record in dataset], dtype=torch.long)
    shuffled_relation = shuffled_label_control(relation_labels, seed + 1)

    _, linear_relation_acc = train_linear_probe(hidden, relation_labels, seed)
    _, mlp_relation_acc = train_mlp_probe(hidden, relation_labels, seed)
    _, tree_relation_acc = train_tree_probe(hidden, relation_labels, seed)
    _, linear_nuisance_acc = train_linear_probe(hidden, nuisance_labels, seed + 10)
    _, mlp_nuisance_acc = train_mlp_probe(hidden, nuisance_labels, seed + 10)
    _, tree_nuisance_acc = train_tree_probe(hidden, nuisance_labels, seed + 10)
    _, linear_random_acc = train_linear_probe(hidden, shuffled_relation, seed + 20)
    _, mlp_random_acc = train_mlp_probe(hidden, shuffled_relation, seed + 20)
    _, tree_random_acc = train_tree_probe(hidden, shuffled_relation, seed + 20)

    return {
        "linear_relation_acc": linear_relation_acc,
        "mlp_relation_acc": mlp_relation_acc,
        "tree_relation_acc": tree_relation_acc,
        "linear_nuisance_acc": linear_nuisance_acc,
        "mlp_nuisance_acc": mlp_nuisance_acc,
        "tree_nuisance_acc": tree_nuisance_acc,
        "linear_relation_selectivity": probe_selectivity(linear_relation_acc, linear_random_acc),
        "mlp_relation_selectivity": probe_selectivity(mlp_relation_acc, mlp_random_acc),
        "tree_relation_selectivity": probe_selectivity(tree_relation_acc, tree_random_acc),
        "nonlinear_relation_gain": max(mlp_relation_acc, tree_relation_acc) - linear_relation_acc,
        "nonlinear_nuisance_gain": max(mlp_nuisance_acc, tree_nuisance_acc) - linear_nuisance_acc,
    }


def majority_label(labels: torch.Tensor) -> int:
    values, counts = torch.unique(labels, return_counts=True)
    return int(values[torch.argmax(counts)].item())


def best_split(hidden: torch.Tensor, labels: torch.Tensor) -> tuple[int, float] | None:
    best: tuple[float, int, float] | None = None
    for feature in range(hidden.shape[1]):
        values = torch.unique(hidden[:, feature])
        if len(values) > 12:
            values = torch.quantile(hidden[:, feature], torch.linspace(0.1, 0.9, 9))
        for threshold in values:
            mask = hidden[:, feature] <= threshold
            if mask.sum() == 0 or (~mask).sum() == 0:
                continue
            score = weighted_gini(labels[mask], labels[~mask])
            if best is None or score < best[0]:
                best = (score, feature, float(threshold.item()))
    if best is None:
        return None
    return best[1], best[2]


def weighted_gini(left: torch.Tensor, right: torch.Tensor) -> float:
    total = len(left) + len(right)
    return (len(left) / total) * gini(left) + (len(right) / total) * gini(right)


def gini(labels: torch.Tensor) -> float:
    _, counts = torch.unique(labels, return_counts=True)
    probs = counts.float() / counts.sum()
    return float(1.0 - torch.sum(probs * probs).item())
