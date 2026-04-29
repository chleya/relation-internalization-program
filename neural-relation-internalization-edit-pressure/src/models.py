from __future__ import annotations

import torch
from torch import nn

from .data import RESOURCES, resource_rule
from .features import FEATURE_DIM, LABEL_DIM, encode_context, encode_label


class PurePredictionMLP(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(FEATURE_DIM, hidden_dim), nn.ReLU())
        self.head = nn.Linear(hidden_dim, LABEL_DIM)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_from_hidden(self.hidden(x))


class BottleneckPredictionMLP(nn.Module):
    def __init__(self, hidden_dim: int = 32, bottleneck_dim: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(FEATURE_DIM, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, bottleneck_dim))
        self.activation = nn.Tanh()
        self.head = nn.Linear(bottleneck_dim, LABEL_DIM)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.encoder(x))

    def forward_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_from_hidden(self.hidden(x))


class CounterfactualMLP(PurePredictionMLP):
    pass


class EditPressureModel(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.query_encoder = nn.Sequential(nn.Linear(FEATURE_DIM, hidden_dim), nn.ReLU())
        self.support_encoder = nn.Sequential(nn.Linear(FEATURE_DIM + LABEL_DIM, hidden_dim), nn.ReLU())
        self.edit_encoder = nn.Sequential(nn.Linear(FEATURE_DIM + LABEL_DIM, hidden_dim), nn.ReLU())
        self.edit_module = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())
        self.head = nn.Linear(hidden_dim * 2, LABEL_DIM)

    def encode_support(self, support_x: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
        y_one_hot = torch.nn.functional.one_hot(support_y, LABEL_DIM).float()
        items = self.support_encoder(torch.cat([support_x, y_one_hot], dim=-1))
        return items.mean(dim=0, keepdim=True)

    def encode_edit(self, edit_condition: torch.Tensor, edit_outcome: torch.Tensor) -> torch.Tensor:
        y_one_hot = torch.nn.functional.one_hot(edit_outcome.view(-1), LABEL_DIM).float()
        return self.edit_encoder(torch.cat([edit_condition.view(1, -1), y_one_hot], dim=-1))

    def apply_edit(self, relation_state: torch.Tensor, edit_embedding: torch.Tensor) -> torch.Tensor:
        return self.edit_module(torch.cat([relation_state, edit_embedding], dim=-1))

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(x)

    def forward_from_hidden(self, hidden: torch.Tensor, relation_state: torch.Tensor | None = None) -> torch.Tensor:
        if relation_state is None:
            relation_state = torch.zeros(1, hidden.shape[-1], device=hidden.device).expand(hidden.shape[0], -1)
        elif relation_state.shape[0] == 1 and hidden.shape[0] > 1:
            relation_state = relation_state.expand(hidden.shape[0], -1)
        return self.head(torch.cat([hidden, relation_state], dim=-1))

    def forward(
        self,
        query_x: torch.Tensor,
        support: tuple[torch.Tensor, torch.Tensor] | None = None,
        edit: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        hidden = self.hidden(query_x)
        relation_state = None
        if support is not None:
            relation_state = self.encode_support(*support)
        if relation_state is not None and edit is not None:
            relation_state = self.apply_edit(relation_state, self.encode_edit(*edit))
        return self.forward_from_hidden(hidden, relation_state)


class ExplicitTableOracle:
    name = "explicit_table_oracle"

    def __init__(self, regime: str = "base") -> None:
        self.regime = regime
        self.edits: dict[tuple[str, str], str] = {}

    def predict_resource(self, context: dict[str, str]) -> str:
        key = (context["texture"], context["wet"])
        if key in self.edits:
            return self.edits[key]
        return resource_rule(context, self.regime)

    def edit_rule(self, condition: dict[str, str], new_outcome: str) -> bool:
        self.edits[(condition["texture"], condition["wet"])] = new_outcome
        return True

    def act(self, context: dict[str, str]) -> str:
        return "eat" if self.predict_resource(context) == "food" else "avoid"


def make_model(name: str, hidden_dim: int = 32, bottleneck_dim: int = 4) -> nn.Module | ExplicitTableOracle:
    if name == "pure_prediction":
        return PurePredictionMLP(hidden_dim)
    if name == "prediction_bottleneck":
        return BottleneckPredictionMLP(hidden_dim, bottleneck_dim)
    if name == "counterfactual_training":
        return CounterfactualMLP(hidden_dim)
    if name == "edit_pressure_training":
        return EditPressureModel(hidden_dim)
    if name == "explicit_table_oracle":
        return ExplicitTableOracle()
    raise ValueError(f"unknown model: {name}")
