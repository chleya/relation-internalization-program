from __future__ import annotations

from typing import Any

from .model_io import make_model_batch
from .models import make_model
from .models.base import BasePLOSModel


def train_model(model_name: str, train_dataset: list[dict[str, Any]], config: dict[str, Any]) -> BasePLOSModel:
    model = make_model(model_name)
    clean_train = [make_model_batch(episode, config) for episode in train_dataset]
    model.fit(clean_train, config)
    return model
