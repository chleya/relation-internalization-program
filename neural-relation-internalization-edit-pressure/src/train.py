from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import RESOURCES, TEXTURES, WETS, generate_dataset, generate_edit_episode, make_counterfactual_pair, resource_rule
from .features import dataset_tensors
from .losses import bottleneck_loss, counterfactual_loss, edit_pressure_loss, prediction_loss
from .models import ExplicitTableOracle, make_model


def train_model(model_name: str, seed: int, config: dict):
    torch.manual_seed(seed)
    random.seed(seed)
    model = make_model(model_name, int(config["hidden_dim"]), int(config["bottleneck_dim"]))
    if isinstance(model, ExplicitTableOracle):
        return model

    train_records = generate_dataset(int(config["n_train"]), seed, regime="base", shortcut_mode="train")
    x, y = dataset_tensors(train_records)
    loader = DataLoader(TensorDataset(x, y), batch_size=int(config["batch_size"]), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    loss_config = config["loss"]

    if model_name == "edit_pressure_training":
        edit_steps = max(int(config["epochs"]) * 8, 240)
        rng = random.Random(seed + 90_000)
        for epoch in range(edit_steps):
            optimizer.zero_grad()
            episodes = []
            for i in range(12):
                condition = {"texture": rng.choice(TEXTURES), "wet": rng.choice(WETS)}
                current = resource_rule({**condition, "color": "red", "odor": "strong"})
                alternatives = [resource for resource in RESOURCES if resource != current]
                episodes.append(
                    generate_edit_episode(
                        seed * 10_000 + epoch * 100 + i,
                        edit_condition=condition,
                        new_outcome=rng.choice(alternatives),
                    )
                )
            loss = edit_pressure_loss(
                model,
                episodes,
                float(loss_config["lambda_local"]),
                float(loss_config["lambda_cons"]),
            )
            loss.backward()
            optimizer.step()
        return model

    for epoch in range(int(config["epochs"])):
        for xb, yb in loader:
            optimizer.zero_grad()
            if model_name == "prediction_bottleneck":
                z = model.hidden(xb)
                loss = bottleneck_loss(model.forward_from_hidden(z), yb, z, float(loss_config["beta_bottleneck"]))
            elif model_name == "counterfactual_training":
                pairs_n = [make_counterfactual_pair(record, "nuisance_change") for record in train_records[: len(xb)]]
                pairs_r = [make_counterfactual_pair(record, "relation_change") for record in train_records[: len(xb)]]
                nuisance_x, _ = dataset_tensors([right for _, right in pairs_n])
                relation_x, relation_y = dataset_tensors([right for _, right in pairs_r])
                loss = counterfactual_loss(
                    model,
                    (xb, yb, nuisance_x[: len(xb)], relation_x[: len(xb)], relation_y[: len(xb)]),
                    float(loss_config["lambda_inv"]),
                    float(loss_config["lambda_cf"]),
                )
            else:
                loss = prediction_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model
