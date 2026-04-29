from __future__ import annotations

import torch

from src.data import generate_dataset
from src.models import make_model
from src.nonlinear_probes import probe_selectivity, run_nonlinear_probe_suite, shuffled_label_control


def test_shuffled_label_control_changes_labels():
    labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    shuffled = shuffled_label_control(labels, seed=0)
    assert shuffled.shape == labels.shape
    assert sorted(shuffled.tolist()) == sorted(labels.tolist())
    assert not torch.equal(shuffled, labels)


def test_probe_selectivity_computation():
    assert probe_selectivity(0.9, 0.4) == 0.5


def test_nonlinear_probe_suite_returns_expected_keys():
    model = make_model("pure_prediction", hidden_dim=8, bottleneck_dim=4)
    dataset = generate_dataset(24, seed=10, regime="base", shortcut_mode="none")
    result = run_nonlinear_probe_suite(model, {"probe": dataset}, seed=0)
    expected = {
        "linear_relation_acc",
        "mlp_relation_acc",
        "tree_relation_acc",
        "linear_nuisance_acc",
        "mlp_nuisance_acc",
        "tree_nuisance_acc",
        "linear_relation_selectivity",
        "mlp_relation_selectivity",
        "tree_relation_selectivity",
        "nonlinear_relation_gain",
        "nonlinear_nuisance_gain",
    }
    assert expected.issubset(result.keys())
