from src.data import make_dataset
from src.model import hidden_states, train_mlp
from src.probe import probe_accuracy, subspace_intervention_drop, train_probe


def test_probe_runs():
    data = make_dataset(200, seed=0)
    model = train_mlp(data.x, data.y_resource, seed=0)
    hidden = hidden_states(model, data.x)
    probe = train_probe(hidden, data.y_relation, seed=0)
    assert probe_accuracy(probe, hidden, data.y_relation) > 0.5


def test_subspace_intervention_runs():
    data = make_dataset(200, seed=1)
    model = train_mlp(data.x, data.y_resource, seed=1)
    hidden = hidden_states(model, data.x)
    probe = train_probe(hidden, data.y_relation, seed=1)
    drop = subspace_intervention_drop(model, data.x, data.y_resource, probe)
    assert -1.0 <= drop <= 1.0
