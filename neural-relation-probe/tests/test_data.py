from src.data import make_dataset, resource_for


def test_resource_rule():
    assert resource_for("A", "dry") == "food"
    assert resource_for("A", "wet") == "poison"
    assert resource_for("B", "dry") == "poison"
    assert resource_for("C", "wet") == "neutral"


def test_dataset_shapes():
    data = make_dataset(10, seed=0)
    assert data.x.shape == (10, 9)
    assert data.y_resource.shape == (10,)
    assert data.y_relation.shape == (10,)
    assert data.y_nuisance.shape == (10,)


def test_shortcut_mode_is_deterministic():
    data = make_dataset(30, seed=0, mode="shortcut")
    by_resource = {}
    for context, resource in zip(data.contexts, data.y_resource):
        cue = (context["color"], context["odor"])
        by_resource.setdefault(int(resource), cue)
        assert by_resource[int(resource)] == cue
