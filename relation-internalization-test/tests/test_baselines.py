from src.baselines import DecisionTreeBaseline


def test_decision_tree_baseline_learns_resource_predictions():
    model = DecisionTreeBaseline(seed=0)
    examples = [
        ({"texture": "A", "wet": "dry", "color": "color_00", "odor": "odor_00"}, "food"),
        ({"texture": "A", "wet": "wet", "color": "color_01", "odor": "odor_01"}, "poison"),
        ({"texture": "B", "wet": "dry", "color": "color_01", "odor": "odor_01"}, "poison"),
        ({"texture": "C", "wet": "dry", "color": "color_02", "odor": "odor_02"}, "neutral"),
    ]
    for context, resource in examples * 4:
        model.observe(context, "avoid", resource, 0.0)
    assert model.infer_resource(examples[0][0]) == "food"
    assert model.describe_relations()
