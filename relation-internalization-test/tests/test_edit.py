from src.features import all_contexts
from src.evaluate import evaluate_edit
from src.relation_model import RelationInternalizationModel


def train_base_relation_model():
    model = RelationInternalizationModel(min_support=1)
    for color in ["color_00", "color_04"]:
        for odor in ["odor_00", "odor_04"]:
            model.observe({"texture": "A", "wet": "dry", "color": color, "odor": odor}, "eat", "food", 1.0)
            model.observe({"texture": "A", "wet": "wet", "color": color, "odor": odor}, "avoid", "poison", 0.2)
            model.observe({"texture": "B", "wet": "dry", "color": color, "odor": odor}, "avoid", "poison", 0.2)
            model.observe({"texture": "C", "wet": "dry", "color": color, "odor": odor}, "avoid", "neutral", 0.0)
    return model


def test_edit_rule_changes_target_and_preserves_non_target():
    model = train_base_relation_model()
    target = {"texture": "A", "wet": "dry", "color": "color_00", "odor": "odor_00"}
    non_targets = [ctx for ctx in all_contexts() if not (ctx["texture"] == "A" and ctx["wet"] == "dry")]
    before_non_target = [model.act(ctx) for ctx in non_targets]
    assert model.act(target) == "eat"
    assert model.edit_rule({"texture": "A", "wet": "dry"}, "poison")
    assert model.act(target) == "avoid"
    after_non_target = [model.act(ctx) for ctx in non_targets]
    unchanged = sum(1 for before, after in zip(before_non_target, after_non_target) if before == after)
    assert unchanged / len(non_targets) >= 0.9


def test_edit_evaluation_checks_resource_change():
    model = train_base_relation_model()
    result = evaluate_edit(model, seed=0)
    assert result["edit_success"] == 1.0
    assert result["edit_resource_success"] == 1.0
    assert result["edit_resource_locality"] >= 0.9
