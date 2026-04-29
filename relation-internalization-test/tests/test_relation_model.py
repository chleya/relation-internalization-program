from src.relation_model import RelationInternalizationModel, RobustWideRelationInternalizationModel, WideRelationInternalizationModel
from src.evaluate import evaluate_relation_table_alignment
from src.metrics import gated_internalization_score


def test_relation_model_updates_rules_and_infers():
    model = RelationInternalizationModel(min_support=1)
    ctx = {"texture": "A", "wet": "dry", "color": "color_00", "odor": "odor_00"}
    for _ in range(5):
        model.observe(ctx, "eat", "food", 1.0)
    assert model.infer_resource(ctx) == "food"
    assert model.act(ctx) == "eat"
    relations = model.describe_relations()
    assert any(r["condition"] == {"texture": "A", "wet": "dry"} for r in relations)


def test_shuffle_rule_outcomes_changes_internal_table():
    model = RelationInternalizationModel(min_support=1)
    examples = [
        ({"texture": "A", "wet": "dry", "color": "color_00", "odor": "odor_00"}, "food"),
        ({"texture": "A", "wet": "wet", "color": "color_01", "odor": "odor_01"}, "poison"),
        ({"texture": "B", "wet": "dry", "color": "color_01", "odor": "odor_01"}, "poison"),
        ({"texture": "C", "wet": "dry", "color": "color_02", "odor": "odor_02"}, "neutral"),
    ]
    for context, resource in examples:
        for _ in range(5):
            model.observe(context, "avoid", resource, 0.0)
    before = [model.infer_resource(context) for context, _ in examples]
    assert model.shuffle_rule_outcomes(seed=3)
    after = [model.infer_resource(context) for context, _ in examples]
    assert before != after


def test_relation_table_alignment_reads_exact_rules():
    model = RelationInternalizationModel(min_support=1)
    resources = {
        ("A", "dry"): "food",
        ("A", "wet"): "poison",
        ("B", "dry"): "poison",
        ("B", "wet"): "poison",
        ("C", "dry"): "neutral",
        ("C", "wet"): "neutral",
    }
    for (texture, wet), resource in resources.items():
        context = {"texture": texture, "wet": wet, "color": "color_00", "odor": "odor_00"}
        for _ in range(3):
            model.observe(context, "avoid", resource, 0.0)
    audit = evaluate_relation_table_alignment(model)
    assert audit["relation_table_coverage"] == 1.0
    assert audit["relation_table_accuracy"] == 1.0
    assert audit["relation_table_alignment"] == 1.0


def test_wide_relation_can_expose_texture_wet_rules():
    model = WideRelationInternalizationModel(min_support=1)
    resources = {
        ("A", "dry"): "food",
        ("A", "wet"): "poison",
        ("B", "dry"): "poison",
        ("B", "wet"): "poison",
        ("C", "dry"): "neutral",
        ("C", "wet"): "neutral",
    }
    for (texture, wet), resource in resources.items():
        for color in ["color_00", "color_04"]:
            context = {"texture": texture, "wet": wet, "color": color, "odor": "odor_00"}
            for _ in range(3):
                model.observe(context, "avoid", resource, 0.0)
    audit = evaluate_relation_table_alignment(model)
    assert audit["relation_table_alignment"] == 1.0


def test_robust_wide_downweights_spurious_rules():
    model = RobustWideRelationInternalizationModel(min_support=1)
    for _ in range(8):
        model.observe({"texture": "A", "wet": "dry", "color": "color_00", "odor": "odor_00"}, "eat", "food", 1.0)
        model.observe({"texture": "A", "wet": "wet", "color": "color_01", "odor": "odor_01"}, "avoid", "poison", 0.2)
        model.observe({"texture": "B", "wet": "dry", "color": "color_01", "odor": "odor_01"}, "avoid", "poison", 0.2)
        model.observe({"texture": "C", "wet": "dry", "color": "color_02", "odor": "odor_02"}, "avoid", "neutral", 0.0)
    attack_context = {"texture": "A", "wet": "dry", "color": "color_01", "odor": "odor_01"}
    assert model.infer_resource(attack_context) == "food"


def test_gated_internalization_score_blocks_missing_gates():
    passing = {
        "ood_success": 1.0,
        "spurious_resource_accuracy": 1.0,
        "counterfactual_accuracy": 1.0,
        "edit_resource_success": 1.0,
        "edit_reversal_success": 1.0,
        "relation_table_alignment": 1.0,
        "relation_shuffle_drop": 0.5,
    }
    assert gated_internalization_score(passing) > 0.0
    failing = dict(passing)
    failing["spurious_resource_accuracy"] = 0.5
    assert gated_internalization_score(failing) == 0.0
