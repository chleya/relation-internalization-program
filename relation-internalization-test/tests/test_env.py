from src.env import RelationWorld
from src.evaluate import adversarial_spurious_contexts


def test_base_resources_and_rewards():
    env = RelationWorld("base", seed=0)
    food_ctx = {"texture": "A", "wet": "dry", "color": "color_00", "odor": "odor_00"}
    poison_ctx = {"texture": "B", "wet": "dry", "color": "color_01", "odor": "odor_01"}
    neutral_ctx = {"texture": "C", "wet": "wet", "color": "color_02", "odor": "odor_02"}
    assert env.get_resource(food_ctx) == "food"
    assert env.get_resource(poison_ctx) == "poison"
    assert env.get_resource(neutral_ctx) == "neutral"
    assert env.step("eat", food_ctx)["reward"] == 1.0
    assert env.step("eat", poison_ctx)["reward"] == -1.0
    assert env.step("avoid", poison_ctx)["reward"] == 0.2


def test_reversal_resources():
    env = RelationWorld("reversal", seed=0)
    assert env.get_resource({"texture": "A", "wet": "dry", "color": "color_00", "odor": "odor_00"}) == "poison"
    assert env.get_resource({"texture": "B", "wet": "wet", "color": "color_01", "odor": "odor_01"}) == "food"


def test_ood_uses_heldout_spurious_features():
    env = RelationWorld("ood", seed=0)
    ctx = env.sample_context()
    assert ctx["color"] not in {"color_00", "color_01", "color_02", "color_03"}
    assert ctx["odor"] not in {"odor_00", "odor_01", "odor_02", "odor_03"}


def test_reversal_spurious_features_do_not_reveal_reversed_resource():
    env = RelationWorld("reversal", seed=0)
    ctx = {"texture": "A", "wet": "dry", "color": "color_00", "odor": "odor_00"}
    assert env.get_resource(ctx) == "poison"
    sampled = None
    for _ in range(100):
        candidate = env.sample_context()
        if candidate["texture"] == "A" and candidate["wet"] == "dry":
            sampled = candidate
            break
    assert sampled is not None
    assert sampled["color"] == "color_00"
    assert sampled["odor"] == "odor_00"


def test_adversarial_spurious_contexts_flip_surface_cues():
    env = RelationWorld("base", seed=0)
    contexts = adversarial_spurious_contexts(env)
    food_attack = [ctx for ctx in contexts if ctx["texture"] == "A" and ctx["wet"] == "dry"]
    poison_attack = [ctx for ctx in contexts if ctx["texture"] == "B"]
    assert any(ctx["color"] == "color_01" and ctx["odor"] == "odor_01" for ctx in food_attack)
    assert any(ctx["color"] == "color_00" and ctx["odor"] == "odor_00" for ctx in poison_attack)
