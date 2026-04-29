from src.features import enumerate_candidate_conditions, matches_condition, one_hot_context


def test_one_hot_length():
    ctx = {"texture": "A", "wet": "dry", "color": "color_00", "odor": "odor_00"}
    assert len(one_hot_context(ctx)) == 37
    assert one_hot_context(ctx).sum() == 4


def test_matches_condition():
    ctx = {"texture": "A", "wet": "dry", "color": "color_00", "odor": "odor_00"}
    assert matches_condition(ctx, {"texture": "A"})
    assert matches_condition(ctx, {"texture": "A", "wet": "dry"})
    assert not matches_condition(ctx, {"texture": "B"})


def test_enumerate_conditions():
    conditions = enumerate_candidate_conditions(max_order=2)
    assert {"texture": "A"} in conditions
    assert {"texture": "A", "wet": "dry"} in conditions
