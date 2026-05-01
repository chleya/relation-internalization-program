from src.b6_hardening.hidden_risk_env import make_hardening_episode
from src.b6_hardening.inspect_cost import inspect_cost_metrics, should_inspect


def test_should_inspect_respects_need_hint_and_cost():
    model_input = {"visible_state": {"inspect_need_hint": True}}
    assert should_inspect(model_input, 0.1)
    assert not should_inspect({"visible_state": {"inspect_need_hint": False}}, 0.1)


def test_inspect_cost_episode_marks_cost():
    episode = make_hardening_episode("inspect_cost", 0, {"b6_1": {"grid_size": 8}}, inspect_cost=0.2)
    assert episode["model_input"]["cost_config_public"]["inspection_cost"] == 0.2


def test_inspect_cost_metrics_computes_over_inspection():
    records = [
        {"requires_inspect": 1, "unnecessary_inspect": 0, "inspected": 1, "over_inspection": 0, "risk_constrained_score": 1.0},
        {"requires_inspect": 0, "unnecessary_inspect": 1, "inspected": 1, "over_inspection": 1, "risk_constrained_score": 0.5},
    ]
    metrics = inspect_cost_metrics(records)
    assert metrics["required_inspect_recall"] == 1.0
    assert metrics["over_inspection_rate"] == 0.5

