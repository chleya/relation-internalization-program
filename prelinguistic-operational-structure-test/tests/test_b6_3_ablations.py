from src.b6_3_structural_necessity.ablations import apply_ablation
from src.b6_3_structural_necessity.env import make_b63_episode
from src.b6_3_structural_necessity.structural_policy import b63_policy


def test_remove_trace_does_not_crash_policy():
    episode = make_b63_episode({"b6_3": {"grid_size": 8}}, 0, "wrong_trace")
    ablated = apply_ablation(episode, "remove_trace", {})
    output = b63_policy(ablated, {})
    assert output["policy_name"] == "b63_policy"
    assert output["trace_conflict"]["trace_conflict_detected"] is True


def test_disable_candidate_search_hurts_hidden_indirect_path():
    episode = make_b63_episode({"b6_3": {"grid_size": 8}}, 0, "hide_indirect_target")
    ablated = apply_ablation(episode, "disable_candidate_search", {})
    output = b63_policy(ablated, {})
    assert output["action"] is None
    assert output["failure_reason"] in {"no_indirect_candidate", "delayed_backfire_risk", "irreversible_risk"}


def test_disable_credit_buffer_prevents_credit_assignment():
    episode = make_b63_episode({"b6_3": {"grid_size": 8}}, 2003, "delayed_indirect_delay5")
    output = b63_policy(episode, {})
    ablated = apply_ablation(episode, "disable_delayed_credit_buffer", {})
    disabled = b63_policy(ablated, {})
    assert output["delayed_credit"]["credit_assigned"] is True
    assert disabled["delayed_credit"]["credit_assigned"] is False
