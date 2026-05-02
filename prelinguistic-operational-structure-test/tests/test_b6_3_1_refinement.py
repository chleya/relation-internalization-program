from src.b6_3_1_refinement.runner import make_b631_episode, policy_for, run_b6_3_1_refinement
from src.b6_3_1_refinement.metrics import score_output


def test_state_only_drops_on_wrong_trace_no_public_state():
    config = {"b6_3_1": {"episodes_per_condition": 2, "conditions": ["wrong_trace_no_public_state"]}}
    summary, _records, _metrics = run_b6_3_1_refinement(config, seed=0)
    b631 = row(summary, "wrong_trace_no_public_state", "b63_1_policy")
    state = row(summary, "wrong_trace_no_public_state", "state_only")
    assert float(b631["risk_constrained_score"]) > float(state["risk_constrained_score"])
    assert float(b631["state_only_drop_on_wrong_trace_no_public_state"]) > 0.20


def test_freeze_feedback_update_drops_feedback_required_split():
    config = {"b6_3_1": {"episodes_per_condition": 2, "conditions": ["feedback_required_trace_repair"]}}
    summary, _records, _metrics = run_b6_3_1_refinement(config, seed=0)
    b631 = row(summary, "feedback_required_trace_repair", "b63_1_policy")
    frozen = row(summary, "feedback_required_trace_repair", "b63_1_no_feedback_update")
    assert float(b631["risk_constrained_score"]) > float(frozen["risk_constrained_score"])
    assert float(b631["drop_under_freeze_feedback_update_on_feedback_required"]) > 0.20


def test_remove_history_drops_history_required_split():
    config = {"b6_3_1": {"episodes_per_condition": 2, "conditions": ["history_required_indirect_discovery"]}}
    summary, _records, _metrics = run_b6_3_1_refinement(config, seed=0)
    b631 = row(summary, "history_required_indirect_discovery", "b63_1_policy")
    no_history = row(summary, "history_required_indirect_discovery", "b63_1_no_history")
    assert float(b631["risk_constrained_score"]) > float(no_history["risk_constrained_score"])
    assert float(b631["drop_under_remove_history_on_history_required"]) > 0.20


def test_disable_credit_buffer_drops_delay5_required_split():
    config = {"b6_3_1": {"episodes_per_condition": 2, "conditions": ["delay5_credit_buffer_required"]}}
    summary, _records, _metrics = run_b6_3_1_refinement(config, seed=0)
    b631 = row(summary, "delay5_credit_buffer_required", "b63_1_policy")
    no_buffer = row(summary, "delay5_credit_buffer_required", "b63_1_no_credit_buffer")
    assert float(b631["risk_constrained_score"]) > float(no_buffer["risk_constrained_score"])
    assert float(b631["drop_under_disable_credit_buffer_on_delay5_required"]) > 0.20


def test_disable_candidate_search_drops_hidden_indirect_split():
    config = {"b6_3_1": {"episodes_per_condition": 2, "conditions": ["hidden_indirect_no_candidate_shortcut"]}}
    summary, _records, _metrics = run_b6_3_1_refinement(config, seed=0)
    b631 = row(summary, "hidden_indirect_no_candidate_shortcut", "b63_1_policy")
    no_search = row(summary, "hidden_indirect_no_candidate_shortcut", "b63_1_no_candidate_search")
    assert float(b631["risk_constrained_score"]) > float(no_search["risk_constrained_score"])
    assert float(b631["hidden_indirect_discovery_score"]) > 0.50


def test_poisoned_evaluator_does_not_change_b631_output():
    config = {}
    episode = make_b631_episode(config, 0, "wrong_trace_feedback_repair_required")
    policy = policy_for("b63_1_policy")
    original = policy(episode, config)
    poisoned = {
        **episode,
        "evaluator_ground_truth": {**episode["evaluator_ground_truth"], "expected_action": None, "condition": "poisoned"},
        "metadata": {**episode["metadata"], "condition": "poisoned"},
    }
    changed = policy(poisoned, config)
    assert comparable(original) == comparable(changed)


def test_wrong_action_does_not_receive_full_score():
    episode = make_b631_episode({}, 0, "wrong_trace_no_public_state")
    wrong_output = {"inspect": False, "action": {"action_type": "apply_local_damping", "region_id": episode["evaluator_ground_truth"]["wrong_target_region"]}}
    scored = score_output(episode, wrong_output)
    assert scored["risk_constrained_score"] < 1.0


def row(summary, condition, policy_name):
    return next(row for row in summary if row["condition"] == condition and row["policy_name"] == policy_name)


def comparable(output):
    action = output.get("action") or {}
    return action.get("action_type"), action.get("region_id"), bool(output.get("inspect")), output.get("trace_repair", {}).get("repair_source")

