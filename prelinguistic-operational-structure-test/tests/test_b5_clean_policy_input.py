import pytest

from src.b5_clean_episode_view import split_b5_episode_for_clean_run
from src.b5_clean_leakage_guard import guard_policy_output
from src.b5_clean_runner import clean_closed_loop_policy
from src.b5_closed_loop_env import make_b5_closed_loop_episode


def test_clean_policy_accepts_only_model_input_shape():
    raw = make_b5_closed_loop_episode({}, 0, "inspect_needed", "recurrent")
    bundle = split_b5_episode_for_clean_run(raw, {}, episode_id=0)
    output = clean_closed_loop_policy(object(), bundle["model_input"], {})
    assert output["provenance"]["oracle_plan_used"] is False
    assert guard_policy_output(output, {})["policy_output_oracle_usage_rate"] == 0.0


def test_clean_policy_does_not_accept_evaluator_argument():
    raw = make_b5_closed_loop_episode({}, 0, "inspect_needed", "recurrent")
    bundle = split_b5_episode_for_clean_run(raw, {}, episode_id=0)
    with pytest.raises(TypeError):
        clean_closed_loop_policy(object(), bundle["model_input"], bundle["evaluator_ground_truth"], {})
