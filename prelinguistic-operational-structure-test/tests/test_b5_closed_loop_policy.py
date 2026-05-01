from src.b5_closed_loop_env import make_b5_closed_loop_episode
from src.b5_closed_loop_policy import closed_loop_policy
from src.models import make_model


def test_closed_loop_policy_returns_trace_update_and_action():
    model = make_model("recurrent_flow_checkpoint_model")
    episode = make_b5_closed_loop_episode({}, 0, "inspect_needed", "recurrent")
    output = closed_loop_policy(model, episode, {})
    assert output["inspect_chosen"] is True
    assert output["trace_after_inspection"]["region"] == episode["ground_truth"]["true_trace_region"]
    assert "intervention_action" in output
    assert output["provenance"]["oracle_closed_loop_plan_used"] is False
