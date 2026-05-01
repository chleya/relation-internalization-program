from src.b51_feedback_revision_specificity import scripted_feedback_revision_baseline
from src.b5_closed_loop_env import make_b5_closed_loop_episode


def test_scripted_feedback_revision_baseline_runs():
    episode = make_b5_closed_loop_episode({}, 0, "inspect_needed", "recurrent")
    revised = scripted_feedback_revision_baseline(episode, {"action_success": True}, {})
    assert revised["region"] == episode["ground_truth"]["true_trace_region"]
