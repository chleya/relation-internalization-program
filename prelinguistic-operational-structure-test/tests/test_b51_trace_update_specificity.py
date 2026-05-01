from src.b51_trace_update_specificity import scripted_trace_update_baseline
from src.b5_closed_loop_env import inspect_region, make_b5_closed_loop_episode


def test_scripted_trace_update_baseline_runs():
    episode = make_b5_closed_loop_episode({}, 0, "inspect_needed", "recurrent")
    obs = inspect_region(episode, episode["ground_truth"]["oracle_inspect_region"], {})
    updated = scripted_trace_update_baseline(episode, obs, {})
    assert updated["region"] == episode["ground_truth"]["true_trace_region"]
