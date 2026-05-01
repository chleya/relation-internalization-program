from src.b5_closed_loop_env import inspect_region, make_b5_closed_loop_episode
from src.b5_trace_update import export_trace_state, trace_update_accuracy, update_trace_after_inspection


def test_trace_update_after_correct_inspection():
    episode = make_b5_closed_loop_episode({}, 0, "inspect_needed", "recurrent")
    before = export_trace_state(None, episode, {})
    result = inspect_region(episode, episode["ground_truth"]["oracle_inspect_region"], {})
    after = update_trace_after_inspection(before, result, episode, {})
    assert trace_update_accuracy(after, episode) == 1.0
    assert after["uncertainty"] < before["uncertainty"]
