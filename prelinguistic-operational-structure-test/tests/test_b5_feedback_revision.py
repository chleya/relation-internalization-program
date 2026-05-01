from src.b5_closed_loop_env import make_b5_closed_loop_episode
from src.b5_feedback_revision import feedback_revision_accuracy, observe_consequence, revise_trace_after_feedback


def test_feedback_revision_tracks_successful_consequence():
    episode = make_b5_closed_loop_episode({}, 0, "inspect_needed", "recurrent")
    action = episode["ground_truth"]["oracle_intervention_action"]
    consequence = observe_consequence(episode, action, {})
    revised = revise_trace_after_feedback(episode["trace_state"], consequence, episode, {})
    assert feedback_revision_accuracy(revised, episode) == 1.0
