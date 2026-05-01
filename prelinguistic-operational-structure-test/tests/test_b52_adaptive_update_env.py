from src.b5_clean_episode_view import assert_model_input_is_sanitized
from src.b52_adaptive_update_env import inspection_content_signature, make_b52_adaptive_update_episode_pair, observation_signature


def test_b52_pair_has_same_initial_and_different_content():
    pair = make_b52_adaptive_update_episode_pair({}, 0, "same_initial_different_inspection")
    a = pair["episode_a"]
    b = pair["episode_b"]
    assert observation_signature(a) == observation_signature(b)
    assert inspection_content_signature(a) != inspection_content_signature(b)
    assert "evaluator_ground_truth" not in a["model_input"]
    assert_model_input_is_sanitized(a["model_input"], {})
