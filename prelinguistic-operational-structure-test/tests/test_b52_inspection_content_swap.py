from src.b5_clean_episode_view import assert_model_input_is_sanitized
from src.b52_adaptive_update_env import make_b52_adaptive_update_episode_pair
from src.b52_inspection_content_swap import evaluate_inspection_content_swap, swap_inspection_content
from src.models import make_model


def test_swap_inspection_content_keeps_evaluator_ground_truth_fixed():
    pair = make_b52_adaptive_update_episode_pair({}, 1, "inspection_content_swap")
    a = pair["episode_a"]
    b = pair["episode_b"]
    swapped_a, swapped_b = swap_inspection_content(a, b, {})
    assert swapped_a["evaluator_ground_truth"] == a["evaluator_ground_truth"]
    assert swapped_b["evaluator_ground_truth"] == b["evaluator_ground_truth"]
    assert swapped_a["model_input"]["inspection_observation"] == b["model_input"]["inspection_observation"]
    assert_model_input_is_sanitized(swapped_a["model_input"], {})


def test_inspection_content_swap_metric_runs():
    model = make_model("recurrent_flow_checkpoint_model")
    pair = make_b52_adaptive_update_episode_pair({}, 1, "inspection_content_swap")
    metrics, records = evaluate_inspection_content_swap(model, [pair], {})
    assert 0.0 <= metrics["inspection_swap_update_change_rate"] <= 1.0
    assert records
