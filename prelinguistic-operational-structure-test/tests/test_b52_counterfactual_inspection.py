from src.b5_clean_episode_view import assert_model_input_is_sanitized
from src.b52_adaptive_update_env import make_b52_adaptive_update_episode_pair
from src.b52_counterfactual_inspection import evaluate_counterfactual_inspection_update, make_counterfactual_inspection_observation
from src.models import make_model


def test_counterfactual_observation_generated_without_oracle_keys():
    bundle = make_b52_adaptive_update_episode_pair({}, 2, "counterfactual_inspection")["episode_a"]
    obs = make_counterfactual_inspection_observation(bundle, "trace_shifted", {})
    bundle["model_input"]["inspection_observation"] = obs
    assert "oracle_value" not in obs
    assert_model_input_is_sanitized(bundle["model_input"], {})


def test_counterfactual_update_metrics_run():
    model = make_model("field_memory_model")
    pair = make_b52_adaptive_update_episode_pair({}, 2, "counterfactual_inspection")
    metrics, records = evaluate_counterfactual_inspection_update(model, [pair["episode_a"]], {})
    assert 0.0 <= metrics["counterfactual_update_switch_rate"] <= 1.0
    assert records
