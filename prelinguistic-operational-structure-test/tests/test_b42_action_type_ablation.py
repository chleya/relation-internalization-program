from __future__ import annotations

from src.b4_action_space import is_valid_action
from src.b42_action_type_ablation import ablate_action_type_scorer, evaluate_action_type_ablation
from src.b42_action_type_env import make_action_type_specific_episode
from src.b42_action_type_policy import action_type_disambiguating_policy
from src.models import make_model
from tests.conftest import small_config


def test_action_type_ablation_keeps_valid_region_and_metrics():
    config = small_config()
    episode = make_action_type_specific_episode(config, 19, "recurrent")
    model = make_model("recurrent_flow_checkpoint_model")
    ablated = ablate_action_type_scorer(model, episode, config)
    action = action_type_disambiguating_policy(model, ablated, config)["action"]
    assert is_valid_action(action, config)
    metrics, records = evaluate_action_type_ablation(model, [episode], config)
    assert records
    assert metrics["action_type_shift_after_ablation"] >= 0.0

