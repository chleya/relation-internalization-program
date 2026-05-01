from src.b5_closed_loop_env import make_b5_closed_loop_episode
from src.b5_epistemic_pragmatic_values import compute_epistemic_value, compute_pragmatic_value, compute_wrong_inspect_penalty


def test_epistemic_value_prefers_oracle_inspect_region():
    episode = make_b5_closed_loop_episode({}, 0, "inspect_needed", "recurrent")
    gt = episode["ground_truth"]
    assert compute_epistemic_value(episode, gt["oracle_inspect_region"], {}) > compute_epistemic_value(episode, gt["wrong_inspect_region"], {})


def test_pragmatic_value_prefers_oracle_action():
    episode = make_b5_closed_loop_episode({}, 0, "inspect_needed", "recurrent")
    gt = episode["ground_truth"]
    wrong = dict(gt["oracle_intervention_action"])
    wrong["region_id"] = gt["wrong_inspect_region"]
    assert compute_pragmatic_value(episode, gt["oracle_intervention_action"], {}) > compute_pragmatic_value(episode, wrong, {})
    assert compute_wrong_inspect_penalty(episode, {}) >= 0.0
