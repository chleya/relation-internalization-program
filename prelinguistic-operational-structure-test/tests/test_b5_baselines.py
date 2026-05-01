from src.b5_closed_loop_baselines import evaluate_b5_baselines
from src.b5_closed_loop_env import make_b5_closed_loop_episode


def test_b5_baselines_run_and_oracle_beats_random():
    episodes = [make_b5_closed_loop_episode({}, idx, "inspect_needed", "recurrent") for idx in range(4)]
    metrics, records = evaluate_b5_baselines(episodes, {}, seed=0)
    assert records
    assert metrics["oracle_closed_loop_score"] >= metrics["random_closed_loop_score"]
