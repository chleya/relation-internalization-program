from src.b51_baseline_sanity import run_b51_baseline_sanity_check
from src.b5_closed_loop_env import make_b5_closed_loop_episode


def test_b51_baseline_sanity_runs():
    episodes = [make_b5_closed_loop_episode({}, idx, "inspect_needed", "recurrent") for idx in range(4)]
    metrics, records = run_b51_baseline_sanity_check(episodes, {}, model_score=1.0)
    assert records
    assert metrics["oracle_closed_loop_score"] >= metrics["random_closed_loop_score"]
