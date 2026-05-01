import yaml

from src.b5_clean_runner import run_b5_clean_closed_loop


def test_run_b5_clean_closed_loop_executes():
    config = yaml.safe_load(open("configs/b5_clean_closed_loop.yaml", encoding="utf-8"))
    config["b5_clean"]["max_eval_episodes"] = 4
    summary, records, leakage = run_b5_clean_closed_loop(config, seed=0)
    assert summary
    assert records
    assert sum(int(row["leakage_count"]) for row in leakage) == 0
