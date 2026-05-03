from pathlib import Path
import inspect

from src.g_line.g1_env import CONDITIONS, make_g1_datasets
from src.g_line import g1_generator
from src.g_line.g1_adversarial_review import build_g1_adversarial_review
from src.g_line.g1_generator import choose_action, fit_generator
from src.g_line.g1_runner import run_g1, write_g1_outputs
from src.g_line.g1_1_runner import run_g1_1, write_g1_1_outputs


def test_g1_datasets_have_expected_splits():
    datasets = make_g1_datasets({"g1": {"train_episodes": 2, "test_episodes": 1, "ood_episodes": 1}}, seed=0)
    assert set(datasets) == set(CONDITIONS)
    assert datasets["train"][0]["model_input"]["interaction_history"]


def test_g1_generator_does_not_need_evaluator_labels():
    datasets = make_g1_datasets({"g1": {"train_episodes": 2, "test_episodes": 1, "ood_episodes": 1}}, seed=0)
    artifact = fit_generator([episode["model_input"] for episode in datasets["train"]], {"g1": {}})
    output = choose_action(datasets["test"][0]["model_input"], artifact)
    assert output["generated_mask"]
    assert output["provenance"]["uses_evaluator_labels"] is False


def test_g1_runner_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {"g1": {"train_episodes": 4, "test_episodes": 2, "ood_episodes": 2, "regions_per_episode": 8}}
    summary, records, metrics, _ = run_g1(config, seed=0)
    write_g1_outputs(summary, records, metrics)
    assert Path("results/g1_minimal/summary.csv").exists()
    assert Path("results/g1_minimal/records.csv").exists()
    assert Path("results/g1_minimal/metrics.json").exists()
    assert Path("reports/G1_MINIMAL_GENERATOR_RESULTS.md").exists()
    assert metrics["invalid_metric_count_total"] == 0
    assert metrics["search_size"] > 0
    assert metrics["g1_oracle_gap"] >= 0.0


def test_g1_generator_source_does_not_read_evaluator_or_oracle():
    source = inspect.getsource(g1_generator)
    forbidden = ["evaluator_ground_truth", "oracle_baseline_view", "expected_decision"]
    assert not any(token in source for token in forbidden)


def test_g1_adversarial_review_records_pressure_gap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {"g1": {"train_episodes": 4, "test_episodes": 2, "ood_episodes": 2, "regions_per_episode": 8}}
    summary, records, metrics, _ = run_g1(config, seed=0)
    write_g1_outputs(summary, records, metrics)
    review = build_g1_adversarial_review()
    assert review["forbidden_reference_count"] == 0
    assert review["oracle_gap_positive"] is True
    assert "remaining_blockers" in review


def test_g1_1_pressure_hardening_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {"g1_1": {"episodes_per_condition": 3, "regions_per_episode": 8}}
    summary, records, metrics, _ = run_g1_1(config, seed=0)
    write_g1_1_outputs(summary, records, metrics)
    assert Path("results/g1_1_pressure_hardening/summary.csv").exists()
    assert Path("reports/G1_1_PRESSURE_USE_HARDENING_RESULTS.md").exists()
    assert records
    assert metrics["invalid_metric_count_total"] == 0
    assert metrics["generated_rule_pressure_coverage"] == 2
    assert metrics["feedback_pressure_gain"] > 0.0
    assert metrics["compression_pressure_gain"] > 0.0
