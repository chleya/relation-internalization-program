import csv
import json
from pathlib import Path

from src.b6_hardening.hardening_metrics import SUMMARY_FIELDS
from src.b6_hardening.hardening_runner import run_b6_1_hardening, write_b6_1_outputs
from src.visualize_b6_1 import plot_summary


def test_b6_1_runner_outputs_csv_json_and_plot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "b6_1": {
            "grid_size": 8,
            "episodes_per_condition": 3,
            "noise_rates": [0.0, 0.4],
            "missing_rates": [0.0, 1.0],
            "delay_steps": [1],
            "inspect_costs": [0.0, 0.2],
            "spurious_modes": ["clean_correlated", "hard_flipped"],
        }
    }
    summary, records, metrics = run_b6_1_hardening(config, seed=0)
    write_b6_1_outputs(summary, records, metrics)
    assert Path("results/b6_1_hardening_summary.csv").exists()
    assert Path("results/b6_1_hardening_metrics.json").exists()
    assert Path("reports/B6_1_REVIEWER_HARDENING_REPORT.md").exists()
    with Path("results/b6_1_hardening_summary.csv").open("r", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    for field in SUMMARY_FIELDS:
        assert field in row
    parsed = json.loads(Path("results/b6_1_hardening_metrics.json").read_text(encoding="utf-8"))
    assert parsed["summary_rows"] == len(summary)
    plot_summary(summary, "results/b6_1_hardening_plot.png")
    assert Path("results/b6_1_hardening_plot.png").exists()

