from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .failure_localization import (
    build_failure_report,
    extraction_subspace_correlation,
    gate_failure_matrix,
    intervention_site_analysis,
    read_records,
    subspace_drop_by_seed,
    write_csv,
)


def run(config_path: str = "configs/sweep.yaml") -> None:
    sweep = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(sweep["base_config"]).read_text(encoding="utf-8"))
    records = read_records("results/records.csv")
    seeds = [int(seed) for seed in sweep["seeds"]]
    gate_rows = gate_failure_matrix(records, config["gates"])
    drop_rows = subspace_drop_by_seed(records)
    site_rows = intervention_site_analysis(config, seeds)
    corr_rows = extraction_subspace_correlation(records)
    write_csv("results/gate_failure_matrix.csv", gate_rows)
    write_csv("results/subspace_drop_by_seed.csv", drop_rows)
    write_csv("results/intervention_site_analysis.csv", site_rows)
    write_csv("results/extraction_subspace_correlation.csv", corr_rows)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/EDIT_PRESSURE_FAILURE_ANALYSIS.md").write_text(
        build_failure_report(gate_rows, drop_rows, site_rows, corr_rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sweep.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
