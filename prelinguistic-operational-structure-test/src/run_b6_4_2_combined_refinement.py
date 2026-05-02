from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .b6_4_2_combined_remap_refinement.runner import run_b6_4_2_combined_refinement, write_b6_4_2_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b6_4_2_combined_refinement.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records, metrics = run_b6_4_2_combined_refinement(config, seed=args.seed)
    write_b6_4_2_outputs(summary, records, metrics)
    print(f"combined_refinement_score={metrics['combined_refinement_score']:.3f}")
    print(f"combined_policy_score={metrics['combined_policy_score']:.3f}")
    print(f"combined_oracle_gap={metrics['combined_oracle_gap']:.3f}")
    print(f"combined_failure_source={metrics['combined_failure_source']}")


if __name__ == "__main__":
    main()
