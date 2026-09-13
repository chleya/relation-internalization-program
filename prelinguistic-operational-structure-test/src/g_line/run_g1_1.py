from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .g1_1_runner import run_g1_1, write_g1_1_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g1_1_pressure_hardening.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records, metrics, _ = run_g1_1(config, seed=args.seed)
    write_g1_1_outputs(summary, records, metrics)
    print(f"g1_1_mean_score={metrics['g1_1_mean_score']:.3f}")
    print(f"g1_1_ood_pressure_score={metrics['g1_1_ood_pressure_score']:.3f}")
    print(f"feedback_pressure_gain={metrics['feedback_pressure_gain']:.3f}")
    print(f"compression_pressure_gain={metrics['compression_pressure_gain']:.3f}")
    print(f"generated_rule_pressure_coverage={metrics['generated_rule_pressure_coverage']}")


if __name__ == "__main__":
    main()
