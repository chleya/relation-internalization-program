from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .b6_2_hardening.runner import run_b6_2_hardening, write_b6_2_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b6_2_hardening.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records, metrics = run_b6_2_hardening(config, seed=args.seed)
    write_b6_2_outputs(summary, records, metrics)
    print(f"best_b6_2_hardening_score={max(float(row.get('b62_score', 0.0)) for row in summary):.3f}")
    print(f"mean_b6_2_hardening_score={metrics['b62_policy_mean_score']:.3f}")


if __name__ == "__main__":
    main()

